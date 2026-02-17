package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"go.uber.org/zap"
)

// logger is the global zap logger for the collector.
var logger *zap.SugaredLogger

// initLogger sets up the global zap logger named
// "[collector]".
func initLogger() {
	cfg := zap.NewProductionConfig()
	cfg.OutputPaths = []string{"stderr"}
	base, err := cfg.Build()
	if err != nil {
		panic(fmt.Sprintf("init logger: %v", err))
	}
	logger = base.Named("[collector]").Sugar()
}

// cgroupInfo holds resolved metadata for a single cgroup.
type cgroupInfo struct {
	path string // full path to cgroup directory
	name string // human-readable name
	pid  int    // container init PID (for /proc net)
}

// containerMeta holds resolved name and PID from docker
// inspect, used for caching.
type containerMeta struct {
	name string
	pid  int
}

// eventActions lists container lifecycle events worth
// recording. Health-check and exec events are excluded
// to avoid noise.
var eventActions = map[string]bool{
	"create":  true,
	"start":   true,
	"stop":    true,
	"die":     true,
	"kill":    true,
	"restart": true,
	"pause":   true,
	"unpause": true,
	"destroy": true,
}

// dockerEvent represents a single docker event JSON line
// emitted by "docker events --format '{{json .}}'".
type dockerEvent struct {
	TimeNano int64  `json:"timeNano"`
	Action   string `json:"Action"`
	Actor    struct {
		Attributes map[string]string `json:"Attributes"`
	} `json:"Actor"`
}

// dockerInspect calls docker inspect to get the container
// name and init PID.
func dockerInspect(
	containerID string,
) (containerMeta, error) {
	cmd := exec.Command(
		"docker", "inspect",
		"--format", "{{.Name}} {{.State.Pid}}",
		containerID,
	)
	out, err := cmd.Output()
	if err != nil {
		return containerMeta{}, fmt.Errorf(
			"docker inspect: %w", err,
		)
	}
	parts := strings.Fields(
		strings.TrimSpace(string(out)),
	)
	if len(parts) < 2 {
		return containerMeta{}, fmt.Errorf(
			"unexpected docker inspect output: %s",
			string(out),
		)
	}
	name := strings.TrimPrefix(parts[0], "/")
	if name == "" {
		return containerMeta{}, fmt.Errorf(
			"empty name from docker inspect",
		)
	}
	pid, err := strconv.Atoi(parts[1])
	if err != nil {
		return containerMeta{}, fmt.Errorf(
			"parse pid: %w", err,
		)
	}
	return containerMeta{name: name, pid: pid}, nil
}

// resolveMeta extracts the container ID from a
// docker-<id>.scope directory name and resolves it to a
// human-readable name and PID via docker inspect.
// Results are cached; use refreshPID to update a stale
// PID when /proc/<pid>/net/dev becomes unreachable.
func resolveMeta(
	dirName string,
	dirPath string,
	cache map[string]containerMeta,
) (containerMeta, error) {
	if cached, ok := cache[dirPath]; ok {
		return cached, nil
	}

	// dirName format: docker-<64hex>.scope
	trimmed := strings.TrimPrefix(dirName, "docker-")
	trimmed = strings.TrimSuffix(trimmed, ".scope")
	if len(trimmed) < 12 {
		return containerMeta{}, fmt.Errorf(
			"unexpected cgroup dir format: %s", dirName,
		)
	}
	shortID := trimmed[:12]

	meta, err := dockerInspect(trimmed)
	if err != nil {
		logger.Warnf(
			"docker inspect %s: %v, using short ID",
			shortID, err,
		)
		meta = containerMeta{name: shortID, pid: 0}
	}

	cache[dirPath] = meta
	return meta, nil
}

// refreshPID re-inspects a container to get its current
// PID. Called when readNetIO fails with a stale PID.
// Updates both the cache and the cgroupInfo in place.
func refreshPID(
	cg *cgroupInfo,
	cache map[string]containerMeta,
) {
	dirName := filepath.Base(cg.path)
	trimmed := strings.TrimPrefix(dirName, "docker-")
	trimmed = strings.TrimSuffix(trimmed, ".scope")
	if len(trimmed) < 12 {
		return
	}

	meta, err := dockerInspect(trimmed)
	if err != nil {
		logger.Warnf(
			"refresh pid for %s: %v", cg.name, err,
		)
		cg.pid = 0
		return
	}

	cg.pid = meta.pid
	cached := cache[cg.path]
	cached.pid = meta.pid
	cache[cg.path] = cached
}

// discoverCgroups scans cgroupBase for docker-*.scope dirs
// and docker.service, returning a slice of cgroupInfo.
// metaCache maps cgroup dir path -> resolved metadata so
// we don't call docker inspect repeatedly.
func discoverCgroups(
	cgroupBase string,
	metaCache map[string]containerMeta,
) ([]cgroupInfo, error) {
	entries, err := os.ReadDir(cgroupBase)
	if err != nil {
		return nil, fmt.Errorf(
			"read cgroup base %s: %w", cgroupBase, err,
		)
	}

	var cgroups []cgroupInfo

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dirName := e.Name()
		dirPath := filepath.Join(cgroupBase, dirName)

		if dirName == "docker.service" {
			metaCache[dirPath] = containerMeta{
				name: "[dockerd]", pid: 0,
			}
			cgroups = append(cgroups, cgroupInfo{
				path: dirPath,
				name: "[dockerd]",
				pid:  0,
			})
			continue
		}

		if !strings.HasPrefix(dirName, "docker-") ||
			!strings.HasSuffix(dirName, ".scope") {
			continue
		}

		meta, err := resolveMeta(
			dirName, dirPath, metaCache,
		)
		if err != nil {
			logger.Warnf(
				"resolve meta for %s: %v",
				dirName, err,
			)
			continue
		}
		cgroups = append(cgroups, cgroupInfo{
			path: dirPath,
			name: meta.name,
			pid:  meta.pid,
		})
	}

	return cgroups, nil
}

// rescanLoop re-discovers cgroups every 100ms, updating
// the shared slice under a write lock. Runs until done is
// closed.
func rescanLoop(
	cgroupBase string,
	metaCache map[string]containerMeta,
	mu *sync.RWMutex,
	cgroups *[]cgroupInfo,
	done <-chan struct{},
) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			updated, err := discoverCgroups(
				cgroupBase, metaCache,
			)
			if err != nil {
				logger.Warnf("rediscovery: %v", err)
				continue
			}

			mu.Lock()
			prev := len(*cgroups)
			*cgroups = updated
			mu.Unlock()

			if len(updated) != prev {
				logger.Infof(
					"cgroup count changed: %d -> %d",
					prev, len(updated),
				)
			}
		}
	}
}

// readCPUUsage reads cpu.stat from the given cgroup path
// and returns the usage_usec value.
func readCPUUsage(cgroupPath string) (int64, error) {
	data, err := os.ReadFile(
		filepath.Join(cgroupPath, "cpu.stat"),
	)
	if err != nil {
		return 0, fmt.Errorf("read cpu.stat: %w", err)
	}

	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "usage_usec ") {
			parts := strings.Fields(line)
			if len(parts) != 2 {
				return 0, fmt.Errorf(
					"unexpected cpu.stat line: %s",
					line,
				)
			}
			v, err := strconv.ParseInt(
				parts[1], 10, 64,
			)
			if err != nil {
				return 0, fmt.Errorf(
					"parse usage_usec: %w", err,
				)
			}
			return v, nil
		}
	}

	return 0, fmt.Errorf(
		"usage_usec not found in cpu.stat",
	)
}

// readMemory reads memory.current and subtracts
// inactive_file (from memory.stat) to match the
// behavior of docker stats.
func readMemory(cgroupPath string) (int64, error) {
	data, err := os.ReadFile(
		filepath.Join(cgroupPath, "memory.current"),
	)
	if err != nil {
		return 0, fmt.Errorf(
			"read memory.current: %w", err,
		)
	}

	total, err := strconv.ParseInt(
		strings.TrimSpace(string(data)), 10, 64,
	)
	if err != nil {
		return 0, fmt.Errorf(
			"parse memory.current: %w", err,
		)
	}

	inactive := readInactiveFile(cgroupPath)
	mem := total - inactive
	if mem < 0 {
		mem = 0
	}
	return mem, nil
}

// readInactiveFile parses inactive_file from
// memory.stat. Returns 0 if the file is unreadable
// or the field is missing.
func readInactiveFile(cgroupPath string) int64 {
	data, err := os.ReadFile(
		filepath.Join(cgroupPath, "memory.stat"),
	)
	if err != nil {
		logger.Warnf(
			"read memory.stat: %v", err,
		)
		return 0
	}

	for _, line := range strings.Split(
		string(data), "\n",
	) {
		parts := strings.Fields(line)
		if len(parts) == 2 &&
			parts[0] == "inactive_file" {
			v, err := strconv.ParseInt(
				parts[1], 10, 64,
			)
			if err != nil {
				logger.Warnf(
					"parse inactive_file: %v",
					err,
				)
				return 0
			}
			return v
		}
	}
	return 0
}

// readDiskIO reads io.stat from the cgroup and returns
// cumulative read and write bytes summed across all block
// devices.
func readDiskIO(
	cgroupPath string,
) (int64, int64, error) {
	data, err := os.ReadFile(
		filepath.Join(cgroupPath, "io.stat"),
	)
	if err != nil {
		return 0, 0, fmt.Errorf(
			"read io.stat: %w", err,
		)
	}

	var totalRead, totalWrite int64
	for _, line := range strings.Split(
		string(data), "\n",
	) {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		for _, field := range strings.Fields(line) {
			if v, ok := strings.CutPrefix(
				field, "rbytes=",
			); ok {
				n, err := strconv.ParseInt(
					v, 10, 64,
				)
				if err == nil {
					totalRead += n
				}
			}
			if v, ok := strings.CutPrefix(
				field, "wbytes=",
			); ok {
				n, err := strconv.ParseInt(
					v, 10, 64,
				)
				if err == nil {
					totalWrite += n
				}
			}
		}
	}
	return totalRead, totalWrite, nil
}

// readNetIO reads /proc/<pid>/net/dev and returns
// cumulative RX and TX bytes summed across all
// interfaces except lo.
func readNetIO(pid int) (int64, int64, error) {
	if pid == 0 {
		return 0, 0, nil
	}

	path := fmt.Sprintf("/proc/%d/net/dev", pid)
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, fmt.Errorf(
			"read net/dev: %w", err,
		)
	}

	var totalRX, totalTX int64
	for _, line := range strings.Split(
		string(data), "\n",
	) {
		line = strings.TrimSpace(line)
		if line == "" || !strings.Contains(line, ":") {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		iface := strings.TrimSpace(parts[0])
		if iface == "lo" {
			continue
		}
		fields := strings.Fields(parts[1])
		// fields[0]=rx_bytes, fields[8]=tx_bytes
		if len(fields) < 10 {
			continue
		}
		rx, err := strconv.ParseInt(
			fields[0], 10, 64,
		)
		if err == nil {
			totalRX += rx
		}
		tx, err := strconv.ParseInt(
			fields[8], 10, 64,
		)
		if err == nil {
			totalTX += tx
		}
	}
	return totalRX, totalTX, nil
}

// eventsLoop runs "docker events" and writes container
// lifecycle events to a CSV file until done is closed.
func eventsLoop(
	eventsPath string,
	done <-chan struct{},
) {
	f, err := os.Create(eventsPath)
	if err != nil {
		logger.Warnf("create events file: %v", err)
		return
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()
	fmt.Fprintln(
		w, "timestamp_ms,container_name,action",
	)

	cmd := exec.Command(
		"docker", "events",
		"--filter", "type=container",
		"--format", "{{json .}}",
	)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logger.Warnf("docker events pipe: %v", err)
		return
	}
	if err := cmd.Start(); err != nil {
		logger.Warnf("docker events start: %v", err)
		return
	}
	logger.Info("docker events listener started")

	go func() {
		<-done
		_ = cmd.Process.Kill()
	}()

	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		var ev dockerEvent
		if err := json.Unmarshal(
			scanner.Bytes(), &ev,
		); err != nil {
			logger.Warnf(
				"parse docker event: %v", err,
			)
			continue
		}

		if !eventActions[ev.Action] {
			continue
		}

		name := ev.Actor.Attributes["name"]
		if name == "" {
			continue
		}

		tsMs := ev.TimeNano / 1_000_000
		fmt.Fprintf(
			w, "%d,%s,%s\n",
			tsMs, name, ev.Action,
		)
		_ = w.Flush()

		logger.Infof(
			"event: %s %s", name, ev.Action,
		)
	}

	if err := scanner.Err(); err != nil {
		logger.Warnf("docker events scanner: %v", err)
	}
	_ = cmd.Wait()
}

// sampleAll snapshots the cgroup list under a read lock
// and sequentially reads CPU and memory for each cgroup.
func sampleAll(
	mu *sync.RWMutex,
	cgroups *[]cgroupInfo,
	w *bufio.Writer,
	metaCache map[string]containerMeta,
) {
	mu.RLock()
	snapshot := make([]cgroupInfo, len(*cgroups))
	copy(snapshot, *cgroups)
	mu.RUnlock()

	tsMs := time.Now().UnixMilli()
	for i := range snapshot {
		cg := &snapshot[i]
		cpu, err := readCPUUsage(cg.path)
		if err != nil {
			logger.Warnf(
				"sample %s cpu: %v", cg.name, err,
			)
			continue
		}
		mem, err := readMemory(cg.path)
		if err != nil {
			logger.Warnf(
				"sample %s mem: %v", cg.name, err,
			)
			continue
		}

		var diskR, diskW int64
		diskR, diskW, err = readDiskIO(cg.path)
		if err != nil {
			logger.Warnf(
				"sample %s disk: %v",
				cg.name, err,
			)
		}

		var netRX, netTX int64
		netRX, netTX, err = readNetIO(cg.pid)
		if err != nil {
			// PID may be stale after container restart.
			refreshPID(cg, metaCache)
			netRX, netTX, err = readNetIO(cg.pid)
			if err != nil {
				logger.Warnf(
					"sample %s net: %v",
					cg.name, err,
				)
			}
		}

		fmt.Fprintf(
			w, "%d,%s,%d,%d,%d,%d,%d,%d\n",
			tsMs, cg.name, cpu, mem,
			diskR, diskW, netRX, netTX,
		)
	}
}

// run is the main loop: discovers cgroups, starts the
// rescan goroutine, samples with a worker pool, flushes
// periodically, and exits cleanly on SIGINT/SIGTERM.
func run(
	interval time.Duration,
	outputPath string,
	cgroupBase string,
	eventsPath string,
) error {
	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	fmt.Fprintln(
		w,
		"timestamp_ms,cgroup_name,"+
			"cpu_usage_usec,memory_bytes,"+
			"disk_read_bytes,disk_write_bytes,"+
			"net_rx_bytes,net_tx_bytes",
	)

	metaCache := make(map[string]containerMeta)
	cgroups, err := discoverCgroups(cgroupBase, metaCache)
	if err != nil {
		return fmt.Errorf("initial discovery: %w", err)
	}
	logger.Infof("discovered %d cgroup(s)", len(cgroups))
	for _, cg := range cgroups {
		logger.Infof("  %s (%s)", cg.name, cg.path)
	}

	var mu sync.RWMutex
	done := make(chan struct{})
	go rescanLoop(
		cgroupBase, metaCache, &mu, &cgroups, done,
	)
	go eventsLoop(eventsPath, done)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	sampleTicker := time.NewTicker(interval)
	defer sampleTicker.Stop()

	flushTicker := time.NewTicker(1 * time.Second)
	defer flushTicker.Stop()

	logger.Infof(
		"collecting every %v, writing to %s "+
			"(Ctrl+C to stop)",
		interval, outputPath,
	)

	for {
		select {
		case <-sigCh:
			logger.Info(
				"signal received, flushing and exiting",
			)
			close(done)
			if err := w.Flush(); err != nil {
				return fmt.Errorf("final flush: %w", err)
			}
			return nil

		case <-sampleTicker.C:
			sampleAll(&mu, &cgroups, w, metaCache)

		case <-flushTicker.C:
			if err := w.Flush(); err != nil {
				logger.Warnf("flush: %v", err)
			}
		}
	}
}

func main() {
	initLogger()
	defer logger.Sync() //nolint:errcheck

	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			"Dockerblame: sample Docker container "+
				"cgroup resource usage.\n\n"+
				"Output is written to a timestamped "+
				"subfolder (dockerblame_YYYY-MM-DD_"+
				"HH-MM-SS/).\n\n"+
				"Usage: dockerblame [flags]\n\n"+
				"Flags:\n",
		)
		flag.PrintDefaults()
	}

	interval := flag.Duration(
		"interval", 100*time.Millisecond,
		"Sampling interval",
	)
	cgroupBase := flag.String(
		"cgroupbase",
		"/sys/fs/cgroup/system.slice",
		"Base cgroup path",
	)
	flag.Parse()

	dir := fmt.Sprintf(
		"dockerblame_%s",
		time.Now().Format("2006-01-02_15-04-05"),
	)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		logger.Fatalf("create output dir: %v", err)
	}
	logger.Infof("output directory: %s", dir)

	outputPath := filepath.Join(dir, "cgroup_stats.csv")
	eventsPath := filepath.Join(dir, "events.csv")

	if err := run(
		*interval, outputPath,
		*cgroupBase, eventsPath,
	); err != nil {
		logger.Fatalf("error: %v", err)
	}
}
