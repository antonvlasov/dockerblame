# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=3.0.0",
#     "plotly>=6.5.2",
# ]
# ///
"""Docker Cgroup Resource Usage grapher.

Reads CSV produced by the Go collector and generates an interactive
HTML report with stacked area charts for CPU, memory, disk IO,
and network IO.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate interactive HTML report from "
            "dockerblame output. Reads cgroup_stats.csv "
            "and events.csv from the given directory "
            "and writes cgroup_report.html into it."
        ),
    )
    parser.add_argument(
        "dir",
        type=Path,
        help="Directory with cgroup_stats.csv and events.csv",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Rolling average window for CPU chart (default: 10)",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """Load and validate the collector CSV."""
    df = pd.read_csv(path)
    required: list[str] = [
        "timestamp_ms",
        "cgroup_name",
        "cpu_usage_usec",
        "memory_bytes",
        "disk_read_bytes",
        "disk_write_bytes",
        "net_rx_bytes",
        "net_tx_bytes",
    ]
    missing: list[str] = [
        c for c in required if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}"
        )
    return df


def compute_cpu_percent(
    df: pd.DataFrame,
    smooth_window: int,
) -> pd.DataFrame:
    """Compute CPU usage as percentage of one core.

    For each cgroup, computes delta of cumulative usage_usec
    divided by delta time, expressed as a percentage.
    Applies rolling average smoothing.
    """
    df = df.sort_values(["cgroup_name", "timestamp_ms"])
    records: list[dict[str, object]] = []

    for name, group in df.groupby("cgroup_name"):
        group = group.reset_index(drop=True)
        delta_usec: pd.Series = group["cpu_usage_usec"].diff()  # type: ignore[assignment]
        delta_ms: pd.Series = group["timestamp_ms"].diff()  # type: ignore[assignment]

        # CPU rate as fraction of one core, then to percent
        cpu_pct: pd.Series = (delta_usec / (delta_ms * 1000)) * 100  # type: ignore[assignment]
        cpu_pct = cpu_pct.clip(lower=0)

        if smooth_window > 1:
            cpu_pct = cpu_pct.rolling(
                window=smooth_window, min_periods=1
            ).mean()

        for i in range(1, len(group)):
            records.append({
                "timestamp_ms": group["timestamp_ms"].iloc[i],
                "cgroup_name": name,
                "cpu_pct": cpu_pct.iloc[i],
            })

    return pd.DataFrame(records)


def compute_io_rate(
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
    smooth_window: int,
) -> pd.DataFrame:
    """Compute IO rate in MB/s from cumulative byte counters.

    Works for disk_read_bytes, disk_write_bytes,
    net_rx_bytes, net_tx_bytes.
    """
    df = df.sort_values(["cgroup_name", "timestamp_ms"])
    records: list[dict[str, object]] = []

    for name, group in df.groupby("cgroup_name"):
        group = group.reset_index(drop=True)
        delta_bytes: pd.Series = group[value_col].diff()  # type: ignore[assignment]
        delta_ms: pd.Series = group["timestamp_ms"].diff()  # type: ignore[assignment]

        # bytes/ms -> MB/s: divide by 1000 to get bytes/s,
        # then by 1024*1024 to get MB/s
        rate: pd.Series = (  # type: ignore[assignment]
            delta_bytes / delta_ms * 1000
            / (1024 * 1024)
        )
        rate = rate.clip(lower=0)

        if smooth_window > 1:
            rate = rate.rolling(
                window=smooth_window, min_periods=1
            ).mean()

        for i in range(1, len(group)):
            records.append({
                "timestamp_ms": group["timestamp_ms"].iloc[i],
                "cgroup_name": name,
                out_col: rate.iloc[i],
            })

    return pd.DataFrame(records)


def load_events(path: Path) -> pd.DataFrame | None:
    """Load events CSV. Returns None if file missing."""
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    required: list[str] = [
        "timestamp_ms", "container_name", "action",
    ]
    missing: list[str] = [
        c for c in required if c not in df.columns
    ]
    if missing:
        print(
            f"Events CSV missing columns: {missing},"
            " skipping events"
        )
        return None
    if df.empty:
        return None
    return df


def _compute_band_midpoints(
    df: pd.DataFrame,
    value_col: str,
    cgroup_names: list[str],
) -> dict[str, pd.DataFrame]:
    """Compute vertical midpoints of stacked bands.

    Returns a dict mapping cgroup_name to a DataFrame with
    columns: timestamp_ms, midpoint, actual_value.
    """
    pivot: pd.DataFrame = df.pivot_table(
        index="timestamp_ms",
        columns="cgroup_name",
        values=value_col,
        fill_value=0,
    )
    pivot = pivot.reindex(
        columns=cgroup_names, fill_value=0
    )
    cumsum: pd.DataFrame = pivot.cumsum(axis=1)

    result: dict[str, pd.DataFrame] = {}
    for i, name in enumerate(cgroup_names):
        bottom = cumsum.iloc[:, i - 1] if i > 0 else 0
        top = cumsum.iloc[:, i]
        mid = (bottom + top) / 2
        result[name] = pd.DataFrame({
            "timestamp_ms": pivot.index,
            "midpoint": mid.values,
            "actual_value": pivot[name].values,
        })
    return result


# (row, stackgroup, value_col, y_label, hover_suffix)
_SUBPLOT_CFG: list[
    tuple[int, str, str, str, str]
] = [
    (1, "cpu", "cpu_pct", "CPU %", "% CPU"),
    (2, "mem", "mem_mb", "MB", " MB"),
    (3, "dr", "disk_read_mbps", "MB/s", " MB/s"),
    (4, "dw", "disk_write_mbps", "MB/s", " MB/s"),
    (5, "nr", "net_rx_mbps", "MB/s", " MB/s"),
    (6, "nt", "net_tx_mbps", "MB/s", " MB/s"),
]


def _build_color_map(
    cgroup_names: list[str],
) -> dict[str, str]:
    """Assign a stable color to each service name."""
    palette = pc.qualitative.Plotly
    return {
        name: palette[i % len(palette)]
        for i, name in enumerate(cgroup_names)
    }


def _add_stacked_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    value_col: str,
    stackgroup: str,
    row: int,
    cgroup_names: list[str],
    show_legend: bool,
    color_map: dict[str, str],
) -> None:
    """Add visible stacked area traces (no hover)."""
    for name in cgroup_names:
        subset = df[df["cgroup_name"] == name]
        if subset.empty:
            continue
        ts = pd.to_datetime(
            subset["timestamp_ms"], unit="ms"
        )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=subset[value_col],
                name=name,
                legendgroup=name,
                stackgroup=stackgroup,
                showlegend=show_legend,
                hoverinfo="skip",
                line=dict(color=color_map[name]),
                fillcolor=color_map[name],
            ),
            row=row,
            col=1,
        )


def _add_hover_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    value_col: str,
    hover_suffix: str,
    row: int,
    cgroup_names: list[str],
) -> None:
    """Add invisible midpoint traces for hover."""
    mids = _compute_band_midpoints(
        df, value_col, cgroup_names
    )
    for name in cgroup_names:
        if name not in mids:
            continue
        mid = mids[name]
        ts = pd.to_datetime(
            mid["timestamp_ms"], unit="ms"
        )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=mid["midpoint"],
                customdata=mid[
                    "actual_value"
                ].values.reshape(-1, 1),
                mode="lines",
                line=dict(
                    width=0,
                    color="rgba(0,0,0,0)",
                ),
                legendgroup=name,
                showlegend=False,
                hovertemplate=(
                    "%{x}<br>"
                    f"{name}: "
                    "%{customdata[0]:.2f}"
                    f"{hover_suffix}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def _add_event_traces(
    fig: go.Figure,
    events_df: pd.DataFrame,
    n_rows: int,
    color_map: dict[str, str],
) -> None:
    """Add vertical line traces for docker events.

    Grouped by container name so each service is a
    separate legend item.  Event type is shown as a
    text label at the top of each vertical line.
    Uses secondary y-axes (range [0,1]) so the lines
    span the full subplot height without distorting
    the primary data axis.  All traces start hidden
    (visible='legendonly').
    """
    services: list[str] = sorted(
        events_df["container_name"].unique()
    )
    first_service = True

    for svc in services:
        group = events_df[
            events_df["container_name"] == svc
        ].sort_values("timestamp_ms")

        color = color_map.get(svc, "#7f7f7f")
        legend_name = f"Events: {svc}"

        xs: list[object] = []
        ys: list[object] = []
        labels: list[str] = []
        hovers: list[str | None] = []

        for _, row in group.iterrows():
            ts = pd.to_datetime(
                row["timestamp_ms"], unit="ms",
            )
            action = str(row["action"])
            xs.extend([ts, ts, None])
            ys.extend([0, 0.85, None])
            labels.extend(["", action, ""])
            hover = f"{svc}: {action}"
            hovers.extend([hover, hover, None])

        for r in range(1, n_rows + 1):
            show = r == 1
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+text",
                    line=dict(
                        color=color,
                        width=1,
                        dash="dot",
                    ),
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=8, color=color),
                    cliponaxis=False,
                    name=legend_name,
                    legendgroup=f"event_{svc}",
                    legendgrouptitle_text=(
                        "Events"
                        if show and first_service
                        else None
                    ),
                    showlegend=show,
                    visible="legendonly",
                    hoverinfo="text",
                    hovertext=hovers,
                ),
                row=r,
                col=1,
                secondary_y=True,
            )
        first_service = False


def build_figure(
    datasets: dict[str, pd.DataFrame],
    events_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Build a plotly figure with 6 subplot rows."""
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[[{"secondary_y": True}]] * 6,
        subplot_titles=(
            "CPU Usage (%)",
            "Memory Usage (MB)",
            "Disk Read (MB/s)",
            "Disk Write (MB/s)",
            "Net RX (MB/s)",
            "Net TX (MB/s)",
        ),
    )

    all_names: set[str] = set()
    for df in datasets.values():
        all_names |= set(df["cgroup_name"].unique())
    cgroup_names: list[str] = sorted(all_names)
    color_map = _build_color_map(cgroup_names)

    for row, sg, vcol, ylabel, hsuf in _SUBPLOT_CFG:
        df = datasets[vcol]
        _add_stacked_traces(
            fig, df, vcol, sg, row,
            cgroup_names, show_legend=(row == 1),
            color_map=color_map,
        )
        _add_hover_traces(
            fig, df, vcol, hsuf, row, cgroup_names,
        )
        fig.update_yaxes(
            title_text=ylabel, row=row, col=1
        )

    if events_df is not None:
        _add_event_traces(
            fig, events_df, n_rows=6,
            color_map=color_map,
        )

    # Hide secondary y-axes used for event lines.
    for r in range(1, 7):
        fig.update_yaxes(
            range=[0, 1],
            fixedrange=True,
            visible=False,
            row=r,
            col=1,
            secondary_y=True,
        )

    fig.update_layout(
        title_text="Docker Cgroup Resource Usage",
        height=1600,
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Time", row=6, col=1)

    return fig


def main() -> None:
    """Entry point: load data, compute metrics, write HTML."""
    args = parse_args()

    input_path: Path = args.dir / "cgroup_stats.csv"
    events_path: Path = args.dir / "events.csv"
    output_path: Path = args.dir / "cgroup_report.html"

    df: pd.DataFrame = load_csv(input_path)
    print(
        f"Loaded {len(df)} rows, "
        f"{df['cgroup_name'].nunique()} cgroup(s)"
    )

    cpu_df: pd.DataFrame = compute_cpu_percent(
        df, args.smooth_window
    )

    mem_df: pd.DataFrame = df[
        ["timestamp_ms", "cgroup_name", "memory_bytes"]
    ].copy()
    mem_df["mem_mb"] = (
        mem_df["memory_bytes"] / (1024 * 1024)
    )

    disk_read_df = compute_io_rate(
        df, "disk_read_bytes",
        "disk_read_mbps", args.smooth_window,
    )
    disk_write_df = compute_io_rate(
        df, "disk_write_bytes",
        "disk_write_mbps", args.smooth_window,
    )
    net_rx_df = compute_io_rate(
        df, "net_rx_bytes",
        "net_rx_mbps", args.smooth_window,
    )
    net_tx_df = compute_io_rate(
        df, "net_tx_bytes",
        "net_tx_mbps", args.smooth_window,
    )

    datasets: dict[str, pd.DataFrame] = {
        "cpu_pct": cpu_df,
        "mem_mb": mem_df,
        "disk_read_mbps": disk_read_df,
        "disk_write_mbps": disk_write_df,
        "net_rx_mbps": net_rx_df,
        "net_tx_mbps": net_tx_df,
    }

    events_df = load_events(events_path)
    if events_df is not None:
        print(
            f"Loaded {len(events_df)} event(s)"
        )

    fig: go.Figure = build_figure(datasets, events_df)
    fig.write_html(str(output_path))
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
