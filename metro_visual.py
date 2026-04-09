from pathlib import Path
import re

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


LINE_COLORS = {
    "Red line": "#d62728",
    "Blue line": "#1f77b4",
    "Blue line branch": "#5fa2dd",
    "Yellow line": "#f1c40f",
    "Green line": "#2ca02c",
    "Green line branch": "#79c779",
    "Pink line": "#e377c2",
    "Magenta line": "#b84dff",
    "Orange line": "#ff7f0e",
    "Gray line": "#7f7f7f",
    "Aqua line": "#17becf",
    "Rapid Metro": "#8c564b",
    "Voilet line": "#9467bd",
}


def normalize_station_name(name: str) -> str:
    """Normalize names so interchanges across lines map to one station node."""
    base = re.sub(r"\[.*?\]", "", str(name))
    base = re.sub(r"\s+", " ", base).strip().lower()
    return base


def clean_label(name: str) -> str:
    base = re.sub(r"\[.*?\]", "", str(name)).strip()
    return re.sub(r"\s+", " ", base)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ID (Station ID)"] = pd.to_numeric(df["ID (Station ID)"], errors="coerce")
    df["Dist. From First Station(km)"] = pd.to_numeric(df["Dist. From First Station(km)"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["ID (Station ID)", "Latitude", "Longitude", "Station Names", "Metro Line"])

    # Keep only Delhi-like coordinates to avoid outliers distorting the plot.
    # A few rows in the source file contain invalid longitude values.
    df = df[df["Latitude"].between(28.0, 29.5) & df["Longitude"].between(76.5, 78.5)]
    return df


def build_station_graph(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()

    station_centroids = (
        df.assign(station_key=df["Station Names"].map(normalize_station_name))
        .groupby("station_key", as_index=False)
        .agg(
            display_name=("Station Names", "first"),
            latitude=("Latitude", "mean"),
            longitude=("Longitude", "mean"),
            lines=("Metro Line", lambda values: sorted(set(values))),
        )
    )

    for _, row in station_centroids.iterrows():
        graph.add_node(
            row["station_key"],
            label=row["display_name"],
            lat=row["latitude"],
            lon=row["longitude"],
            lines=row["lines"],
        )

    # Build route edges within each line using the station sequence in the dataset.
    for line_name, line_df in df.groupby("Metro Line"):
        ordered = line_df.sort_values("Dist. From First Station(km)")
        ordered_keys = ordered["Station Names"].map(normalize_station_name).tolist()
        for src, dst in zip(ordered_keys, ordered_keys[1:]):
            if src == dst:
                continue
            if graph.has_edge(src, dst):
                graph[src][dst]["lines"].add(line_name)
            else:
                graph.add_edge(src, dst, lines={line_name})

    return graph


def visualize(df: pd.DataFrame, graph: nx.Graph) -> None:
    pagerank = nx.pagerank(graph, alpha=0.9)
    top_stations = sorted(pagerank.items(), key=lambda pair: pair[1], reverse=True)[:12]

    fig = plt.figure(figsize=(20, 10))
    grid = fig.add_gridspec(
        2, 3, width_ratios=[1.9, 1.7, 1.2], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.24
    )
    ax_map = fig.add_subplot(grid[:, 0])
    ax_conn = fig.add_subplot(grid[:, 1])
    ax_rank = fig.add_subplot(grid[0, 2])
    ax_line = fig.add_subplot(grid[1, 2])

    # Plot lines with moderate alpha to avoid clutter in dense areas.
    for line_name, line_df in df.groupby("Metro Line"):
        ordered = line_df.sort_values("Dist. From First Station(km)")
        color = LINE_COLORS.get(line_name, "#4f4f4f")
        ax_map.plot(
            ordered["Longitude"],
            ordered["Latitude"],
            color=color,
            linewidth=1.8,
            alpha=0.45,
            zorder=1,
        )

    nodes = list(graph.nodes())
    x = [graph.nodes[node]["lon"] for node in nodes]
    y = [graph.nodes[node]["lat"] for node in nodes]
    colors = [pagerank[node] for node in nodes]
    sizes = [140 + pagerank[node] * 22000 for node in nodes]

    scatter = ax_map.scatter(
        x, y, c=colors, s=sizes, cmap="plasma", edgecolor="white", linewidth=0.6, zorder=2
    )

    # Label only top stations and spread label offsets for readability.
    label_offsets = [(6, 6), (-42, 8), (8, -13), (12, 12), (-56, -4), (8, 14)]
    for idx, (station_key, _) in enumerate(top_stations[:6]):
        dx, dy = label_offsets[idx % len(label_offsets)]
        ax_map.annotate(
            clean_label(graph.nodes[station_key]["label"]),
            (graph.nodes[station_key]["lon"], graph.nodes[station_key]["lat"]),
            fontsize=7,
            xytext=(dx, dy),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.65, "ec": "none"},
        )

    ax_map.set_title("Delhi Metro Network (Connectivity Map)", fontsize=12, pad=8)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(alpha=0.2, linestyle="--")
    ax_map.set_aspect("equal", adjustable="box")

    # Tighten map limits with small padding.
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    ax_map.set_xlim(x_min - 0.02, x_max + 0.02)
    ax_map.set_ylim(y_min - 0.02, y_max + 0.02)

    cbar = fig.colorbar(scatter, ax=ax_map, fraction=0.03, pad=0.02)
    cbar.set_label("PageRank Score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Connectivity topology panel (non-geographic network view).
    pos_topology = nx.spring_layout(graph, seed=42, k=0.33, iterations=80)
    topo_sizes = [90 + pagerank[node] * 12000 for node in nodes]
    topo_colors = [pagerank[node] for node in nodes]

    nx.draw_networkx_edges(
        graph,
        pos_topology,
        ax=ax_conn,
        edge_color="#A0A0A0",
        alpha=0.25,
        width=0.7,
    )
    nx.draw_networkx_nodes(
        graph,
        pos_topology,
        ax=ax_conn,
        node_size=topo_sizes,
        node_color=topo_colors,
        cmap="plasma",
        edgecolors="white",
        linewidths=0.45,
    )

    for station_key, _ in top_stations[:5]:
        ax_conn.text(
            pos_topology[station_key][0],
            pos_topology[station_key][1],
            clean_label(graph.nodes[station_key]["label"]),
            fontsize=7,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.6, "ec": "none"},
        )

    ax_conn.set_title("Metro Connectivity Topology (Graph View)", fontsize=12, pad=8)
    ax_conn.set_axis_off()

    rank_labels = [clean_label(graph.nodes[station]["label"]) for station, _ in top_stations]
    rank_scores = [score for _, score in top_stations]
    ax_rank.barh(rank_labels[::-1], rank_scores[::-1], color="#6f63d9", alpha=0.92)
    ax_rank.set_title("Top Stations by PageRank", fontsize=11, pad=8)
    ax_rank.set_xlabel("PageRank Score")
    ax_rank.tick_params(axis="y", labelsize=8)
    ax_rank.tick_params(axis="x", labelsize=8)
    ax_rank.grid(axis="x", alpha=0.2, linestyle="--")

    line_counts = df.groupby("Metro Line")["Station Names"].nunique().sort_values(ascending=False).head(8)
    bar_colors = [LINE_COLORS.get(line, "#4f4f4f") for line in line_counts.index]
    ax_line.barh(line_counts.index[::-1], line_counts.values[::-1], color=bar_colors[::-1], alpha=0.9)
    ax_line.set_title("Stations per Metro Line (Top 8)", fontsize=11, pad=8)
    ax_line.set_xlabel("Unique Stations")
    ax_line.tick_params(axis="y", labelsize=8)
    ax_line.tick_params(axis="x", labelsize=8)
    ax_line.grid(axis="x", alpha=0.2, linestyle="--")

    total_lines = df["Metro Line"].nunique()
    total_stations = len(nodes)
    top_name = clean_label(graph.nodes[top_stations[0][0]]["label"])
    fig.suptitle(
        f"Metro Network Dashboard  |  Stations: {total_stations}  |  Lines: {total_lines}  |  Top Hub: {top_name}",
        fontsize=13,
        y=0.98,
    )

    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.07, top=0.93)
    output_path = Path(__file__).with_name("metro_dashboard.png")
    plt.savefig(output_path, dpi=220)
    print(f"Saved dashboard: {output_path}")
    plt.show()


def main() -> None:
    csv_path = Path(__file__).with_name("metro.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    metro_df = load_data(csv_path)
    metro_graph = build_station_graph(metro_df)
    visualize(metro_df, metro_graph)


if __name__ == "__main__":
    main()