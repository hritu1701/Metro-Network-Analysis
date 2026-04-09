from pathlib import Path
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def normalize_station_name(name: str) -> str:
    base = re.sub(r"\[.*?\]", "", str(name))
    return re.sub(r"\s+", " ", base).strip().lower()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "ID (Station ID)",
        "Dist. From First Station(km)",
        "Latitude",
        "Longitude",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(
        subset=[
            "ID (Station ID)",
            "Dist. From First Station(km)",
            "Station Names",
            "Metro Line",
            "Latitude",
            "Longitude",
        ]
    )


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()

    station_table = (
        df.assign(station_key=df["Station Names"].map(normalize_station_name))
        .groupby("station_key", as_index=False)
        .agg(
            station_name=("Station Names", "first"),
            latitude=("Latitude", "mean"),
            longitude=("Longitude", "mean"),
            lines=("Metro Line", lambda values: sorted(set(values))),
        )
    )

    for _, row in station_table.iterrows():
        graph.add_node(
            row["station_key"],
            label=row["station_name"],
            lat=row["latitude"],
            lon=row["longitude"],
            lines=row["lines"],
        )

    for _, line_df in df.groupby("Metro Line"):
        ordered = line_df.sort_values("Dist. From First Station(km)")
        keys = ordered["Station Names"].map(normalize_station_name).tolist()
        for src, dst in zip(keys, keys[1:]):
            if src == dst:
                continue
            if graph.has_edge(src, dst):
                graph[src][dst]["weight"] += 1
            else:
                graph.add_edge(src, dst, weight=1)

    return graph


def print_graph_metrics(graph: nx.DiGraph) -> pd.DataFrame:
    degree = dict(graph.degree())
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph, alpha=0.9)

    stats_df = pd.DataFrame(
        {
            "station_key": list(graph.nodes()),
            "station_name": [graph.nodes[n]["label"] for n in graph.nodes()],
            "degree": [degree[n] for n in graph.nodes()],
            "in_degree": [in_degree[n] for n in graph.nodes()],
            "out_degree": [out_degree[n] for n in graph.nodes()],
            "pagerank": [pagerank[n] for n in graph.nodes()],
            "num_lines": [len(graph.nodes[n]["lines"]) for n in graph.nodes()],
        }
    ).sort_values("degree", ascending=False)

    print("\n===== BUSIEST STATIONS (TOP 10 by degree) =====")
    print(stats_df[["station_name", "degree"]].head(10).to_string(index=False))

    print("\n===== LEAST CONNECTED STATIONS (BOTTOM 10 by degree) =====")
    print(stats_df[["station_name", "degree"]].tail(10).to_string(index=False))

    isolated = stats_df[stats_df["degree"] == 0]
    print("\n===== ISOLATED STATIONS =====")
    if isolated.empty:
        print("No isolated stations found.")
    else:
        print(isolated[["station_name"]].to_string(index=False))

    print("\n===== TOP 10 BY PAGERANK =====")
    print(
        stats_df.sort_values("pagerank", ascending=False)[["station_name", "pagerank"]]
        .head(10)
        .to_string(index=False)
    )

    return stats_df


def visualize_connectivity(graph: nx.DiGraph, out_file: Path) -> None:
    pagerank = nx.pagerank(graph, alpha=0.9)
    nodes = list(graph.nodes())
    x = [graph.nodes[n]["lon"] for n in nodes]
    y = [graph.nodes[n]["lat"] for n in nodes]
    c = [pagerank[n] for n in nodes]
    s = [180 + pagerank[n] * 15000 for n in nodes]

    plt.figure(figsize=(12, 9))
    for src, dst in graph.edges():
        plt.plot(
            [graph.nodes[src]["lon"], graph.nodes[dst]["lon"]],
            [graph.nodes[src]["lat"], graph.nodes[dst]["lat"]],
            color="#8d8d8d",
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )
    scatter = plt.scatter(
        x, y, c=c, s=s, cmap="plasma", edgecolor="white", linewidth=0.4, zorder=2
    )
    plt.colorbar(scatter, label="PageRank")
    plt.title("Metro Station Connectivity Network")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"\nSaved connectivity visualization: {out_file}")


def train_random_forest(stats_df: pd.DataFrame) -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print(
            "\nRandom Forest step skipped. Install scikit-learn with: "
            "python3 -m pip install scikit-learn"
        )
        return

    df = stats_df.copy()
    rng = np.random.default_rng(42)

    # Proxy target because the source dataset has no passenger-count column.
    degree_n = df["degree"] / max(df["degree"].max(), 1)
    pagerank_n = df["pagerank"] / max(df["pagerank"].max(), 1e-9)
    lines_n = df["num_lines"] / max(df["num_lines"].max(), 1)
    noise = rng.normal(loc=0.0, scale=1200.0, size=len(df))
    df["passenger_traffic_proxy"] = (
        30000 + 55000 * degree_n + 90000 * pagerank_n + 35000 * lines_n + noise
    ).clip(lower=5000)

    features = df[["degree", "in_degree", "out_degree", "pagerank", "num_lines"]]
    target = df["passenger_traffic_proxy"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print("\n===== RANDOM FOREST TRAFFIC PREDICTION =====")
    print(f"R2 Score: {r2_score(y_test, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")

    fi = pd.DataFrame(
        {"feature": features.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nFeature importance:")
    print(fi.to_string(index=False))

    df["predicted_traffic"] = model.predict(features)
    out_csv = Path(__file__).with_name("station_analysis_with_predictions.csv")
    df.sort_values("predicted_traffic", ascending=False).to_csv(out_csv, index=False)
    print(f"\nSaved predictions: {out_csv}")


def main() -> None:
    csv_path = Path(__file__).with_name("metro.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {csv_path}")

    metro_df = load_data(csv_path)
    graph = build_graph(metro_df)
    stats = print_graph_metrics(graph)
    visualize_connectivity(graph, Path(__file__).with_name("metro_connectivity.png"))
    train_random_forest(stats)


if __name__ == "__main__":
    main()
