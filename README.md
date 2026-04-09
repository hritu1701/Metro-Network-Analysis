# Metro Station Connectivity Analysis

Small college assignment project on metro network analysis using graph methods and machine learning.

## Assignment Objectives

- Load metro route data and construct a directed graph.
- Compute degree centrality to find busiest stations.
- Identify least connected and isolated stations.
- Compute PageRank to rank station importance.
- Visualize metro network connectivity.
- Train a Random Forest model to predict passenger traffic.

## Project Files

- `metro.csv` - source dataset.
- `metro_assignment.py` - one-command terminal workflow for all assignment tasks.
- `metro_visual.py` - enhanced metro visualization (map + PageRank panel).
- `metro_gx.scala` - GraphX script for Spark shell.
- `metro_nodes/`, `metro_edges/` - exported graph artifacts.

## Requirements

- Python 3.9+
- `pandas`
- `networkx`
- `matplotlib`
- `numpy`
- `scikit-learn`

Install:

```bash
python3 -m pip install pandas networkx matplotlib numpy scikit-learn
```

## Run Complete Assignment in Terminal

From project folder:

```bash
cd /Users/hrituraj/Desktop/clusterproject
python3 metro_assignment.py
```

Outputs generated:

- Console report with busiest/least/isolated stations and top PageRank stations.
- `metro_connectivity.png` network connectivity visualization.
- `station_analysis_with_predictions.csv` with predicted traffic values.

## GraphX Version (Spark)

Run in Spark shell:

```bash
cd /Users/hrituraj/Desktop/clusterproject
spark-shell
```

Then inside Spark shell:

```scala
:load metro_gx.scala
```

Optional dataset path override:

```bash
METRO_CSV_PATH=/full/path/to/metro.csv spark-shell
```

## Important Note for Report

The dataset does not include real passenger-count labels.  
For the Random Forest step, `metro_assignment.py` creates a clearly marked **traffic proxy target** from graph features (`degree`, `pagerank`, `num_lines`) to demonstrate the ML pipeline expected in the assignment.
