"""Create visualizations for the basic network-analysis tables."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "data" / "tmp" / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_NETWORK_DIR = PROJECT_ROOT / "data" / "processed" / "network_analysis"
DEFAULT_POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_inputs" / "posts_for_analysis_topic_labeled.csv"
DEFAULT_TOPIC_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bertopic_posts"
    / "final_mcs15_ms15_nn15_nc5"
    / "topic_summary.csv"
)

REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "pandas": "pandas",
}

ROLE_COLORS = {
    "comment_only": "#4C78A8",
    "post_only": "#F58518",
    "both_post_and_comment": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize basic network-analysis outputs.")
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--posts-path", type=Path, default=DEFAULT_POSTS_PATH)
    parser.add_argument("--topic-summary-path", type=Path, default=DEFAULT_TOPIC_SUMMARY_PATH)
    parser.add_argument("--top-n", type=int, default=20, help="Top N rows for bar charts.")
    parser.add_argument("--top-network-users", type=int, default=30, help="Top users for bipartite network.")
    parser.add_argument("--top-network-videos", type=int, default=30, help="Top videos for bipartite network.")
    parser.add_argument("--top-time-topics", type=int, default=12, help="Top topics to show in time-based charts.")
    parser.add_argument("--force-network-top-users", type=int, default=100, help="Top comment users for force network.")
    parser.add_argument(
        "--force-network-min-edge-comments",
        type=int,
        default=2,
        help="Minimum user-video comment_count edge weight for force network.",
    )
    return parser.parse_args()


def check_dependencies() -> None:
    missing = [
        package_name
        for module_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise SystemExit("Missing required dependencies: " + ", ".join(missing))


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_required_csv(network_dir: Path, name: str) -> pd.DataFrame:
    path = network_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing network-analysis file: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def read_posts_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing post-analysis file: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def read_topic_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "topic" not in df.columns:
        return {}

    name_col = "topic_name" if "topic_name" in df.columns else "top_words" if "top_words" in df.columns else ""
    if not name_col:
        return {}

    topic_names: dict[str, str] = {}
    for row in df.itertuples(index=False):
        topic = str(getattr(row, "topic", "")).strip()
        name = str(getattr(row, name_col, "")).strip()
        if topic and name:
            topic_names[topic] = name
    return topic_names


def as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def parse_create_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    # Douyin create_time is normally a Unix timestamp in seconds. If a value is
    # unexpectedly in milliseconds, convert it before building the datetime.
    numeric = numeric.where(numeric < 10_000_000_000, numeric / 1000)
    return pd.to_datetime(numeric, unit="s", errors="coerce")


def short_id(value: str, left: int = 4, right: int = 4) -> str:
    value = str(value)
    if len(value) <= left + right + 2:
        return value
    return f"{value[:left]}...{value[-right:]}"


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_user_role_counts(user_roles: pd.DataFrame, output_dir: Path) -> Path:
    counts = user_roles["role"].value_counts().reindex(
        ["comment_only", "post_only", "both_post_and_comment"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[ROLE_COLORS[role] for role in counts.index])
    ax.set_title("User Role Counts", fontsize=16, fontweight="bold")
    ax.set_xlabel("role")
    ax.set_ylabel("user_count")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        height = int(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:,}", ha="center", va="bottom", fontsize=9)
    path = output_dir / "user_role_counts.png"
    save_fig(path)
    return path


def plot_user_activity_scatter(user_roles: pd.DataFrame, output_dir: Path) -> Path:
    df = user_roles.copy()
    df["post_count"] = as_int(df["post_count"])
    df["comment_count"] = as_int(df["comment_count"])

    fig, ax = plt.subplots(figsize=(9, 7))
    for role, group in df.groupby("role"):
        ax.scatter(
            group["post_count"],
            group["comment_count"],
            s=8,
            alpha=0.35,
            color=ROLE_COLORS.get(role, "#999999"),
            label=role,
            edgecolors="none",
        )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("User Activity Scatter", fontsize=16, fontweight="bold")
    ax.set_xlabel("post_count (symlog)")
    ax.set_ylabel("comment_count (symlog)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "user_activity_scatter.png"
    save_fig(path)
    return path


def plot_top_comment_users(user_roles: pd.DataFrame, output_dir: Path, top_n: int) -> Path:
    df = user_roles.copy()
    df["comment_count"] = as_int(df["comment_count"])
    df["commented_video_count"] = as_int(df["commented_video_count"])
    top = df.sort_values(["comment_count", "commented_video_count"], ascending=False).head(top_n).iloc[::-1]

    labels = [short_id(value) for value in top["user_id"]]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(labels, top["comment_count"], color="#4C78A8")
    ax.set_title(f"Top {top_n} Comment Users", fontsize=16, fontweight="bold")
    ax.set_xlabel("comment_count")
    ax.set_ylabel("user_id")
    ax.grid(axis="x", alpha=0.25)
    for index, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["comment_count"],
            index,
            f" videos={row['commented_video_count']}",
            va="center",
            fontsize=8,
            color="#333333",
        )
    path = output_dir / "top_comment_users.png"
    save_fig(path)
    return path


def plot_topic_comment_volume(topic_summary: pd.DataFrame, output_dir: Path) -> Path:
    df = topic_summary.copy()
    df["comment_count"] = as_int(df["comment_count"])
    df = df.sort_values("comment_count", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["topic"].astype(str), df["comment_count"], color="#72B7B2")
    ax.set_title("Topic Comment Volume", fontsize=16, fontweight="bold")
    ax.set_xlabel("topic")
    ax.set_ylabel("comment_count")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["topic"].astype(str), rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "topic_comment_volume.png"
    save_fig(path)
    return path


def plot_top_commented_videos(user_video_edges: pd.DataFrame, output_dir: Path, top_n: int) -> Path:
    df = user_video_edges.copy()
    df["comment_count"] = as_int(df["comment_count"])
    video_stats = (
        df.groupby(["aweme_id", "topic"], dropna=False)
        .agg(comment_count=("comment_count", "sum"), comment_user_count=("user_id", pd.Series.nunique))
        .reset_index()
        .sort_values(["comment_count", "comment_user_count"], ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    labels = [f"{short_id(row.aweme_id)} | t={row.topic}" for row in video_stats.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(labels, video_stats["comment_count"], color="#F58518")
    ax.set_title(f"Top {top_n} Commented Videos", fontsize=16, fontweight="bold")
    ax.set_xlabel("comment_count")
    ax.set_ylabel("aweme_id | topic")
    ax.grid(axis="x", alpha=0.25)
    for index, row in enumerate(video_stats.itertuples()):
        ax.text(row.comment_count, index, f" users={row.comment_user_count}", va="center", fontsize=8, color="#333333")
    path = output_dir / "top_commented_videos.png"
    save_fig(path)
    return path


def plot_user_video_bipartite_network(
    user_video_edges: pd.DataFrame,
    output_dir: Path,
    top_users: int,
    top_videos: int,
) -> Path:
    df = user_video_edges.copy()
    df["comment_count"] = as_int(df["comment_count"])

    selected_users = (
        df.groupby("user_id")["comment_count"].sum().sort_values(ascending=False).head(top_users).index
    )
    selected_videos = (
        df.groupby("aweme_id")["comment_count"].sum().sort_values(ascending=False).head(top_videos).index
    )
    sub = df[df["user_id"].isin(selected_users) & df["aweme_id"].isin(selected_videos)].copy()

    graph = nx.Graph()
    for row in sub.itertuples(index=False):
        user_node = f"user:{row.user_id}"
        video_node = f"video:{row.aweme_id}"
        graph.add_node(user_node, bipartite="user", label=short_id(row.user_id))
        graph.add_node(video_node, bipartite="video", label=f"{short_id(row.aweme_id)}\nt{row.topic}")
        graph.add_edge(user_node, video_node, weight=max(int(row.comment_count), 1))

    user_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("bipartite") == "user"]
    video_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("bipartite") == "video"]
    pos: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(user_nodes):
        pos[node] = (0.0, index)
    for index, node in enumerate(video_nodes):
        pos[node] = (1.0, index)

    fig_height = max(8, max(len(user_nodes), len(video_nodes)) * 0.28)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    widths = [0.5 + min(graph.edges[edge]["weight"], 20) * 0.08 for edge in graph.edges]

    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, alpha=0.25, edge_color="#666666")
    nx.draw_networkx_nodes(graph, pos, nodelist=user_nodes, node_color="#4C78A8", node_size=140, ax=ax, label="users")
    nx.draw_networkx_nodes(graph, pos, nodelist=video_nodes, node_color="#F58518", node_size=180, ax=ax, label="videos")
    labels = {node: attrs["label"] for node, attrs in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title(f"User-Video Bipartite Network | Top {top_users} Users + Top {top_videos} Videos", fontsize=16, fontweight="bold")
    ax.text(0, -1.5, "comment users", ha="center", fontsize=11, fontweight="bold")
    ax.text(1, -1.5, "videos", ha="center", fontsize=11, fontweight="bold")
    ax.axis("off")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    path = output_dir / "user_video_bipartite_top_network.png"
    save_fig(path)
    return path


def plot_user_video_force_network_top(
    user_video_edges: pd.DataFrame,
    output_dir: Path,
    top_users: int,
    min_edge_comments: int,
) -> Path:
    df = user_video_edges.copy()
    df["comment_count"] = as_int(df["comment_count"])
    df = df[df["comment_count"] >= min_edge_comments].copy()

    selected_users = df.groupby("user_id")["comment_count"].sum().sort_values(ascending=False).head(top_users).index
    sub = df[df["user_id"].isin(selected_users)].copy()

    graph = nx.Graph()
    user_comment_totals = sub.groupby("user_id")["comment_count"].sum().to_dict()
    video_comment_totals = sub.groupby("aweme_id")["comment_count"].sum().to_dict()

    for row in sub.itertuples(index=False):
        user_node = f"user:{row.user_id}"
        video_node = f"video:{row.aweme_id}"
        graph.add_node(user_node, node_type="user", total_comments=user_comment_totals.get(row.user_id, 0))
        graph.add_node(video_node, node_type="video", total_comments=video_comment_totals.get(row.aweme_id, 0))
        graph.add_edge(user_node, video_node, weight=max(int(row.comment_count), 1))

    if graph.number_of_nodes() == 0:
        raise ValueError("No force-network edges remain after filtering.")

    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.35, iterations=120)
    user_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("node_type") == "user"]
    video_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("node_type") == "video"]

    user_sizes = [40 + min(graph.nodes[node].get("total_comments", 0), 500) * 0.55 for node in user_nodes]
    video_sizes = [30 + min(graph.nodes[node].get("total_comments", 0), 800) * 0.35 for node in video_nodes]
    edge_widths = [0.25 + min(attrs.get("weight", 1), 20) * 0.08 for _, _, attrs in graph.edges(data=True)]

    fig, ax = plt.subplots(figsize=(15, 12))
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, alpha=0.16, edge_color="#555555")
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=video_nodes,
        node_color="#F58518",
        node_size=video_sizes,
        alpha=0.58,
        linewidths=0.35,
        edgecolors="white",
        ax=ax,
        label="videos",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=user_nodes,
        node_color="#4C78A8",
        node_size=user_sizes,
        alpha=0.78,
        linewidths=0.45,
        edgecolors="white",
        ax=ax,
        label="users",
    )

    ax.set_title(
        f"User-Video Force Network | Top {top_users} Comment Users | edge comment_count >= {min_edge_comments}",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.01,
        f"nodes={graph.number_of_nodes():,}  edges={graph.number_of_edges():,}",
        transform=ax.transAxes,
        fontsize=10,
        color="#333333",
    )
    ax.axis("off")
    ax.legend(frameon=False, loc="upper right")
    path = output_dir / "user_video_force_network_top.png"
    save_fig(path)
    return path


def build_video_interaction_table(posts: pd.DataFrame, user_video_edges: pd.DataFrame) -> pd.DataFrame:
    post_cols = ["aweme_id", "create_time", "topic", "include_in_analysis"]
    missing = [col for col in post_cols if col not in posts.columns]
    if missing:
        raise ValueError("posts file is missing required columns: " + ", ".join(missing))

    edge_cols = ["aweme_id", "user_id", "comment_count"]
    missing = [col for col in edge_cols if col not in user_video_edges.columns]
    if missing:
        raise ValueError("user_video_edges file is missing required columns: " + ", ".join(missing))

    selected_posts = posts[posts["include_in_analysis"].str.lower().eq("true")].copy()
    selected_posts = selected_posts[["aweme_id", "create_time", "topic"]].drop_duplicates("aweme_id")
    selected_posts["published_at"] = parse_create_time(selected_posts["create_time"])

    edges = user_video_edges.copy()
    edges["comment_count"] = as_int(edges["comment_count"])
    video_stats = (
        edges.groupby("aweme_id", dropna=False)
        .agg(comment_count=("comment_count", "sum"), comment_user_count=("user_id", pd.Series.nunique))
        .reset_index()
    )

    video_df = selected_posts.merge(video_stats, on="aweme_id", how="left")
    video_df["comment_count"] = video_df["comment_count"].fillna(0).astype(int)
    video_df["comment_user_count"] = video_df["comment_user_count"].fillna(0).astype(int)
    return video_df.dropna(subset=["published_at"]).copy()


def choose_time_topic_labels(video_df: pd.DataFrame, top_topics: int) -> tuple[pd.DataFrame, list[str]]:
    df = video_df.copy()
    df["topic"] = df["topic"].replace("", "blank").astype(str)
    topic_totals = df[df["topic"] != "-1"].groupby("topic")["comment_count"].sum().sort_values(ascending=False)
    selected_topics = topic_totals.head(top_topics).index.tolist()
    df["topic_for_time_plot"] = df["topic"].where(df["topic"].isin(selected_topics), "other")
    order = selected_topics + (["other"] if (df["topic_for_time_plot"] == "other").any() else [])
    return df, order


def plot_video_interaction_over_time_bubble(
    posts: pd.DataFrame,
    user_video_edges: pd.DataFrame,
    output_dir: Path,
    top_topics: int,
) -> Path:
    video_df = build_video_interaction_table(posts, user_video_edges)
    video_df, topic_order = choose_time_topic_labels(video_df, top_topics)

    color_map = plt.get_cmap("tab20")
    colors = {topic: color_map(index % 20) for index, topic in enumerate(topic_order)}
    colors.setdefault("other", "#B8B8B8")

    fig, ax = plt.subplots(figsize=(13, 7))
    for topic in topic_order:
        group = video_df[video_df["topic_for_time_plot"] == topic]
        if group.empty:
            continue
        sizes = 18 + (group["comment_user_count"].clip(upper=500) ** 0.5) * 10
        ax.scatter(
            group["published_at"],
            group["comment_count"],
            s=sizes,
            alpha=0.55,
            color=colors.get(topic, "#999999"),
            edgecolors="white",
            linewidths=0.35,
            label=f"topic {topic}",
        )

    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Video Interaction Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("publish_time")
    ax.set_ylabel("comment_count (symlog)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / "video_interaction_over_time_bubble.png"
    save_fig(path)
    return path


def plot_topic_interaction_over_time_area(
    posts: pd.DataFrame,
    user_video_edges: pd.DataFrame,
    output_dir: Path,
    top_topics: int,
    topic_name_map: dict[str, str],
) -> Path:
    video_df = build_video_interaction_table(posts, user_video_edges)
    video_df, topic_order = choose_time_topic_labels(video_df, top_topics)
    video_df["year"] = video_df["published_at"].dt.year

    yearly = (
        video_df.groupby(["year", "topic_for_time_plot"], dropna=False)["comment_count"]
        .sum()
        .reset_index()
        .pivot(index="year", columns="topic_for_time_plot", values="comment_count")
        .fillna(0)
        .sort_index()
    )
    yearly = yearly[[topic for topic in topic_order if topic in yearly.columns]]

    fig, ax = plt.subplots(figsize=(13, 7))
    yearly.plot.area(ax=ax, alpha=0.85, linewidth=0.5, colormap="tab20")
    ax.set_title("Topic Interaction Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("publish_year")
    ax.set_ylabel("comment_count")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    display_labels = [
        label if label == "other" else topic_name_map.get(label, label)
        for label in labels
    ]
    ax.legend(
        handles,
        display_labels,
        title="topic",
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    path = output_dir / "topic_interaction_over_time_area.png"
    save_fig(path)
    return path


def write_summary(output_dir: Path, paths: list[Path]) -> Path:
    path = output_dir / "visualization_summary.md"
    lines = [
        "# Network Analysis Visualizations",
        "",
        "These figures visualize user roles and user-video comment relations based on include_in_analysis=true rows.",
        "",
    ]
    for output_path in paths:
        lines.append(f"- {output_path.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    check_dependencies()
    set_plot_style()

    network_dir = args.network_dir.resolve()
    output_dir = network_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    user_roles = read_required_csv(network_dir, "user_role_summary.csv")
    user_video_edges = read_required_csv(network_dir, "user_video_edges.csv")
    topic_summary = read_required_csv(network_dir, "topic_interaction_summary.csv")
    posts = read_posts_csv(args.posts_path.resolve())
    topic_name_map = read_topic_name_map(args.topic_summary_path.resolve())

    paths = [
        plot_user_role_counts(user_roles, output_dir),
        plot_user_activity_scatter(user_roles, output_dir),
        plot_top_comment_users(user_roles, output_dir, args.top_n),
        plot_top_commented_videos(user_video_edges, output_dir, args.top_n),
        plot_topic_comment_volume(topic_summary, output_dir),
        plot_video_interaction_over_time_bubble(posts, user_video_edges, output_dir, args.top_time_topics),
        plot_topic_interaction_over_time_area(
            posts,
            user_video_edges,
            output_dir,
            args.top_time_topics,
            topic_name_map,
        ),
        plot_user_video_bipartite_network(
            user_video_edges,
            output_dir,
            args.top_network_users,
            args.top_network_videos,
        ),
        plot_user_video_force_network_top(
            user_video_edges,
            output_dir,
            args.force_network_top_users,
            args.force_network_min_edge_comments,
        ),
    ]
    summary_path = write_summary(output_dir, paths)

    print(f"Visualization output dir: {output_dir}")
    for path in paths:
        print(f"Generated: {path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
