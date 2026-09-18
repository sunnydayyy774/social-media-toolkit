"""Build network-analysis tables from labeled post/comment analysis inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_inputs"
DEFAULT_POSTS = DEFAULT_INPUT_DIR / "posts_for_analysis_topic_labeled.csv"
DEFAULT_COMMENTS = DEFAULT_INPUT_DIR / "comments_for_analysis_topic_labeled.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "network_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build basic user/video network-analysis tables.")
    parser.add_argument("--posts", type=Path, default=DEFAULT_POSTS)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().eq("true")


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def to_numeric_sum(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    temp_path.replace(path)


def build_user_role_summary(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    post_counts = (
        posts.loc[posts["author_id"].notna() & posts["author_id"].ne("")]
        .groupby("author_id", dropna=False)
        .size()
        .rename("post_count")
        .reset_index()
        .rename(columns={"author_id": "user_id"})
    )

    comment_stats = (
        comments.loc[comments["user_uid"].notna() & comments["user_uid"].ne("")]
        .groupby("user_uid", dropna=False)
        .agg(
            comment_count=("comment_id", "size"),
            commented_video_count=("aweme_id", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"user_uid": "user_id"})
    )

    users = pd.DataFrame(
        {"user_id": sorted(set(post_counts["user_id"]).union(set(comment_stats["user_id"])))}
    )
    summary = users.merge(post_counts, on="user_id", how="left").merge(comment_stats, on="user_id", how="left")
    summary[["post_count", "comment_count", "commented_video_count"]] = summary[
        ["post_count", "comment_count", "commented_video_count"]
    ].fillna(0).astype(int)
    summary["is_post_author"] = summary["post_count"].gt(0).map({True: "true", False: "false"})
    summary["is_comment_author"] = summary["comment_count"].gt(0).map({True: "true", False: "false"})

    def role(row: pd.Series) -> str:
        if row["post_count"] > 0 and row["comment_count"] > 0:
            return "both_post_and_comment"
        if row["post_count"] > 0:
            return "post_only"
        return "comment_only"

    summary["role"] = summary.apply(role, axis=1)
    return summary[
        [
            "user_id",
            "is_post_author",
            "is_comment_author",
            "post_count",
            "comment_count",
            "commented_video_count",
            "role",
        ]
    ].sort_values(["comment_count", "post_count", "user_id"], ascending=[False, False, True])


def build_user_video_edges(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    post_topic = posts[["aweme_id", "topic"]].drop_duplicates(subset=["aweme_id"])
    comments = comments.copy()
    comments["comment_like_numeric"] = to_numeric_sum(comments.get("likes", pd.Series(index=comments.index)))

    edges = (
        comments.loc[
            comments["user_uid"].notna()
            & comments["user_uid"].ne("")
            & comments["aweme_id"].notna()
            & comments["aweme_id"].ne("")
        ]
        .groupby(["user_uid", "aweme_id"], dropna=False)
        .agg(
            comment_count=("comment_id", "size"),
            comment_like_sum=("comment_like_numeric", "sum"),
        )
        .reset_index()
        .rename(columns={"user_uid": "user_id"})
    )
    edges["comment_like_sum"] = edges["comment_like_sum"].astype(int)
    edges = edges.merge(post_topic, on="aweme_id", how="left")
    return edges[["user_id", "aweme_id", "comment_count", "comment_like_sum", "topic"]].sort_values(
        ["comment_count", "comment_like_sum", "user_id", "aweme_id"], ascending=[False, False, True, True]
    )


def build_topic_interaction_summary(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    post_topic = posts[["aweme_id", "topic"]].drop_duplicates(subset=["aweme_id"])
    comments_with_topic = comments.merge(post_topic, on="aweme_id", how="left")

    video_counts = (
        post_topic.groupby("topic", dropna=False)
        .agg(video_count=("aweme_id", pd.Series.nunique))
        .reset_index()
    )
    comment_stats = (
        comments_with_topic.groupby("topic", dropna=False)
        .agg(
            comment_count=("comment_id", "size"),
            comment_user_count=("user_uid", pd.Series.nunique),
        )
        .reset_index()
    )
    summary = video_counts.merge(comment_stats, on="topic", how="left")
    summary[["comment_count", "comment_user_count"]] = summary[["comment_count", "comment_user_count"]].fillna(0).astype(int)
    summary["avg_comments_per_video"] = summary.apply(
        lambda row: row["comment_count"] / row["video_count"] if row["video_count"] else 0,
        axis=1,
    )
    summary["avg_comments_per_video"] = summary["avg_comments_per_video"].round(4)
    return summary[["topic", "video_count", "comment_count", "comment_user_count", "avg_comments_per_video"]].sort_values(
        "comment_count", ascending=False
    )


def build_network_summary(posts: pd.DataFrame, comments: pd.DataFrame, user_roles: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "post_count": len(posts),
        "comment_count": len(comments),
        "video_count": posts["aweme_id"].nunique(),
        "post_author_count": posts["author_id"].replace("", pd.NA).dropna().nunique(),
        "comment_user_count": comments["user_uid"].replace("", pd.NA).dropna().nunique(),
        "all_user_count": len(user_roles),
        "post_only_user_count": int(user_roles["role"].eq("post_only").sum()),
        "comment_only_user_count": int(user_roles["role"].eq("comment_only").sum()),
        "both_post_and_comment_user_count": int(user_roles["role"].eq("both_post_and_comment").sum()),
        "user_video_edge_count": len(edges),
        "max_user_comment_count": int(user_roles["comment_count"].max()) if not user_roles.empty else 0,
        "max_user_commented_video_count": int(user_roles["commented_video_count"].max()) if not user_roles.empty else 0,
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    posts = pd.read_csv(args.posts, dtype=str).fillna("")
    comments = pd.read_csv(args.comments, dtype=str).fillna("")
    require_columns(posts, {"aweme_id", "author_id", "topic", "include_in_analysis"}, "posts")
    require_columns(comments, {"comment_id", "aweme_id", "user_uid", "likes", "include_in_analysis"}, "comments")

    posts = posts.loc[normalize_bool(posts["include_in_analysis"])].copy()
    comments = comments.loc[normalize_bool(comments["include_in_analysis"])].copy()

    user_roles = build_user_role_summary(posts, comments)
    user_video_edges = build_user_video_edges(posts, comments)
    topic_interactions = build_topic_interaction_summary(posts, comments)
    network_summary = build_network_summary(posts, comments, user_roles, user_video_edges)

    write_csv(output_dir / "user_role_summary.csv", user_roles)
    write_csv(output_dir / "user_video_edges.csv", user_video_edges)
    write_csv(output_dir / "topic_interaction_summary.csv", topic_interactions)
    write_csv(output_dir / "network_summary.csv", network_summary)

    print(f"Input posts after include filter: {len(posts)}")
    print(f"Input comments after include filter: {len(comments)}")
    print(f"User role rows: {len(user_roles)}")
    print(f"User-video edge rows: {len(user_video_edges)}")
    print(f"Topic interaction rows: {len(topic_interactions)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
