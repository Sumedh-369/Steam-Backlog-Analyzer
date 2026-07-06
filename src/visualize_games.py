"""
visualize_games.py

Reads data/steam_games.csv (produced by fetch_games.py), classifies each game
into a backlog status, and generates charts to help understand your backlog.

Backlog definition:
    - Active:        playtime_2weeks > 0            (currently being played)
    - Abandoned:      playtime_forever > 0 AND NOT active   (started, then dropped)
    - Never played:  playtime_forever == 0          (bought, untouched)

Backlog = Abandoned + Never played (i.e. everything that isn't Active).

All plot_* functions return a matplotlib Figure so they can be reused by
app.py (the Streamlit dashboard) without duplicating logic. Running this
file directly saves every chart as a PNG under visualizations/.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Resolve paths relative to this file, not the current working directory,
# so this works whether you run it from src/ or somewhere else (e.g. Streamlit).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "steam_games.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")
TOP_N = 15

STATUS_COLORS = {
    "active": "#55A868",
    "abandoned": "#DD8452",
    "never_played": "#C44E52",
}
STATUS_LABELS = {
    "active": "Active",
    "abandoned": "Abandoned",
    "never_played": "Never Played",
}


# ---------------------------------------------------------------------------
# Data loading / classification
# ---------------------------------------------------------------------------

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find {path}. Run fetch_games.py first to generate it."
        )

    df = pd.read_csv(path)
    df["playtime_hours"] = df["playtime_forever"] / 60.0

    df["last_played"] = pd.to_datetime(
        df["rtime_last_played"], unit="s", utc=True, errors="coerce"
    )
    df.loc[df["rtime_last_played"] == 0, "last_played"] = pd.NaT

    now = datetime.now(timezone.utc)
    df["days_since_played"] = (now - df["last_played"]).dt.days

    return df


def classify_games(df):
    """Add a 'status' column: active / abandoned / never_played."""
    df = df.copy()

    # playtime_2weeks is absent from the API response (and CSV) for games
    # not played in the last 2 weeks, so missing == 0.
    if "playtime_2weeks" not in df.columns:
        df["playtime_2weeks"] = 0
    df["playtime_2weeks"] = df["playtime_2weeks"].fillna(0)

    is_active = df["playtime_2weeks"] > 0
    is_never_played = df["playtime_forever"] == 0
    is_abandoned = (~is_active) & (~is_never_played)

    df["status"] = np.select(
        [is_active, is_never_played, is_abandoned],
        ["active", "never_played", "abandoned"],
        default="active",
    )
    df["is_backlog"] = df["status"] != "active"
    return df


def backlog_summary(df):
    """Return a dict of headline stats used by both the script and dashboard."""
    total = len(df)
    counts = df["status"].value_counts().to_dict()
    active = counts.get("active", 0)
    abandoned = counts.get("abandoned", 0)
    never_played = counts.get("never_played", 0)
    backlog = abandoned + never_played

    return {
        "total_games": total,
        "active": active,
        "abandoned": abandoned,
        "never_played": never_played,
        "backlog_count": backlog,
        "backlog_pct": (backlog / total * 100) if total else 0,
        "hours_in_abandoned": df.loc[df["status"] == "abandoned", "playtime_hours"].sum(),
        "hours_total": df["playtime_hours"].sum(),
    }


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Charts (each returns a Figure; caller decides whether to save or display)
# ---------------------------------------------------------------------------

def plot_backlog_donut(df):
    """Active vs Abandoned vs Never Played, as a donut with backlog % in the center."""
    counts = df["status"].value_counts()
    order = ["active", "abandoned", "never_played"]
    values = [counts.get(s, 0) for s in order]
    colors = [STATUS_COLORS[s] for s in order]
    labels = [f"{STATUS_LABELS[s]} ({counts.get(s, 0)})" for s in order]

    summary = backlog_summary(df)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4),
    )
    ax.text(
        0, 0, f"{summary['backlog_pct']:.0f}%\nbacklog",
        ha="center", va="center", fontsize=16, fontweight="bold",
    )
    ax.set_title("Backlog Breakdown: Active vs Abandoned vs Never Played")
    fig.tight_layout()
    return fig


def plot_pareto(df):
    """Cumulative % of total playtime as games are added, sorted by hours desc."""
    sorted_df = df.sort_values("playtime_hours", ascending=False).reset_index(drop=True)
    total_hours = sorted_df["playtime_hours"].sum()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    if total_hours == 0:
        ax1.text(0.5, 0.5, "No playtime recorded yet", ha="center", va="center")
        return fig

    cumulative_pct = sorted_df["playtime_hours"].cumsum() / total_hours * 100
    game_rank = np.arange(1, len(sorted_df) + 1)
    bar_colors = [STATUS_COLORS[s] for s in sorted_df["status"]]

    ax1.bar(game_rank, sorted_df["playtime_hours"], color=bar_colors, alpha=0.8)
    ax1.set_xlabel("Games, ranked by playtime (most-played first)")
    ax1.set_ylabel("Hours played")

    ax2 = ax1.twinx()
    ax2.plot(game_rank, cumulative_pct, color="black", linewidth=2)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulative % of total playtime")
    ax2.set_ylim(0, 105)

    ax1.set_title("Pareto Chart: Where Your Playtime Actually Goes")
    fig.tight_layout()
    return fig


def plot_top_n(df, n=TOP_N):
    top = df.sort_values("playtime_hours", ascending=False).head(n)
    colors = [STATUS_COLORS[s] for s in top["status"]][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.4)))
    ax.barh(top["name"][::-1], top["playtime_hours"][::-1], color=colors)
    ax.set_xlabel("Hours played")
    ax.set_title(f"Top {n} Most-Played Games (colored by status)")
    fig.tight_layout()
    return fig


def plot_playtime_histogram(df):
    """Log-scale histogram of playtime, split by active vs abandoned."""
    played = df[df["playtime_hours"] > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    if played.empty:
        ax.text(0.5, 0.5, "No games with playtime > 0", ha="center", va="center")
        return fig

    bins = np.logspace(np.log10(played["playtime_hours"].min() + 0.01),
                        np.log10(played["playtime_hours"].max()), 30)

    for status in ["abandoned", "active"]:
        subset = played[played["status"] == status]["playtime_hours"]
        if not subset.empty:
            ax.hist(subset, bins=bins, alpha=0.6, label=STATUS_LABELS[status],
                     color=STATUS_COLORS[status], edgecolor="black")

    ax.set_xscale("log")
    ax.set_xlabel("Hours played (log scale)")
    ax.set_ylabel("Number of games")
    ax.set_title("Playtime Distribution: Active vs Abandoned")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_recency_scatter(df):
    """Playtime vs. days since last played, color-coded by status."""
    subset = df[(df["playtime_hours"] > 0) & (df["days_since_played"].notna())]

    fig, ax = plt.subplots(figsize=(8, 6))
    if subset.empty:
        ax.text(0.5, 0.5, "No games with valid last-played data", ha="center", va="center")
        return fig

    for status in ["active", "abandoned"]:
        s = subset[subset["status"] == status]
        ax.scatter(s["days_since_played"], s["playtime_hours"], alpha=0.6,
                   color=STATUS_COLORS[status], label=STATUS_LABELS[status])

    standout = subset.sort_values(
        by=["playtime_hours", "days_since_played"], ascending=[False, False]
    ).head(5)
    for _, row in standout.iterrows():
        ax.annotate(row["name"], (row["days_since_played"], row["playtime_hours"]),
                    fontsize=8, alpha=0.8)

    ax.set_xlabel("Days since last played")
    ax.set_ylabel("Hours played")
    ax.set_title("Playtime vs. Recency")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Backlog list export (not a chart, but the "actionable" output)
# ---------------------------------------------------------------------------

def export_backlog_lists(df):
    ensure_output_dir()

    never_played = (
        df[df["status"] == "never_played"][["name", "appid"]]
        .sort_values("name")
    )
    abandoned = (
        df[df["status"] == "abandoned"][["name", "appid", "playtime_hours", "days_since_played"]]
        .sort_values("playtime_hours", ascending=False)
    )

    never_played.to_csv(os.path.join(OUTPUT_DIR, "never_played_games.csv"), index=False)
    abandoned.to_csv(os.path.join(OUTPUT_DIR, "abandoned_games.csv"), index=False)

    print(f"Saved {len(never_played)} never-played games to never_played_games.csv")
    print(f"Saved {len(abandoned)} abandoned games to abandoned_games.csv")


# ---------------------------------------------------------------------------
# Main: generate and save every chart as a PNG
# ---------------------------------------------------------------------------

def main():
    ensure_output_dir()
    df = classify_games(load_data())

    summary = backlog_summary(df)
    print(f"Loaded {summary['total_games']} games")
    print(f"  Active:        {summary['active']}")
    print(f"  Abandoned:     {summary['abandoned']}")
    print(f"  Never played:  {summary['never_played']}")
    print(f"  Backlog:       {summary['backlog_count']} ({summary['backlog_pct']:.1f}%)")

    charts = {
        "backlog_donut.png": plot_backlog_donut,
        "pareto_playtime.png": plot_pareto,
        "top_n_games.png": plot_top_n,
        "playtime_histogram.png": plot_playtime_histogram,
        "recency_scatter.png": plot_recency_scatter,
    }

    for filename, plot_fn in charts.items():
        fig = plot_fn(df)
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close(fig)

    export_backlog_lists(df)
    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
