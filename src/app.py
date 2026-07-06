"""
app.py

Streamlit dashboard for the Steam Backlog Analyzer.
Reuses the exact same data loading, classification, and chart functions
from visualize_games.py so both files stay in sync.

Run from the src/ directory with:
    streamlit run app.py
"""

import streamlit as st

from visualize_games import (
    load_data,
    classify_games,
    backlog_summary,
    plot_backlog_donut,
    plot_pareto,
    plot_top_n,
    plot_playtime_histogram,
    plot_recency_scatter,
)

st.set_page_config(page_title="Steam Backlog Analyzer", layout="wide")

st.title("🎮 Steam Backlog Analyzer")
st.caption(
    "Backlog = games you've never played, or started and haven't touched "
    "in the last 2 weeks."
)

# ---------------------------------------------------------------------------
# Load + classify data (cached so re-running the app doesn't re-read the CSV)
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return classify_games(load_data())

try:
    df = get_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

summary = backlog_summary(df)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total games", summary["total_games"])
col2.metric("Backlog games", summary["backlog_count"],
            f"{summary['backlog_pct']:.0f}% of library")
col3.metric("Never played", summary["never_played"])
col4.metric("Abandoned", summary["abandoned"],
            f"{summary['hours_in_abandoned']:.0f} hrs invested, then dropped")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

top_row_left, top_row_right = st.columns(2)
with top_row_left:
    st.pyplot(plot_backlog_donut(df))
with top_row_right:
    st.pyplot(plot_pareto(df))

mid_row_left, mid_row_right = st.columns(2)
with mid_row_left:
    n = st.slider("Number of top games to show", 5, 30, 15)
    st.pyplot(plot_top_n(df, n=n))
with mid_row_right:
    st.pyplot(plot_playtime_histogram(df))

st.pyplot(plot_recency_scatter(df))

st.divider()

# ---------------------------------------------------------------------------
# Backlog tables
# ---------------------------------------------------------------------------

st.subheader("Backlog Explorer")
status_filter = st.multiselect(
    "Filter by status",
    options=["active", "abandoned", "never_played"],
    default=["abandoned", "never_played"],
)

filtered = df[df["status"].isin(status_filter)][
    ["name", "status", "playtime_hours", "days_since_played"]
].sort_values("playtime_hours", ascending=False)

st.dataframe(filtered, use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered list as CSV",
    data=filtered.to_csv(index=False),
    file_name="backlog_filtered.csv",
    mime="text/csv",
)
