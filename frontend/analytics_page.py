"""
frontend/analytics_page.py
──────────────────────────────────────────────────────────────────
Full analytics dashboard — 4 zones:
  Zone 1 – Ingestion Profiling table
  Zone 2 – Diagnostic KPI metrics (Plain English)
  Zone 3 – Graphical matrices (2 Side-by-Side Charts)
  Zone 4 – Tabular grid of extracted complaints
"""

import traceback
import pandas as pd
import streamlit as st
from frontend.workspace_page import render_history_dialog

# ─────────────────────────────────────────────────────────────────
# Plotting imports
# ─────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ─────────────────────────────────────────────────────────────────
# Chart theme
# ─────────────────────────────────────────────────────────────────
_BG_TRANSPARENT = "rgba(0,0,0,0)"
_PLOT_PAPER_BG  = "rgba(13,17,23,0)"
_GRID_COLOR     = "#1e2d3d"
_FONT_COLOR     = "#8899aa"
_FONT_MONO      = "Share Tech Mono, monospace"
_CYAN           = "#00e5ff"
_CRIMSON        = "#ff1744"

def _base_layout(title: str, title_color: str = _CYAN) -> dict:
    return dict(
        title=dict(text=title, font=dict(family=_FONT_MONO, size=15, color=title_color)),
        paper_bgcolor=_PLOT_PAPER_BG,
        plot_bgcolor=_BG_TRANSPARENT,
        font=dict(family=_FONT_MONO, color=_FONT_COLOR, size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    )


# ─────────────────────────────────────────────────────────────────
# Zone renderers
# ─────────────────────────────────────────────────────────────────
def _zone1_ingestion_profiling(r: dict) -> None:
    st.markdown(
        "<div class='fi-zone-title'>◈ ZONE 1 — DATA PIPELINE SUMMARY</div>",
        unsafe_allow_html=True,
    )

    engine_badge = (
        "<span class='fi-badge fi-badge-llm'>LLM ENGINE</span>"
        if r["engine_used"] == "LLM"
        else "<span class='fi-badge fi-badge-heuristic'>HEURISTIC ENGINE</span>"
    )

    table_data = {
        "Metric": [
            "Total Reviews Uploaded",
            "Negative Reviews Isolated (Dropped 4 & 5 Stars)",
            "Final Analyzed Volume (Capped)",
            "Active Processing Engine",
            "Data Source",
            "Text Column Detected",
            "Rating Column Detected",
        ],
        "Value": [
            f"{r['raw_rows']:,}",
            f"{r['filtered_rows']:,}",
            f"{r['capped_rows']}",
            r["engine_used"],
            r["source"].upper(),
            r["text_col"] or "—",
            r["rating_col"] or "— (not found)",
        ],
    }

    prof_df = pd.DataFrame(table_data)
    st.dataframe(
        prof_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Value":  st.column_config.TextColumn("Value",  width="medium"),
        },
    )
    st.markdown(f"Analysis powered by: {engine_badge}", unsafe_allow_html=True)


def _zone2_diagnostic_metrics(r: dict) -> None:
    st.markdown(
        "<div class='fi-zone-title'>◈ ZONE 2 — AI PERFORMANCE DIAGNOSTICS</div>",
        unsafe_allow_html=True,
    )

    text_conf_pct  = round(r["text_conf"]   * 100, 1)
    rating_conf_pct = round(r["rating_conf"] * 100, 1)
    
    # Translate Silhouette Score (typically -1 to 1) into an easy percentage
    sil_score_raw = r["silhouette"]
    accuracy_pct = round(max(0, sil_score_raw) * 100, 1) 
    
    latency = round(r["latency"], 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique Flaw Categories Found", r["optimal_k"])
    c2.metric("Clustering Accuracy", f"{ accuracy_pct}%", help="How well the AI separated the different types of complaints.")
    c3.metric("Pipeline Processing Speed", f"{latency}s")
    c4.metric("Reviews Processed", r["capped_rows"])

    st.markdown("")
    st.markdown("**AI Schema Mapping Confidence**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="fi-confidence-bar">
              <div class="fi-confidence-label">TEXT COLUMN</div>
              <div class="fi-confidence-track">
                <div class="fi-confidence-fill" style="width:{text_conf_pct}%;"></div>
              </div>
              <div class="fi-confidence-pct">{text_conf_pct}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="fi-confidence-bar">
              <div class="fi-confidence-label">RATING COLUMN</div>
              <div class="fi-confidence-track">
                <div class="fi-confidence-fill" style="width:{rating_conf_pct}%;"></div>
              </div>
              <div class="fi-confidence-pct">{rating_conf_pct}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _zone3_graphical_matrices(r: dict) -> None:
    st.markdown("<div class='fi-zone-title'>◈ ZONE 3 — VISUAL DEFECT ANALYSIS</div>", unsafe_allow_html=True)
    extractions = r.get("extractions", [])
    if not extractions: return

    df = pd.DataFrame(extractions)
    df["Severity"] = df["Sentiment_Score"].astype(float).abs() * 100

    # ROW 1: Leaderboard (Full Width)
    st.markdown("**1. Most Frequent Complaints (Volume)**")
    flaw_counts = df["Feature_Mentioned"].value_counts().reset_index()
    flaw_counts.columns = ["Feature", "Count"]
    fig1 = px.bar(flaw_counts, x="Count", y="Feature", orientation="h", color="Count", color_continuous_scale="Viridis")
    fig1.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
    st.plotly_chart(fig1, width='stretch')

    # ROW 2: Distribution and Anger Level (Split Width)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**2. Severity Distribution**")
        df_overall = df.groupby("Feature_Mentioned")["Severity"].sum().reset_index()
        fig2 = px.pie(df_overall, values='Severity', names='Feature_Mentioned', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig2, width='stretch')

    with col2:
        st.markdown("**3. Average Anger Level per Component**")
        df_ang = df.groupby("Feature_Mentioned")["Severity"].mean().reset_index()

        # Define color logic
        def get_anger_color(val):
            if val <= 60: return "#00e676"   # Green (Low)
            if val <= 85: return "#ffeb3b"   # Yellow (Higher than low)
            if val <= 90: return "#ff9100"   # Orange (Less than high)
            return "#ff1744"                 # Red (High)

        # Generate a list of colors corresponding to the Severity of each feature
        bar_colors = [get_anger_color(val) for val in df_ang["Severity"]]

        # Build the bar chart
        fig3 = go.Figure(go.Bar(
            x=df_ang["Severity"],
            y=df_ang["Feature_Mentioned"],
            orientation="h",
            marker_color=bar_colors,  # Apply the custom color list
            text=[f"{val:.1f}%" for val in df_ang["Severity"]],
            textposition="outside"
        ))

        fig3.update_layout(
            xaxis=dict(range=[0, 110], title="Severity %"),
            margin=dict(t=20, b=20, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_FONT_COLOR)
        )
        
        st.plotly_chart(fig3, width='stretch')
def _zone4_tabular_grid(r: dict) -> None:
    st.markdown(
        "<div class='fi-zone-title'>◈ ZONE 4 — DETAILED COMPLAINT LEDGER</div>",
        unsafe_allow_html=True,
    )

    extractions = r.get("extractions", [])
    if not extractions:
        st.warning("No extraction results to display.")
        return

    rows = []
    for e in extractions:
        # Convert sentiment float to a 0-100 severity index
        raw_sentiment = float(e.get("Sentiment_Score", 0.0))
        severity_idx = int(abs(raw_sentiment) * 100)

        rows.append({
            "Defect Category": e.get("Feature_Mentioned", "—"),
            "Severity Index": f"{severity_idx}%",
            "Customer Complaint": e.get("Specific_Complaint", "—"),
        })

    grid_df = pd.DataFrame(rows)

    st.dataframe(
        grid_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Defect Category": st.column_config.TextColumn(width="medium"),
            "Severity Index": st.column_config.TextColumn(width="small"),
            "Customer Complaint": st.column_config.TextColumn(width="large"),
        },
    )

    # Download button
    csv_bytes = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬡ EXPORT COMPLAINTS AS CSV",
        data=csv_bytes,
        file_name="flawintel_complaints.csv",
        mime="text/csv",
        key="download_grid_csv",
    )

# ─────────────────────────────────────────────────────────────────
# Public renderer
# ─────────────────────────────────────────────────────────────────
def render_analytics_page() -> None:
    """Entry point called by app.py for the analytics dashboard."""

        # Sidebar history button (same behavior as Workspace)
    with st.sidebar:
        st.markdown("### 🛠️ WORKSPACE TOOLS")
        if st.button(" Watch History", use_container_width=True, key="hist_btn_sidebar_analytics"):
            render_history_dialog(st.session_state.get("username", "unknown"))


    st.markdown(
        "<div class='fi-zone-title'>◈ ANALYTICS INTELLIGENCE DASHBOARD</div>",
        unsafe_allow_html=True,
    )

    results = st.session_state.get("pipeline_results")

    if not results:
        st.info(
            "No pipeline run detected. Navigate to the **Workspace** page, "
            "upload a data source, and execute the pipeline first."
        )
        return

    try:
        st.markdown("---")
        _zone1_ingestion_profiling(results)
        st.markdown("---")
        _zone2_diagnostic_metrics(results)
        st.markdown("---")
        _zone3_graphical_matrices(results)
        st.markdown("---")
        _zone4_tabular_grid(results)

    except Exception:
        traceback.print_exc()
        st.error(
            "Dashboard rendering encountered an unexpected error. "
            "See the terminal for the full traceback."
        )