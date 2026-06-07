"""
frontend/workspace_page.py
──────────────────────────────────────────────────────────────────
Data ingestion workspace:
  - CSV upload OR live URL fetch via Apify
  - Automated credential loading (.env)
  - Pipeline trigger (preprocessing → ML → LLM)
  - Stores all results in st.session_state for analytics_page
  - SQLite-backed history tracking in a sidebar dialog
"""

import os
import time
import json
import sqlite3
import traceback
import pandas as pd
import streamlit as st

from backend.apify_fetch import fetch_reviews_from_url
from backend.auth import persist_analysis, _get_connection 
from backend.data_preprocessing import apify_to_dataframe, load_csv, preprocess_pipeline
from backend.llm_fetch import extract_with_llm
from backend.ml_engine import cluster_reviews, group_by_cluster


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _init_session_results() -> None:
    """Ensure all result keys exist in session state."""
    defaults = {
        "pipeline_results": None,    # final dict passed to analytics page
        "pipeline_ran": False,
        "pipeline_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_full_analysis_from_db(analysis_id: int) -> dict:
    """Reconstructs the pipeline_results dict from the database tables."""
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    
    # 1. Fetch the primary analytics metadata row
    meta_row = conn.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,)).fetchone()
    
    # 2. Fetch all child extraction complaints mapped to this run
    complaints_cursor = conn.execute(
        "SELECT cluster_id, feature_mentioned, sentiment_score, specific_complaint "
        "FROM extracted_complaints WHERE analysis_id = ?",
        (analysis_id,)
    )
    
    extractions = []
    for row in complaints_cursor:
        extractions.append({
            "cluster_id": row["cluster_id"],
            "Feature_Mentioned": row["feature_mentioned"],
            "Sentiment_Score": row["sentiment_score"],
            "Specific_Complaint": row["specific_complaint"]
        })
        
    conn.close()

    # Reassemble the dictionary structure required by analytics_page.py
    return {
        "source": meta_row["source"],
        "raw_rows": meta_row["raw_rows"],
        "filtered_rows": meta_row["filtered_rows"],
        "capped_rows": meta_row["capped_rows"],
        "optimal_k": meta_row["k_clusters"],
        "silhouette": meta_row["silhouette_score"],
        "engine_used": meta_row["engine_used"],
        "latency": meta_row["latency_seconds"],
        "text_conf": 1.0, 
        "rating_conf": 1.0,
        "text_col": "Database Reconstructed",
        "rating_col": "Database Reconstructed",
        "extractions": extractions
    }


# ─────────────────────────────────────────────────────────────────
# History Modal Dialog (This hides the table in a popup)
# ─────────────────────────────────────────────────────────────────

@st.dialog("◈ RECENT ANALYSIS HISTORY", width="large")
def render_history_dialog(username: str) -> None:
    """Renders an interactive database table showing past analysis jobs in an overlay."""
    conn = _get_connection()
    query = """
        SELECT id, created_at, source, capped_rows, engine_used 
        FROM analysis_history 
        WHERE username = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    """
    df_hist = pd.read_sql(query, conn, params=(username,))
    conn.close()

    if df_hist.empty:
        st.info("No previous processing runs found for this user account.")
        return

    # Column layout header matching
    cols = st.columns([1, 2, 2, 2, 2, 2])
    cols[0].write("**ID**")
    cols[1].write("**Timestamp**")
    cols[2].write("**Source**")
    cols[3].write("**Processed Rows**")
    cols[4].write("**Engine**")
    cols[5].write("**Action**")

    st.markdown("---")

    # Render dynamic layout elements with associated state loading actions
    for _, row in df_hist.iterrows():
        cols = st.columns([1, 2, 2, 2, 2, 2])
        cols[0].write(f"#{row['id']}")
        cols[1].write(row['created_at'].split()[0])
        cols[2].write(row['source'].upper())
        cols[3].write(f"{row['capped_rows']} records")
        
        engine_color = "#00e5ff" if row['engine_used'] == "LLM" else "#ff9100"
        cols[4].markdown(f"<span style='color:{engine_color}; font-weight:bold;'>{row['engine_used']}</span>", unsafe_allow_html=True)
        
        # This button loads the data AND switches the page
        if cols[5].button("View Dashboard", key=f"hist_btn_{row['id']}", use_container_width=True):
            with st.spinner("Reconstructing analytical dataset..."):
                full_results = get_full_analysis_from_db(row['id'])
                st.session_state["pipeline_results"] = full_results
                st.session_state["pipeline_ran"] = True
            
            # ---> AUTOMATIC REDIRECT TO ANALYTICS <---
            st.session_state["navigate_to"] = "Analytics" 
            st.rerun() 


# ─────────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────────

def _run_full_pipeline(
    df_raw: pd.DataFrame,
    source: str,
    groq_key: str,
) -> dict:
    """Execute the full FlawIntel pipeline and return a results dict."""
    t_start = time.perf_counter()

    # ── Step 1: Preprocess ───────────────────────────────────────
    (
        df_processed,
        text_col,
        rating_col,
        raw_rows,
        filtered_rows,
        capped_rows,
        text_conf,
        rating_conf,
    ) = preprocess_pipeline(df_raw)

    # ── Step 2: ML clustering ────────────────────────────────────
    df_clustered, optimal_k, silhouette, score_map, vectors = cluster_reviews(
        df_processed, text_col
    )

    # ── Step 3: Group by cluster ─────────────────────────────────
    cluster_groups = group_by_cluster(df_clustered, text_col)

    # ── Step 4: LLM / heuristic extraction ──────────────────────
    extractions, engine_used = extract_with_llm(cluster_groups, groq_key)

    latency = round(time.perf_counter() - t_start, 3)

    # ── Step 5: Persist to DB ────────────────────────────────────
    try:
        analysis_id = persist_analysis(
            session_id=st.session_state.get("session_id", "anonymous"),
            username=st.session_state.get("username", "unknown"),
            source=source,
            raw_rows=raw_rows,
            filtered_rows=filtered_rows,
            capped_rows=capped_rows,
            k_clusters=optimal_k,
            silhouette_score=silhouette,
            engine_used=engine_used,
            latency_seconds=latency,
            complaints=extractions,
        )
    except Exception:
        traceback.print_exc()
        analysis_id = None

    return {
        "df_raw": df_raw,
        "df_processed": df_processed,
        "df_clustered": df_clustered,
        "text_col": text_col,
        "rating_col": rating_col,
        "raw_rows": raw_rows,
        "filtered_rows": filtered_rows,
        "capped_rows": capped_rows,
        "text_conf": text_conf,
        "rating_conf": rating_conf,
        "optimal_k": optimal_k,
        "silhouette": silhouette,
        "score_map": score_map,
        "vectors": vectors,
        "cluster_groups": cluster_groups,
        "extractions": extractions,
        "engine_used": engine_used,
        "source": source,
        "latency": latency,
        "analysis_id": analysis_id,
    }


# ─────────────────────────────────────────────────────────────────
# Public renderer
# ─────────────────────────────────────────────────────────────────

def render_workspace_page() -> None:
    _init_session_results()

    # ── Secure Credentials Extraction ─────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    apify_key = os.getenv("APIFY_TOKEN")

    if not groq_key or not apify_key:
        st.error("Configuration Error: API keys missing from local backend environment (.env). Please configure GROQ_API_KEY and APIFY_TOKEN.")
        st.stop()

    current_user = st.session_state.get('username', 'unknown')

    # ── Sidebar Setup (This puts the history button on the left) ──
    with st.sidebar:
        st.markdown("### 🛠️ WORKSPACE TOOLS")
        if st.button("Watch History", width="stretch"):
            render_history_dialog(current_user)

    # ── Page header ───────────────────────────────────────────────
    st.markdown(
        """<div class='fi-zone-title'>◈ DATA INGESTION WORKSPACE</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"Logged in as **{current_user}**. "
        "Select your target data source to initialize the intelligence pipeline.",
    )

    # ── Source tabs ───────────────────────────────────────────────
    tab_csv, tab_url = st.tabs(["▸ CSV UPLOAD", "▸ LIVE URL FETCH"])

    df_raw: pd.DataFrame | None = None
    source: str = ""

    # ── Tab 1: CSV ────────────────────────────────────────────────
    with tab_csv:
        st.markdown(
            """
            <div class='fi-zone'>
              <div class='fi-zone-title'>▸ CSV FILE UPLOAD</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "Upload a CSV with at least one review-text column and (optionally) "
            "a star-rating column. Column names are auto-detected by the AI mapper."
        )

        uploaded_file = st.file_uploader(
            "Drop CSV file here",
            type=["csv"],
            key="csv_uploader",
        )

        if uploaded_file is not None:
            with st.spinner("Parsing CSV…"):
                df_tmp, err = load_csv(uploaded_file)
            if err:
                st.error(f"CSV error: {err}")
            else:
                st.success(f"Loaded **{len(df_tmp):,}** rows · **{len(df_tmp.columns)}** columns")
                st.dataframe(df_tmp.head(5), use_container_width=True)
                df_raw = df_tmp
                source = "csv"

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: URL ────────────────────────────────────────────────
    with tab_url:
        st.markdown(
            """
            <div class='fi-zone'>
              <div class='fi-zone-title'>▸ LIVE URL INGESTION</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "Paste an Amazon product or review page URL. "
            "Apify will scrape up to 200 reviews in real time."
        )

        url_input = st.text_input(
            "Product / Review URL",
            placeholder="https://www.amazon.com/dp/XXXXXXXXXX",
            key="apify_url_input",
        )

        fetch_btn = st.button("⬡ FETCH LIVE REVIEWS", key="fetch_apify_btn")

        if fetch_btn:
            if not url_input:
                st.error("Please enter a URL.")
            else:
                with st.spinner(f"Scraping '{url_input[:60]}…'  This may take 30–90 seconds."):
                    items, err = fetch_reviews_from_url(url_input, apify_key)

                if err:
                    st.error(f"Apify fetch error: {err}")
                elif not items:
                    st.warning("No reviews were returned from Apify.")
                else:
                    df_tmp, err2 = apify_to_dataframe(items)
                    if err2:
                        st.error(f"Conversion error: {err2}")
                    else:
                        st.success(f"Fetched **{len(df_tmp):,}** reviews from Apify.")
                        st.dataframe(df_tmp.head(5), use_container_width=True)
                        st.session_state["apify_df"] = df_tmp
                        df_raw = df_tmp  
                        source = "url"

        if "apify_df" in st.session_state and st.session_state["apify_df"] is not None:
            if df_raw is None: 
                df_raw = st.session_state["apify_df"]
                source = "url"

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Pipeline trigger ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """<div class='fi-zone-title'>◈ PIPELINE EXECUTION</div>""",
        unsafe_allow_html=True,
    )

    col_run, col_reset = st.columns([3, 1])

    with col_run:
        run_btn = st.button(
            "⬡ EXECUTE FLAWINTEL PIPELINE",
            key="run_pipeline_btn",
            disabled=(df_raw is None),
            width="stretch",
        )

    with col_reset:
        reset_btn = st.button(
            "↺ RESET",
            key="reset_btn",
            width="stretch",
        )

    if reset_btn:
        for key in ["pipeline_results", "pipeline_ran", "pipeline_error", "apify_df"]:
            st.session_state[key] = None if key != "pipeline_ran" else False
        st.rerun()

    if df_raw is None and not st.session_state.get("pipeline_ran"):
        st.info("Upload a CSV or fetch a URL to enable pipeline execution.")

    if run_btn and df_raw is not None:
        st.session_state["pipeline_error"] = None
        progress = st.progress(0, text="Initialising pipeline…")

        try:
            progress.progress(10, text="AI Column Mapping…")
            time.sleep(0.1)

            with st.spinner("Running full FlawIntel pipeline (this may take 60–120s)…"):
                progress.progress(25, text="Preprocessing & Filtering…")
                results = _run_full_pipeline(df_raw, source, groq_key)
                progress.progress(80, text="LLM Extraction…")

            progress.progress(100, text="Pipeline complete.")
            time.sleep(0.3)
            progress.empty()

            st.session_state["pipeline_results"] = results
            st.session_state["pipeline_ran"] = True

            st.success(
                f"✓ Pipeline complete in **{results['latency']:.2f}s** | "
                f"Engine: **{results['engine_used']}** | "
                f"Clusters: **{results['optimal_k']}**"
            )
            
            # ---> AUTOMATIC REDIRECT TO ANALYTICS <---
            st.session_state["navigate_to"] = "Analytics" 
            st.rerun() 

        except ValueError as ve:
            progress.empty()
            st.session_state["pipeline_error"] = str(ve)
            st.error(f"Pipeline error: {ve}")

        except Exception:
            progress.empty()
            err_msg = traceback.format_exc()
            print(err_msg)
            st.session_state["pipeline_error"] = "Unexpected pipeline failure."
            st.error("An unexpected error occurred. Check the terminal for the full traceback.")

    # ── Show last run summary if available ───────────────────────
    if st.session_state.get("pipeline_ran") and st.session_state.get("pipeline_results"):
        r = st.session_state["pipeline_results"]
        st.markdown("---")
        st.markdown(
            """<div class='fi-zone-title'>◈ LAST RUN SUMMARY</div>""",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", f"{r['raw_rows']:,}")
        c2.metric("Filtered", f"{r['filtered_rows']:,}")
        c3.metric("Capped", f"{r['capped_rows']}")
        c4.metric("Clusters k", f"{r['optimal_k']}")

    # NOTE: The render_history_section() call that was previously here has been completely removed!