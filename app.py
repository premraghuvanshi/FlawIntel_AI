"""
app.py — FlawIntel Root Orchestrator
──────────────────────────────────────────────────────────────────
Zero business logic. Sole responsibilities:
  1. Bootstrap the SQLite schema on first run
  2. Inject global CSS
  3. Manage session state (auth, active_page)
  4. Route to the correct frontend page
  5. Render the navigation sidebar

Run: streamlit run app.py
"""

 # check if the .env file is present and if the API keys are present
import os
from dotenv import load_dotenv
load_dotenv()
if not os.getenv("GROQ_API_KEY") or not os.getenv("APIFY_TOKEN"):
    print("WARNING: Missing API keys in root .env file.")
import sys
from pathlib import Path

# ── Ensure the project root is on the import path ────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Streamlit page config (MUST be first Streamlit call) ─────────
st.set_page_config(
    page_title="FlawIntel AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend bootstrap ─────────────────────────────────────────────
from backend.auth import bootstrap_schema, invalidate_session

try:
    bootstrap_schema()
except Exception as exc:
    st.error(f"Fatal: Database bootstrap failed — {exc}")
    st.stop()

# ── Frontend imports ───────────────────────────────────────────────
from frontend.auth_page import render_auth_page
from frontend.workspace_page import render_workspace_page
from frontend.analytics_page import render_analytics_page


# ─────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    css_path = ROOT / "frontend" / "style.css"
    try:
        css_text = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found — running with default theme.")


# ─────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "authenticated": False,
        "session_id":    None,
        "username":      None,
        "active_page":   "workspace",   # "workspace" | "analytics"
        "pipeline_results": None,
        "pipeline_ran":  False,
        "pipeline_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────
# Navigation sidebar
# ─────────────────────────────────────────────────────────────────

def _render_nav() -> None:
    with st.sidebar:
        # Brand header
        st.markdown(
            """
            <div class="fi-header">
              <span class="fi-logo-glyph">⬡</span>
              <div>
                <div class="fi-brand-name">FlawIntel AI</div>
                <div class="fi-brand-sub">Defect Intelligence</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style='font-family:monospace; font-size:11px;
                        color:#334455; margin-bottom:12px;
                        letter-spacing:1px;'>
              USER: <span style='color:#00e5ff;'>
                {st.session_state.get("username", "—")}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Page navigation
        st.markdown(
            "<div style='font-size:10px; color:#445566; "
            "letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;'>"
            "NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        if st.button("⬡ Workspace", use_container_width=True, key="nav_workspace"):
            st.session_state["active_page"] = "workspace"
            st.rerun()

        if st.button("◈ Analytics", use_container_width=True, key="nav_analytics"):
            st.session_state["active_page"] = "analytics"
            st.rerun()

        st.markdown("---")

        # Analytics readiness indicator
        if st.session_state.get("pipeline_ran"):
            st.markdown(
                "<span class='fi-badge fi-badge-llm'>PIPELINE COMPLETE</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='fi-badge fi-badge-error'>NO DATA</span>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Logout
        if st.button("↩ LOGOUT", use_container_width=True, key="nav_logout"):
            sid = st.session_state.get("session_id")
            if sid:
                invalidate_session(sid)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# Main router — zero business logic
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    _init_session()


    # Respect automatic navigation requests from other pages (e.g., history dialog)
    nav_target = st.session_state.get("navigate_to")
    if nav_target:
        # Normalize to the same keys used by the router
        st.session_state["active_page"] = str(nav_target).lower()
        # Clear the flag so the redirect happens only once
        del st.session_state["navigate_to"]


    # Unauthenticated gate
    if not st.session_state.get("authenticated"):
        render_auth_page()
        return

    # Authenticated: render nav + route to page
    _render_nav()

    active = st.session_state.get("active_page", "workspace")

    if active == "workspace":
        render_workspace_page()
    elif active == "analytics":
        render_analytics_page()
    else:
        # Fallback safety net
        st.session_state["active_page"] = "workspace"
        render_workspace_page()


if __name__ == "__main__":
    main()