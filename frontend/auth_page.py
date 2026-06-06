"""
frontend/auth_page.py
──────────────────────────────────────────────────────────────────
Renders the login / registration UI.
Writes session state keys:
  st.session_state.authenticated  → bool
  st.session_state.session_id     → str
  st.session_state.username       → str
"""

import streamlit as st

from backend.auth import authenticate_user, register_user


# ─────────────────────────────────────────────────────────────────
# Public renderer
# ─────────────────────────────────────────────────────────────────

def render_auth_page() -> None:
    """Entry point called by app.py when the user is not authenticated."""

    st.markdown(
        """
        <div class="fi-auth-container">
          <div class="fi-auth-title">⬡ FLAWINTEL</div>
          <div class="fi-auth-sub">ENTERPRISE DEFECT INTELLIGENCE PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_login, tab_register = st.tabs(["▸ LOGIN", "▸ REGISTER"])

    # ── Login tab ────────────────────────────────────────────────
    with tab_login:
        st.markdown("#### Access your workspace")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="your_username",
                key="login_username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="••••••••",
                key="login_password",
            )
            submit_login = st.form_submit_button(
                "AUTHENTICATE", use_container_width=True
            )

        if submit_login:
            if not username or not password:
                st.error("Both username and password are required.")
            else:
                with st.spinner("Verifying credentials…"):
                    success, result = authenticate_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.session_id = result
                    st.session_state.username = username.strip()
                    st.session_state.active_page = "workspace"
                    st.success(f"Welcome back, {username}.")
                    st.rerun()
                else:
                    st.error(f"Authentication failed: {result}")

    # ── Register tab ─────────────────────────────────────────────
    with tab_register:
        st.markdown("#### Create a new account")
        with st.form("register_form", clear_on_submit=True):
            new_user = st.text_input(
                "Choose a username",
                placeholder="min. 3 characters",
                key="reg_username",
            )
            new_pass = st.text_input(
                "Choose a password",
                type="password",
                placeholder="min. 8 characters",
                key="reg_password",
            )
            confirm_pass = st.text_input(
                "Confirm password",
                type="password",
                placeholder="repeat password",
                key="reg_confirm",
            )
            submit_register = st.form_submit_button(
                "CREATE ACCOUNT", use_container_width=True
            )

        if submit_register:
            if new_pass != confirm_pass:
                st.error("Passwords do not match.")
            elif not new_user or not new_pass:
                st.error("All fields are required.")
            else:
                with st.spinner("Creating account…"):
                    success, message = register_user(new_user, new_pass)
                if success:
                    st.success(f"{message} You can now log in.")
                else:
                    st.error(message)

    # ── Footer ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; margin-top:32px;'>
          <span style='font-family:monospace; font-size:11px;
                       color:#334455; letter-spacing:2px;'>
            FLAWINTEL v1.0 · ANTHROPIC POWERED · ALL RIGHTS RESERVED
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )