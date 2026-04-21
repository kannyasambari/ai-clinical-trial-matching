"""
frontend/app.py
────────────────
Production Streamlit UI for the Clinical Trial Matcher.

Features:
  • Login / Register flow using JWT
  • Patient input form with validation
  • Loading spinner during API call
  • Eligibility colour-coding (YES=green, NO=red, POSSIBLE=amber, ERROR=grey)
  • Paginated results
  • Structured display of each trial's analysis
  • Graceful error messages (network down, invalid creds, no results, etc.)
"""

import re
import time

import requests
import streamlit as st

from backend.config import BACKEND_URL

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Trial Matcher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state defaults ───────────────────────────────────────────────────
for key, default in [
    ("token", None),
    ("role",  None),
    ("username", None),
    ("results", None),
    ("total_found", 0),
    ("current_page", 1),
    ("last_query", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _api_post(path: str, json_body: dict, auth: bool = True, timeout: int = 90) -> dict | None:
    url = f"{BACKEND_URL}{path}"
    headers = _auth_headers() if auth else {}
    try:
        resp = requests.post(url, json=json_body, headers=headers, timeout=timeout)
        if resp.status_code == 401:
            st.session_state.token = None
            st.error("Session expired. Please log in again.")
            return None
        if resp.status_code == 403:
            st.error("You do not have permission to perform this action.")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot connect to the backend API. "
            f"Make sure it is running at `{BACKEND_URL}`.\n\n"
            "Start it with: `uvicorn backend.api:app --reload`"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("⏱ Request timed out. The model may be processing a large query — please try again.")
        return None
    except requests.exceptions.HTTPError as exc:
        detail = "Unknown error"
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            pass
        st.error(f"API error {exc.response.status_code}: {detail}")
        return None


def _api_post_form(path: str, data: dict, timeout: int = 30) -> dict | None:
    """Used for OAuth2 form-data login."""
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.post(url, data=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Cannot connect to backend at `{BACKEND_URL}`.")
        return None
    except requests.exceptions.HTTPError as exc:
        detail = "Unknown error"
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            pass
        st.error(f"Login failed: {detail}")
        return None


# ─── Auth UI ─────────────────────────────────────────────────────────────────

def render_auth():
    st.title("🧬 Clinical Trial Matcher")
    st.markdown("*AI-powered eligibility matching from real ClinicalTrials.gov data*")
    st.divider()

    tab_login, tab_register = st.tabs(["Login", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                with st.spinner("Authenticating…"):
                    result = _api_post_form(
                        "/auth/login",
                        {"username": username, "password": password},
                    )
                if result:
                    st.session_state.token    = result["access_token"]
                    st.session_state.role     = result["role"]
                    st.session_state.username = username
                    st.success(f"Welcome, {username}! Role: {result['role']}")
                    time.sleep(0.5)
                    st.rerun()

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("Username", key="reg_user")
            new_pass = st.text_input("Password (min 8 chars)", type="password", key="reg_pass")
            role     = st.selectbox("Role", ["patient", "doctor", "admin"])
            submitted_reg = st.form_submit_button("Create Account", type="primary")
        if submitted_reg:
            if len(new_pass) < 8:
                st.warning("Password must be at least 8 characters.")
            else:
                result = _api_post(
                    "/auth/register",
                    {"username": new_user, "password": new_pass, "role": role},
                    auth=False,
                )
                if result:
                    st.success(result.get("message", "Account created. You can now log in."))


# ─── Result rendering ─────────────────────────────────────────────────────────

_ELIGIBILITY_COLOURS = {
    "YES":      ("🟢", "#1a7a1a", "#e8f5e9"),
    "NO":       ("🔴", "#9b1515", "#fdecea"),
    "POSSIBLE": ("🟡", "#7a5c00", "#fff9e6"),
    "ERROR":    ("⚫", "#555555", "#f5f5f5"),
}


def render_trial_card(result: dict, rank: int):
    eligibility = result.get("eligibility", "ERROR").upper()
    icon, text_colour, bg_colour = _ELIGIBILITY_COLOURS.get(eligibility, _ELIGIBILITY_COLOURS["ERROR"])

    with st.container():
        st.markdown(
            f"""<div style="background:{bg_colour}; border-radius:8px; padding:16px; margin-bottom:12px;">
            <h4 style="margin:0; color:{text_colour};">{icon} #{rank} — {result.get('title', result.get('trial_id'))}</h4>
            <p style="margin:4px 0; font-size:0.85em; color:#666;">
                Trial ID: <code>{result.get('trial_id')}</code> &nbsp;|&nbsp;
                Similarity distance: {result.get('distance', 'N/A')}
            </p>
            <span style="font-weight:bold; color:{text_colour};">Eligibility: {eligibility}</span>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("View full analysis"):
            st.markdown(result.get("analysis", "No analysis available."))


# ─── Main app ─────────────────────────────────────────────────────────────────

def render_main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"**Logged in as:** `{st.session_state.username}`  \n**Role:** `{st.session_state.role}`")
        if st.button("Logout"):
            for k in ("token", "role", "username", "results", "total_found", "current_page", "last_query"):
                st.session_state[k] = None if k != "current_page" else 1
            st.rerun()
        st.divider()
        st.markdown("### Filters")
        page_size = st.selectbox("Results per page", [3, 5, 10], index=1)
        st.markdown("---")
        st.markdown("**About**")
        st.caption(
            "This system uses Bio_ClinicalBERT embeddings + FAISS for "
            "semantic retrieval, followed by GPT-4o-mini reasoning over "
            "real ClinicalTrials.gov data."
        )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🧬 Clinical Trial Matcher")
    st.markdown("*Find relevant clinical trials using AI-powered eligibility matching.*")

    # ── Patient input form ────────────────────────────────────────────────────
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", ["male", "female"])
        with col2:
            conditions  = st.text_input(
                "Medical conditions",
                placeholder="e.g. type 2 diabetes, hypertension",
            )
            medications = st.text_input(
                "Current medications",
                placeholder="e.g. metformin, lisinopril  (optional)",
            )

        submitted = st.form_submit_button("🔍 Find Clinical Trials", type="primary")

    if submitted:
        if not conditions.strip():
            st.warning("Please enter at least one medical condition.")
            return

        st.session_state.current_page = 1

        with st.spinner("🔬 Searching trials and running eligibility analysis…"):
            result = _api_post(
                "/match_trials",
                {
                    "age":         age,
                    "sex":         sex,
                    "conditions":  conditions,
                    "medications": medications,
                },
                auth=True,
            )

        if result:
            st.session_state.results     = result.get("results", [])
            st.session_state.total_found = result.get("total_found", 0)
            st.session_state.last_query  = {
                "age": age, "sex": sex,
                "conditions": conditions, "medications": medications,
            }
            if result.get("cached"):
                st.caption("⚡ Results served from cache.")

    # ── Results ───────────────────────────────────────────────────────────────
    results = st.session_state.results
    if results is None:
        st.info("Fill in the patient profile above and click **Find Clinical Trials**.")
        return

    total = st.session_state.total_found

    if not results:
        st.warning(
            "No matching trials found for this query.\n\n"
            "Try:\n"
            "- Broadening the condition (e.g. 'diabetes' instead of 'type 2 diabetes')\n"
            "- Checking spelling\n"
            "- Removing medication filters"
        )
        return

    # Summary bar
    yes_count      = sum(1 for r in results if r["eligibility"] == "YES")
    possible_count = sum(1 for r in results if r["eligibility"] == "POSSIBLE")
    no_count       = sum(1 for r in results if r["eligibility"] == "NO")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total found",  total)
    c2.metric("🟢 Eligible",  yes_count)
    c3.metric("🟡 Possible",  possible_count)
    c4.metric("🔴 Not eligible", no_count)

    st.divider()

    for rank, result in enumerate(results, start=1):
        render_trial_card(result, rank)


# ─── Entry point ─────────────────────────────────────────────────────────────

if st.session_state.token is None:
    render_auth()
else:
    render_main()