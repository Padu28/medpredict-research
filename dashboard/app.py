"""MedPredict Research dashboard (educational, non-clinical).

Run:  streamlit run dashboard/app.py --server.port 8504
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="MedPredict Research · Reliable Risk Models", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

import theme  # noqa: E402

import data  # noqa: E402

theme.apply_theme()
data.ensure_pipeline()

pages = {
    "Study": [
        st.Page("views/overview.py", title="Overview", icon=":material/monitor_heart:", default=True),
        st.Page("views/data_audit.py", title="Data Audit", icon=":material/fact_check:", url_path="data_audit"),
        st.Page("views/discrimination.py", title="Model Comparison", icon=":material/leaderboard:", url_path="discrimination"),
        st.Page("views/calibration.py", title="Calibration", icon=":material/tune:", url_path="calibration"),
    ],
    "Reliability": [
        st.Page("views/subgroups.py", title="Subgroups & Fairness", icon=":material/diversity_3:", url_path="subgroups"),
        st.Page("views/shift_tests.py", title="Shift Tests", icon=":material/swap_horiz:", url_path="shift_tests"),
        st.Page("views/explainability.py", title="Explainability", icon=":material/psychology:", url_path="explainability"),
        st.Page("views/risk_explorer.py", title="Risk Explorer (research)", icon=":material/science:", url_path="risk_explorer"),
    ],
}

with st.sidebar:
    theme.sidebar_brand()

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    s = data.summary()
    p = s.get("primary", {})
    st.caption(f"Selected: {p.get('model', '—')} + {p.get('calibration', '—')} · thr {p.get('threshold', 0):.3f}")
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown('<div class="disclaimer"><b>Not for clinical use.</b> Educational research prototype on a synthetic, '
                'de-identified cohort. No diagnosis or treatment recommendation is displayed.</div>', unsafe_allow_html=True)

theme.safety_banner()
nav.run()
theme.footer()
