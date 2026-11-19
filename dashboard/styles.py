from pathlib import Path

import plotly.express as px
import streamlit as st

from dashboard.common import BASE_COLORS

THEME_PATH = Path(__file__).with_name("assets") / "theme.css"


def inject_global_css():
    px.defaults.color_discrete_sequence = BASE_COLORS
    theme = THEME_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{theme}</style>", unsafe_allow_html=True)

