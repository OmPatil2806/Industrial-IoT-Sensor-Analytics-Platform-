"""Industrial IoT Sensor Analytics: dashboard entry point.

Run from the project folder with:  iiot dashboard
(or: streamlit run dashboard/app.py)
"""

from pathlib import Path

import streamlit as st

VIEWS = Path(__file__).parent / "views"

st.set_page_config(
    page_title="IIoT Sensor Analytics",
    page_icon="🏭",
    layout="wide",
)

pages = [
    st.Page(VIEWS / "overview.py", title="Fleet overview", icon="📊", default=True),
    st.Page(VIEWS / "home.py", title="Platform status", icon="✅"),
]
st.navigation(pages).run()
