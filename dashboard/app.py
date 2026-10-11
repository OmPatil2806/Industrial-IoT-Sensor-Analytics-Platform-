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
    st.Page(VIEWS / "risk.py", title="Machines at risk", icon="🚨"),
    st.Page(VIEWS / "machine.py", title="Machine detail", icon="🔧"),
    st.Page(VIEWS / "planner.py", title="Maintenance planner", icon="🗓️"),
    st.Page(VIEWS / "business.py", title="Business case", icon="💼"),
    st.Page(VIEWS / "health.py", title="Model & data health", icon="✅"),
]
st.navigation(pages).run()
