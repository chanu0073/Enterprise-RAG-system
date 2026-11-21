"""Enterprise Knowledge Intelligence Platform — Streamlit UI."""

import streamlit as st

import client

st.set_page_config(page_title="Knowledge Intelligence", page_icon="🧠", layout="wide")

pages = {
    "Use": [
        st.Page("views/ask.py", title="Ask & Search", icon="💬", default=True),
        st.Page("views/knowledge.py", title="Knowledge Base", icon="📚"),
    ],
    "Improve": [
        st.Page("views/finetune.py", title="Datasets & Fine-tuning", icon="🧪"),
        st.Page("views/evaluation.py", title="Evaluation", icon="📊"),
        # st.Page("views/optimization.py", title="Optimization", icon="⚙️"),  # hidden; API /optimize still works
    ],
    "Operate": [st.Page("views/system.py", title="System & Analytics", icon="🖥️")],
}

with st.sidebar:
    st.markdown("### 🧠 Knowledge Intelligence")
    if client.healthy():
        st.caption(f"🟢 Backend connected · `{client.API_URL}`")
    else:
        st.caption(f"🔴 Backend unreachable · `{client.API_URL}`")

st.navigation(pages).run()
