import pandas as pd
import streamlit as st

import client

st.title("🖥️ System & Analytics")

info = client.get("/system/info")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Device", info["device"].upper(), info["gpu"]["name"] if info.get("gpu") else None)
c2.metric("LLM", info["llm"]["name"], "online" if info["llm"]["available"] else "offline (extractive fallback)",
          delta_color="normal" if info["llm"]["available"] else "inverse")
c3.metric("Cache hit rate", f"{info['query_cache']['hit_rate']:.0%}", f"{info['query_cache']['size']} entries")
c4.metric("Auth", "API key" if info["auth_enabled"] else "open")
st.markdown("**Active models** — " + " · ".join(f"{k}: `{v}`" for k, v in info["active_models"].items()))

st.divider()
days = st.slider("Window (days)", 1, 90, 30)
a = client.get("/analytics/queries", days=days)
s = a["summary"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Queries", s["total"] or 0)
k2.metric("Avg latency", f"{(s['avg_latency_ms'] or 0):.0f} ms")
k3.metric("Avg generation", f"{(s['avg_generation_ms'] or 0):.0f} ms")
k4.metric("👍 / 👎", f"{s['positive'] or 0} / {s['negative'] or 0}")
k5.metric("No-result queries", s["no_result"] or 0)
if a["daily"]:
    st.bar_chart(pd.DataFrame(a["daily"]).set_index("day")["queries"])
st.markdown("**Knowledge gaps** — recent questions with low retrieval confidence (candidates for new documentation)")
st.dataframe(pd.DataFrame(a["knowledge_gaps"]), hide_index=True, width="stretch")
st.markdown("**Recent queries**")
if a["recent"]:
    df = pd.DataFrame(a["recent"])
    df["when"] = df["created_at"].map(client.fmt_time)
    st.dataframe(df[["when", "collection", "question", "num_sources", "top_score", "latency_ms", "cached", "feedback"]],
                 hide_index=True, width="stretch")

st.divider()
st.subheader("Background jobs")
jobs = client.get("/jobs", limit=50)
if jobs:
    df = pd.DataFrame(jobs)
    df["created"] = df["created_at"].map(client.fmt_time)
    st.dataframe(df[["id", "type", "status", "progress", "message", "error", "created"]], hide_index=True,
                 width="stretch")
    running = [j["id"] for j in jobs if j["status"] in ("queued", "running")]
    if running:
        jid = st.selectbox("Cancel job", running)
        if st.button("Cancel"):
            client.post(f"/jobs/{jid}/cancel")
            st.rerun()
