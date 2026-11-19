import streamlit as st

import client

st.title("💬 Ask your organisation's knowledge")

try:
    collections = [c["name"] for c in client.get("/collections")]
except Exception as exc:
    st.error(f"Backend unavailable: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Scope")
    selected = st.multiselect("Collections", collections, default=collections[:1])
    with st.expander("Retrieval overrides"):
        mode = st.selectbox("Mode", ["(default)", "hybrid", "vector", "keyword"])
        top_k = st.slider("Passages to LLM", 1, 15, 5)
        rerank = st.selectbox("Reranker", ["(default)", "on", "off"])
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

overrides = {"top_k": top_k}
if mode != "(default)":
    overrides["mode"] = mode
if rerank != "(default)":
    overrides["use_reranker"] = rerank == "on"

tab_chat, tab_search = st.tabs(["Chat", "Semantic search"])


def render_sources(sources: list[dict], cited: list[int] | None = None):
    cited = set(cited or [])
    for s in sources:
        tag = "✅ cited · " if s["ref"] in cited else ""
        loc = f"{s['source']}" + (f" · p.{s['page']}" if s.get("page") else "") + (f" · {s['section']}" if s.get("section") else "")
        with st.expander(f"[{s['ref']}] {tag}{loc} — score {s['score']:.3f}"):
            st.write(s["text"])
            st.caption(f"vector {s.get('vector_score')} · keyword {s.get('keyword_score')} · rerank {s.get('rerank_score')} · "
                       f"collection {s.get('collection')} · chunk {s['chunk_id']}")


with tab_chat:
    st.session_state.setdefault("messages", [])
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                meta = m["meta"]
                st.caption(f"model `{meta.get('model')}` · confidence {meta.get('confidence', 0):.2f} · "
                           f"{meta.get('latency', {}).get('total_ms', 0):.0f} ms")
                render_sources(m.get("sources", []), meta.get("cited_refs"))
                qid = meta.get("query_id")
                if qid:
                    c1, c2, _ = st.columns([1, 1, 10])
                    if c1.button("👍", key=f"up{i}"):
                        client.post("/feedback", {"query_id": qid, "rating": 1})
                        st.toast("Thanks! Positive answers feed future fine-tuning datasets.")
                    if c2.button("👎", key=f"down{i}"):
                        client.post("/feedback", {"query_id": qid, "rating": -1})
                        st.toast("Feedback recorded.")

    if question := st.chat_input("Ask a question about your documents…", disabled=not selected):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            placeholder, sources, meta, text = st.empty(), [], {}, ""
            try:
                with st.spinner("Searching knowledge base…"):
                    stream = client.ask_stream({"query": question, "collections": selected, "overrides": overrides,
                                                "history": history})
                    for event, data in stream:
                        if event == "sources":
                            sources = data["sources"]
                            break
                for event, data in stream:
                    if event == "token":
                        text += data["text"]
                        placeholder.markdown(text + "▌")
                    elif event == "done":
                        meta = data
                    elif event == "error":
                        st.warning(data["message"])
                placeholder.markdown(text)
            except client.APIError as exc:
                text = f"⚠️ {exc}"
                placeholder.error(text)
        st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources, "meta": meta})
        st.rerun()

with tab_search:
    q = st.text_input("Search query", placeholder="e.g. backup retention period")
    if q and selected:
        res = client.post("/search", {"query": q, "collections": selected, "overrides": overrides})
        st.caption(" · ".join(f"{k} {v:.0f} ms" for k, v in res["latency"].items()))
        if not res["results"]:
            st.info("No matching passages.")
        render_sources(res["results"])
