import pandas as pd
import streamlit as st

import client

st.title("📚 Knowledge Base")
st.caption("Collections group documents (e.g. per department). Supported: PDF, DOCX, Markdown, TXT, HTML, CSV.")

collections = client.get("/collections")
names = [c["name"] for c in collections]

c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(
        pd.DataFrame(collections)[["name", "description", "documents", "chunks"]] if collections else pd.DataFrame(),
        hide_index=True, width="stretch",
    )
with c2:
    with st.form("new_collection", clear_on_submit=True):
        st.markdown("**New collection**")
        name = st.text_input("Name", placeholder="hr-policies")
        desc = st.text_input("Description")
        if st.form_submit_button("Create") and name:
            try:
                client.post("/collections", {"name": name, "description": desc})
                st.success(f"Created {name}")
                st.rerun()
            except client.APIError as exc:
                st.error(str(exc))

st.divider()
collection = st.selectbox("Collection", names)
if not collection:
    st.stop()
info = client.get(f"/collections/{collection}")
idx = info["index"]
m1, m2, m3 = st.columns(3)
m1.metric("Documents", info["documents"])
m2.metric("Chunks", idx["num_chunks"])
m3.metric("Embedding model", idx["embedder"].split("/")[-1])
if idx["chunking_stale"]:
    st.warning("Chunking settings changed since this collection was indexed. Re-index to apply them.")

files = st.file_uploader("Upload documents", accept_multiple_files=True,
                         type=["pdf", "docx", "md", "markdown", "txt", "html", "htm", "csv"])
if files and st.button(f"Ingest {len(files)} file(s)", type="primary"):
    res = client.upload(collection, files)
    for r in res["results"]:
        if r.get("error"):
            st.error(f"{r['filename']}: {r['error']}")
        elif r.get("duplicate"):
            st.info(f"{r['filename']}: already in collection (skipped)")
        elif r.get("job_id"):
            client.track_job({"id": r["job_id"]}, f"Indexing {r['filename']}")
    st.rerun()

docs = client.get("/documents", collection=collection)
if docs:
    df = pd.DataFrame(docs)
    df["created"] = df["created_at"].map(client.fmt_time)
    df["size_kb"] = (df["size_bytes"] / 1024).round(1)
    st.dataframe(df[["filename", "status", "num_pages", "num_chunks", "size_kb", "created", "error", "id"]],
                 hide_index=True, width="stretch")
    c1, c2, c3 = st.columns([3, 1, 1])
    doc_label = {d["id"]: d["filename"] for d in docs}
    doc_id = c1.selectbox("Document", list(doc_label), format_func=doc_label.get)
    if c2.button("Preview chunks", width="stretch"):
        for ch in client.get(f"/documents/{doc_id}/chunks", limit=20):
            with st.expander(f"#{ch['chunk_index']} · p.{ch['page']} · {ch['metadata'].get('heading') or ''}"):
                st.write(ch["text"])
    if c3.button("Delete", width="stretch", type="secondary"):
        client.delete(f"/documents/{doc_id}")
        st.rerun()
else:
    st.info("No documents yet — upload some above.")

if st.button("Re-index collection"):
    client.track_job(client.post(f"/collections/{collection}/reindex"), "Re-indexing")
    st.rerun()
