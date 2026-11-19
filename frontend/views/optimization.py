import pandas as pd
import streamlit as st

import client

st.title("⚙️ Optimization")
st.caption("Search the retrieval configuration space (chunking, retrieval mode, fusion, depth, reranking) against your "
           "dataset. Tuning uses the train split; the winner is confirmed on the held-out test split.")

cfg = client.get("/config/retrieval")
with st.expander("Current retrieval configuration", expanded=False):
    with st.form("cfg"):
        c1, c2, c3 = st.columns(3)
        chunk_size = c1.number_input("Chunk size", 100, 4000, cfg["chunk_size"], step=50)
        chunk_overlap = c2.number_input("Chunk overlap", 0, 1000, cfg["chunk_overlap"], step=10)
        mode = c3.selectbox("Mode", ["hybrid", "vector", "keyword"], index=["hybrid", "vector", "keyword"].index(cfg["mode"]))
        fusion = c1.selectbox("Fusion", ["rrf", "weighted"], index=["rrf", "weighted"].index(cfg["fusion"]))
        alpha = c2.slider("Vector weight (weighted fusion)", 0.0, 1.0, cfg["alpha"])
        candidate_k = c3.number_input("Candidates per retriever", 1, 200, cfg["candidate_k"])
        top_k = c1.number_input("Passages to LLM", 1, 50, cfg["top_k"])
        use_reranker = c2.checkbox("Use reranker", cfg["use_reranker"])
        boost = c3.slider("Identifier boost", 0.0, 2.0, cfg.get("identifier_boost", 0.5))
        if st.form_submit_button("Save configuration"):
            res = client.put("/config/retrieval", {**cfg, "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
                                                   "mode": mode, "fusion": fusion, "alpha": alpha, "candidate_k": int(candidate_k),
                                                   "top_k": int(top_k), "use_reranker": use_reranker, "identifier_boost": boost})
            for jid in res["reindex_jobs"]:
                client.track_job({"id": jid}, "Re-indexing (chunking changed)")
            st.success("Saved")

datasets = client.get("/datasets")
if not datasets:
    st.info("Create a dataset first.")
    st.stop()
ds_map = {d["id"]: f"{d['name']} ({d['num_items']} items)" for d in datasets}

with st.form("opt"):
    ds_id = st.selectbox("Dataset", list(ds_map), format_func=ds_map.get)
    c1, c2 = st.columns(2)
    chunk_sizes = c1.multiselect("Chunk sizes", [300, 500, 800, 1200, 1600], default=[cfg["chunk_size"]])
    modes = c2.multiselect("Modes", ["hybrid", "vector", "keyword"], default=["hybrid", "vector", "keyword"])
    fusions = c1.multiselect("Fusion", ["rrf", "weighted"], default=["rrf", "weighted"])
    alphas = c2.multiselect("Vector weights", [0.2, 0.3, 0.5, 0.7, 0.8], default=[0.3, 0.5, 0.7])
    cks = c1.multiselect("Candidate depth", [5, 10, 20, 40, 80], default=[10, 20, 40])
    rr = c2.multiselect("Reranking", [True, False], default=[True, False])
    metric = c1.selectbox("Objective metric", ["ndcg@10", "mrr", "hit@1", "hit@3", "hit@5", "recall@5"])
    lw = c2.number_input("Latency penalty (metric points per second)", 0.0, 10.0, 0.0, step=0.05,
                         help="objective = metric − penalty × latency_seconds. Use e.g. 0.1 to favour faster pipelines.")
    trials = st.slider("Max trials", 5, 300, 80)
    if st.form_submit_button("Run optimization", type="primary"):
        job = client.post("/optimize", {"dataset_id": ds_id, "chunk_sizes": chunk_sizes or None, "modes": modes or None,
                                        "fusions": fusions or None, "alphas": alphas or None, "candidate_ks": cks or None,
                                        "rerank_options": rr or None, "metric": metric, "latency_weight": lw,
                                        "max_trials": trials})
        if res := client.track_job(job, "Optimizing"):
            st.session_state["opt_run"] = res["id"]

runs = client.get("/eval/runs", kind="optimization", limit=20)
if runs:
    labels = {r["id"]: f"{r['name']} · {client.fmt_time(r['created_at'])}" for r in runs}
    default = st.session_state.get("opt_run", runs[0]["id"])
    run_id = st.selectbox("Optimization run", list(labels), index=list(labels).index(default) if default in labels else 0,
                          format_func=labels.get)
    run = client.get(f"/eval/runs/{run_id}")
    m = run["metrics"]
    st.subheader("Best configuration")
    st.json(m["best"])
    if m.get("confirmation"):
        st.markdown("**Held-out confirmation (test split)**")
        st.dataframe(pd.DataFrame(m["confirmation"]).T, width="stretch")
    if m.get("requires_reindex"):
        st.warning("Applying this configuration changes chunking and will re-index all collections.")
    if st.button("Apply best configuration", type="primary"):
        res = client.post(f"/optimize/{run_id}/apply")
        for jid in res["reindex_jobs"]:
            client.track_job({"id": jid}, "Re-indexing")
        st.success("Applied")
    lb = pd.DataFrame(run["details"])
    st.markdown("**Leaderboard**")
    st.dataframe(lb, hide_index=True, width="stretch", height=350)
    metric = run["config"]["metric"]
    if "est_latency_ms" in lb:
        st.markdown(f"**Quality vs latency** ({metric})")
        st.scatter_chart(lb, x="est_latency_ms", y=metric, color="mode")
