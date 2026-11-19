import pandas as pd
import streamlit as st

import client

st.title("📊 Evaluation")
st.caption("Retrieval metrics use span overlap with the gold passage, so results stay comparable across chunk sizes "
           "and models. Generation metrics check grounding, relevance, citation and refusal behaviour.")

datasets = client.get("/datasets")
if not datasets:
    st.info("Create a dataset on the *Datasets & Fine-tuning* page first.")
    st.stop()
ds_map = {d["id"]: f"{d['name']} ({d['num_items']} items)" for d in datasets}
models = client.get("/models")

tab_run, tab_history, tab_compare = st.tabs(["Run", "History", "Compare embedding models"])

with tab_run:
    kind = st.radio("Evaluation type", ["Retrieval", "End-to-end answers"], horizontal=True)
    with st.form("eval"):
        ds_id = st.selectbox("Dataset", list(ds_map), format_func=ds_map.get)
        split = st.selectbox("Split", ["test", "train", "all"])
        if kind == "Retrieval":
            emb = {m["id"]: m["name"] for m in models if m["kind"] == "embedding"}
            rr = {m["id"]: m["name"] for m in models if m["kind"] == "reranker"}
            c1, c2 = st.columns(2)
            emb_id = c1.selectbox("Embedding model", [None, *emb], format_func=lambda x: "(active)" if x is None else emb[x])
            rr_id = c2.selectbox("Reranker", [None, *rr], format_func=lambda x: "(active)" if x is None else rr[x])
            mode = st.selectbox("Mode", ["(config)", "hybrid", "vector", "keyword"])
            use_rr = st.selectbox("Use reranker", ["(config)", "yes", "no"])
        else:
            max_items = st.slider("Max questions", 5, 200, 30)
            judge = st.checkbox("Use LLM-as-judge (slower)", value=False)
        if st.form_submit_button("Run evaluation", type="primary"):
            if kind == "Retrieval":
                ov = {}
                if mode != "(config)":
                    ov["mode"] = mode
                if use_rr != "(config)":
                    ov["use_reranker"] = use_rr == "yes"
                job = client.post("/eval/retrieval", {"dataset_id": ds_id, "split": split, "overrides": ov or None,
                                                      "embedding_model_id": emb_id, "reranker_model_id": rr_id})
            else:
                job = client.post("/eval/generation", {"dataset_id": ds_id, "split": split, "max_items": max_items,
                                                       "use_llm_judge": judge})
            run = client.track_job(job, "Evaluating")
            if run:
                st.session_state["last_run"] = run["id"]

    run_id = st.session_state.get("last_run")
    if run_id:
        run = client.get(f"/eval/runs/{run_id}")
        m = run["metrics"]
        st.subheader(run["name"])
        keys = [k for k in ("hit@1", "hit@5", "mrr", "ndcg@10", "recall@5", "faithfulness", "answer_relevance",
                            "citation_rate", "refusal_accuracy", "false_refusal_rate", "answer_f1", "judge_correctness",
                            "latency_p95_ms") if k in m]
        cols = st.columns(min(len(keys), 6) or 1)
        for i, k in enumerate(keys):
            cols[i % len(cols)].metric(k, f"{m[k]:.0f}" if k.endswith("_ms") else f"{m[k]:.3f}")
        if m.get("by_category"):
            st.markdown("**By question category**")
            st.dataframe(pd.DataFrame(m["by_category"]).T, width="stretch")
        st.markdown("**Per question**")
        st.dataframe(pd.DataFrame(run["details"]), hide_index=True, width="stretch", height=320)

with tab_history:
    runs = client.get("/eval/runs", limit=100)
    runs = [r for r in runs if r["kind"] in ("retrieval", "generation")]
    if runs:
        rows = []
        for r in runs:
            row = {"id": r["id"], "kind": r["kind"], "name": r["name"], "when": client.fmt_time(r["created_at"])}
            row.update({k: round(v, 4) for k, v in r["metrics"].items() if isinstance(v, (int, float))})
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        ret = df[df.kind == "retrieval"]
        if len(ret) > 1 and "mrr" in ret:
            st.markdown("**Retrieval quality over runs**")
            st.line_chart(ret.iloc[::-1].set_index("when")[["hit@1", "hit@5", "mrr", "ndcg@10"]])
        pick = st.selectbox("Open run", df["id"])
        if st.button("Show details"):
            st.session_state["last_run"] = pick
            st.rerun()
    else:
        st.info("No runs yet.")

with tab_compare:
    emb = {m["id"]: m["name"] for m in models if m["kind"] == "embedding"}
    with st.form("cmp"):
        ds_id = st.selectbox("Dataset ", list(ds_map), format_func=ds_map.get)
        ids = st.multiselect("Embedding models", list(emb), default=list(emb)[:2], format_func=emb.get)
        if st.form_submit_button("Compare") and ids:
            res = client.track_job(client.post("/eval/compare", {"dataset_id": ds_id, "model_ids": ids}), "Comparing")
            if res:
                path_to_name = {m["path"]: m["name"] for m in models}
                df = pd.DataFrame(res).T
                df.index = [path_to_name.get(i, i) for i in df.index]
                st.dataframe(df[[c for c in ["hit@1", "hit@5", "mrr", "ndcg@10", "latency_p95_ms"] if c in df]],
                             width="stretch")
                st.bar_chart(df[["hit@1", "hit@5", "mrr", "ndcg@10"]].T)
