import json

import pandas as pd
import streamlit as st

import client

st.title("🧪 Datasets & Fine-tuning")
st.caption("Adapt the retrieval and generation models to your organisation's vocabulary using question/passage pairs "
           "generated from your own documents.")

collections = [c["name"] for c in client.get("/collections")]
tab_data, tab_train, tab_models = st.tabs(["1 · Datasets", "2 · Train", "3 · Model registry"])

with tab_data:
    c1, c2 = st.columns(2)
    with c1, st.form("gen"):
        st.markdown("**Generate a synthetic dataset**")
        col = st.selectbox("Collection", collections)
        method = st.selectbox("Method", ["auto", "llm", "heuristic"],
                              help="llm: questions written by the local LLM (best). heuristic: offline rules. auto: llm with fallback.")
        max_chunks = st.number_input("Chunks to sample", 5, 5000, 150)
        qpc = st.slider("Questions per chunk", 1, 5, 2)
        test_ratio = st.slider("Held-out test ratio", 0.05, 0.9, 0.25)
        name = st.text_input("Name (optional)")
        if st.form_submit_button("Generate", type="primary"):
            job = client.post("/datasets/generate", {"collection": col, "method": method, "max_chunks": int(max_chunks),
                                                     "questions_per_chunk": qpc, "test_ratio": test_ratio, "name": name or None})
            if res := client.track_job(job, "Generating dataset"):
                st.success(f"Dataset {res['id']} with {res['num_items']} items ({res['method']})")
    with c2, st.form("manual"):
        st.markdown("**Import a hand-labelled evaluation set**")
        col_m = st.selectbox("Collection ", collections)
        up = st.file_uploader("JSON list of {question, answer?, category?, source?, page?, reference?}", type=["json"])
        if st.form_submit_button("Import") and up:
            try:
                res = client.post("/datasets/manual", {"collection": col_m, "items": json.loads(up.getvalue())})
                st.success(f"Imported {res['num_items']} items as {res['id']}")
            except (client.APIError, ValueError) as exc:
                st.error(str(exc))

    datasets = client.get("/datasets")
    if datasets:
        df = pd.DataFrame(datasets)
        df["created"] = df["created_at"].map(client.fmt_time)
        st.dataframe(df[["id", "name", "collection", "method", "num_items", "splits", "created"]], hide_index=True,
                     width="stretch")
        ds_id = st.selectbox("Inspect dataset", [d["id"] for d in datasets])
        d = client.get(f"/datasets/{ds_id}", limit=200)
        st.caption(f"categories: {d.get('categories')}")
        items = pd.DataFrame(d["items"])
        cols = [c for c in ["question", "answer", "category", "split", "origin", "source", "page"] if c in items]
        st.dataframe(items[cols], hide_index=True, width="stretch", height=300)
        if st.button("Delete dataset"):
            client.delete(f"/datasets/{ds_id}")
            st.rerun()

with tab_train:
    datasets = client.get("/datasets")
    if not datasets:
        st.info("Generate a dataset first.")
    else:
        ds_map = {d["id"]: f"{d['name']} ({d['num_items']} items)" for d in datasets}
        target = st.radio("What to adapt", ["Embedding model", "Reranker", "LLM (LoRA)"], horizontal=True)
        with st.form("train"):
            ds_id = st.selectbox("Dataset", list(ds_map), format_func=ds_map.get)
            c1, c2, c3 = st.columns(3)
            if target == "LLM (LoRA)":
                st.caption("Trains a LoRA adapter on a small HF model (default Qwen2.5-0.5B-Instruct) to answer with citations "
                           "and refuse when context is insufficient. Needs `requirements-finetune.txt` and ideally a GPU.")
                epochs = c1.number_input("Epochs", 1, 10, 2)
                lr = c2.number_input("Learning rate", 1e-6, 1e-2, 2e-4, format="%.6f")
                lora_r = c3.number_input("LoRA rank", 2, 128, 16)
                base = st.text_input("Base HF model (blank = default)")
            else:
                st.caption("Contrastive (MNRL + hard negatives)" if target == "Embedding model" else
                           "Binary relevance with mined hard negatives")
                epochs = c1.number_input("Epochs", 1, 20, 2)
                lr = c2.number_input("Learning rate", 1e-7, 1e-2, 2e-5, format="%.6f")
                bs = c3.number_input("Batch size", 2, 256, 16)
                base = st.text_input("Base model (blank = currently active)")
            name = st.text_input("Model name (optional)")
            if st.form_submit_button("Start training", type="primary"):
                body = {"dataset_id": ds_id, "epochs": int(epochs), "learning_rate": float(lr), "name": name or None,
                        "base_model": base or None}
                if target == "LLM (LoRA)":
                    job = client.post("/finetune/llm", {**body, "lora_r": int(lora_r)})
                else:
                    job = client.post("/finetune/embedding" if target == "Embedding model" else "/finetune/reranker",
                                      {**body, "batch_size": int(bs)})
                res = client.track_job(job, f"Training {target}")
                if res:
                    st.success(f"Registered model {res['model']['name']} ({res['model']['id']})")
                    m = res["metrics"]
                    if "improvement" in m:
                        rows = [{"metric": k, "baseline": m["baseline"].get(k), "fine-tuned": m["eval"].get(k), "Δ": v}
                                for k, v in m["improvement"].items()]
                        st.dataframe(pd.DataFrame(rows), hide_index=True)
                    st.json(m.get("train", {}))
                    if m.get("recommended") is False:
                        st.warning("The fine-tuned model did not beat the current one on held-out data. Keep the current "
                                   "model, or retry with a lower learning rate / more data.")
                    else:
                        st.info("Activate it in the Model registry tab to use it in production.")

with tab_models:
    models = client.get("/models")
    for kind in ("embedding", "reranker", "llm"):
        st.subheader(kind.capitalize())
        for m in [x for x in models if x["kind"] == kind]:
            c1, c2, c3 = st.columns([4, 4, 1])
            c1.markdown(f"{'🟢 ' if m['is_active'] else ''}**{m['name']}**  \n`{m['id']}` · {client.fmt_time(m['created_at'])}")
            imp = (m.get("metrics") or {}).get("improvement")
            if imp:
                rec = m["metrics"].get("recommended")
                badge = "✅ recommended · " if rec else ("⚠️ not recommended · " if rec is False else "")
                c2.caption(badge + " · ".join(f"{k} {v:+.3f}" for k, v in imp.items()))
            elif (m.get("metrics") or {}).get("train"):
                c2.caption(f"loss {m['metrics']['train'].get('final_loss', 0):.4f}")
            if not m["is_active"] and c3.button("Activate", key=m["id"]):
                res = client.post(f"/models/{m['id']}/activate")
                if res.get("job"):
                    client.track_job(res["job"], "Re-embedding collections")
                st.rerun()
