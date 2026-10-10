"""VideoMem — Index once. Remember forever.  Run: streamlit run app.py"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluate import break_even, configured_pricing, cost, run_baseline_query
from extract import (DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL, ExtractionError,
                     extract_observations, friendly_error, get_client)
from memory import VideoMemory
from retrieve import answer_question
from video import VideoError, file_hash, get_metadata, sample_frames

load_dotenv()
DATA = Path(__file__).parent / "data"
UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
DEMO_FILE = DATA / "demo_observations.json"

st.set_page_config(page_title="VideoMem", page_icon="🎞️", layout="wide")
ss = st.session_state
ss.setdefault("memory", None)
ss.setdefault("frames", None)
ss.setdefault("queries", [])
ss.setdefault("baseline", [])
ss.setdefault("last_answer", None)


def has_key():
    load_dotenv(override=True)
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def build_memory(observations, meta):
    mem = VideoMemory()
    for o in observations:
        mem.add_event(o["timestamp"], o["objects"], o["description"], o.get("actions"))
    mem.meta = meta
    return mem


def load_demo():
    demo = json.loads(DEMO_FILE.read_text())
    meta = {"source": "demo", "video_name": "DEMO (synthetic desk scene)", "duration": demo["duration"],
            "sample_interval": demo["sample_interval"], "frames_sampled": len(demo["observations"]),
            "indexing": {"vision_calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0}}
    ss.memory = build_memory(demo["observations"], meta)
    ss.frames, ss.queries, ss.baseline, ss.last_answer = None, [], [], None


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    if has_key():
        st.success("OpenAI API key detected")
    else:
        st.error("OPENAI_API_KEY missing — add it to `.env`. Demo Mode still works.")
    vision_model = st.text_input("Vision model", DEFAULT_VISION_MODEL)
    text_model = st.text_input("Answer model (text-only)", DEFAULT_TEXT_MODEL)
    interval = st.slider("Sample every N seconds", 1.0, 5.0, 2.0, 0.5)
    batch_size = st.slider("Frames per vision request", 1, 8, 4)
    detail = st.selectbox("Image detail", ["auto", "low", "high"], index=0)
    st.divider()
    st.subheader("Pricing (USD / 1M tokens)")
    env_in, env_out = configured_pricing()
    price_in = st.number_input("Input", value=env_in or 0.0, min_value=0.0, step=0.05, format="%.3f")
    price_out = st.number_input("Output", value=env_out or 0.0, min_value=0.0, step=0.05, format="%.3f")
    pricing_set = price_in > 0 or price_out > 0
    st.caption("Used only for cost estimates. Verify against OpenAI's current pricing. "
               "Leave at 0 to compare raw tokens.")
    st.divider()
    if st.button("Load Demo Mode data", width="stretch"):
        load_demo()
    st.caption("Demo Mode uses hand-written sample observations, not AI analysis.")

# ---------------- Header ----------------
st.title("🎞️ VideoMem")
st.markdown("#### Index once. Remember forever.")
st.caption("Persistent visual memory for video AI: OpenAI Vision indexes each video once into a "
           "hash-map memory; every later question is answered from memory — no frames resent.")

# ---------------- Upload & process ----------------
upload = st.file_uploader("Upload an MP4 (30–60 s works best)", type=["mp4", "mov", "m4v"])
if upload is not None:
    raw = upload.getvalue()
    if not raw:
        st.error("The uploaded file is empty.")
        st.stop()
    tmp = UPLOADS / "_incoming.mp4"
    tmp.write_bytes(raw)
    vid_hash = file_hash(tmp)
    video_path = UPLOADS / f"{vid_hash}.mp4"
    tmp.replace(video_path)
    try:
        meta = get_metadata(str(video_path))
    except VideoError as exc:
        st.error(str(exc))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{meta['duration']} s")
    c2.metric("FPS", meta["fps"])
    c3.metric("Resolution", f"{meta['width']}×{meta['height']}")
    c4.metric("Frames to sample", int(meta["duration"] // interval) + 1)

    cache_path = DATA / f"memory_{vid_hash}_{interval}.json"
    col_a, col_b = st.columns([1, 3])
    process = col_a.button("Process Video", type="primary", width="stretch")
    if cache_path.exists():
        col_b.info("This video is already indexed. Processing will load the saved memory (0 vision calls). "
                   "Tick below to force re-indexing.")
        force = col_b.checkbox("Force re-index with OpenAI")
    else:
        force = False

    if process:
        ss.queries, ss.baseline, ss.last_answer = [], [], None
        try:
            ss.frames = sample_frames(str(video_path), interval=interval, max_width=640)
        except VideoError as exc:
            st.error(str(exc))
            st.stop()
        if cache_path.exists() and not force:
            ss.memory = VideoMemory.load(cache_path)
            ss.memory.meta["loaded_from_cache"] = True
            st.success(f"Loaded persisted memory from `{cache_path.name}` — no vision calls made.")
        else:
            bar = st.progress(0.0, text=f"Sampled {len(ss.frames)} frames. Sending to OpenAI Vision…")
            try:
                obs, stats = extract_observations(
                    ss.frames, model=vision_model, batch_size=batch_size, detail=detail,
                    progress=lambda p, msg: bar.progress(p, text=msg))
            except ExtractionError as exc:
                bar.empty()
                st.error(f"Vision extraction failed: {exc}")
                st.info("You can still explore the pipeline with **Load Demo Mode data** in the sidebar "
                        "(clearly labeled synthetic data).")
                st.stop()
            bar.progress(1.0, text="Building hash-map memory…")
            mem_meta = {"source": "openai", "video_name": upload.name, "video_hash": vid_hash,
                        **meta, "sample_interval": interval, "frames_sampled": len(ss.frames),
                        "indexing": stats, "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            ss.memory = build_memory(obs, mem_meta)
            ss.memory.save(cache_path)
            bar.empty()
            if stats["errors"]:
                st.warning("Some batches failed: " + "; ".join(stats["errors"]))
            st.success(f"Indexed {len(obs)} observations with {stats['vision_calls']} vision calls in "
                       f"{stats['latency_s']} s. Memory saved to `{cache_path.name}`.")

mem: VideoMemory = ss.memory
if mem is None:
    st.info("Upload a video and click **Process Video** — or load Demo Mode data from the sidebar.")
    st.stop()

if mem.meta.get("source") == "demo":
    st.warning("⚠️ DEMO MODE — these are hand-written sample observations of a fictional scene, "
               "NOT AI analysis of any uploaded video.")

idx = mem.meta.get("indexing", {})
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Indexed events", len(mem.events_by_id))
s2.metric("Unique objects", len(mem.object_index))
s3.metric("Indexing vision calls", idx.get("vision_calls", 0))
s4.metric("Indexing tokens (in/out)", f"{idx.get('input_tokens', 0):,} / {idx.get('output_tokens', 0):,}")
s5.metric("Indexing latency", f"{idx.get('latency_s', 0)} s")

tab_ask, tab_tl, tab_idx, tab_eval, tab_json = st.tabs(
    ["💬 Ask", "🕒 Timeline", "#️⃣ Object Index", "📊 Metrics & Benchmark", "🧾 JSON"])

# ---------------- Ask ----------------
with tab_ask:
    examples = ["When was the bottle visible?", "What objects appeared in the video?",
                "What happened around 10 seconds?", "Was a laptop visible?", "Summarize the video."]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        col.button(ex, width="stretch", on_click=lambda q=ex: ss.update(question=q))
    with st.form("ask", clear_on_submit=False):
        question = st.text_input("Ask about the video", key="question")
        submitted = st.form_submit_button("Ask memory", type="primary")
    if submitted and question.strip():
        client = None
        if has_key():
            try:
                client = get_client()
            except ExtractionError:
                client = None
        with st.spinner("Retrieving from memory…"):
            res = answer_question(mem, question.strip(), client=client, model=text_model)
        ss.queries.append(res)
        ss.last_answer = res

    res = ss.last_answer
    if res:
        st.markdown(f"**Q:** {res['question']}")
        st.success(res["answer"])
        if res.get("error"):
            st.warning(f"LLM answer failed ({res['error']}); showing deterministic answer from memory.")
        elif res["mode"] == "deterministic":
            st.caption("No API key — deterministic answer built directly from the memory index.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hash-map retrieval", f"{res['retrieval_ms']:.2f} ms")
        m2.metric("Answer latency", f"{res['total_latency_s']} s")
        m3.metric("Vision calls this query", res["vision_calls"])
        m4.metric("Text tokens (in/out)", f"{res['input_tokens']} / {res['output_tokens']}")
        plan = res["plan"]
        st.caption(f"Index paths used → objects: {plan['object_hits'] or '—'} · times: {plan['time_hits'] or '—'} · "
                   f"global: {plan['global']} · keyword search: {plan['keyword_search']}")
        with st.expander(f"Evidence sent to the model ({len(res['evidence'])} observations)"):
            if res["evidence"]:
                st.dataframe(pd.DataFrame(res["evidence"])[["timestamp", "objects", "actions", "description", "id"]],
                             hide_index=True, width="stretch")
            else:
                st.write("No matching observations in memory.")

# ---------------- Timeline ----------------
with tab_tl:
    events = mem.all_events()
    st.dataframe(pd.DataFrame(events)[["timestamp", "objects", "actions", "description", "id"]],
                 hide_index=True, width="stretch")
    if ss.frames:
        with st.expander("Sampled frames"):
            frame_map = dict(ss.frames)
            cols = st.columns(5)
            for i, ev in enumerate(events):
                if ev["timestamp"] in frame_map:
                    cols[i % 5].image(frame_map[ev["timestamp"]], caption=f"{ev['timestamp']}s",
                                      width="stretch")

# ---------------- Object index ----------------
with tab_idx:
    st.markdown("`object_index: dict[str, set[event_id]]` — inverted index, average O(1) lookup per object.")
    rows = [{"object": o, "count": len(ids),
             "timestamps (s)": ", ".join(str(e["timestamp"]) for e in mem.get_events_by_object(o)),
             "event_ids": ", ".join(sorted(ids))} for o, ids in mem.object_index.items()]
    rows.sort(key=lambda r: -r["count"])
    left, right = st.columns([3, 2])
    left.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if rows:
        right.bar_chart(pd.DataFrame(rows).set_index("object")["count"])
    pick = st.selectbox("Look up an object", mem.objects())
    if pick:
        t0 = time.perf_counter()
        hits = mem.get_events_by_object(pick)
        st.caption(f"`get_events_by_object('{pick}')` → {len(hits)} events in "
                   f"{(time.perf_counter() - t0) * 1000:.3f} ms")
        st.dataframe(pd.DataFrame(hits)[["timestamp", "description"]], hide_index=True, width="stretch")

# ---------------- Metrics & benchmark ----------------
with tab_eval:
    st.subheader("Measured VideoMem metrics")
    idx_in, idx_out = idx.get("input_tokens", 0), idx.get("output_tokens", 0)
    q = ss.queries
    llm_q = [r for r in q if r["text_calls"]]
    st.markdown(f"**Indexing (one-time):** {idx.get('vision_calls', 0)} vision calls · "
                f"{idx_in:,} input / {idx_out:,} output tokens · {idx.get('latency_s', 0)} s"
                + (" · *(loaded from cache this session — 0 new calls)*" if mem.meta.get("loaded_from_cache") else ""))
    if q:
        qdf = pd.DataFrame([{"question": r["question"], "retrieval_ms": r["retrieval_ms"],
                             "total_latency_s": r["total_latency_s"], "vision_calls": r["vision_calls"],
                             "text_calls": r["text_calls"], "input_tokens": r["input_tokens"],
                             "output_tokens": r["output_tokens"], "evidence": len(r["evidence"])} for r in q])
        st.markdown(f"**Per-query (memory):** {len(q)} questions · vision calls during queries: "
                    f"**{int(qdf.vision_calls.sum())}** · avg retrieval {qdf.retrieval_ms.mean():.3f} ms · "
                    f"avg end-to-end {qdf.total_latency_s.mean():.2f} s")
        st.dataframe(qdf, hide_index=True, width="stretch")
    else:
        st.caption("Ask a few questions to collect per-query metrics.")

    st.divider()
    st.subheader("Optional baseline: re-send all frames for every question")
    st.caption("Baseline = one vision call per question containing every sampled frame. "
               "This spends real API credits, so it only runs when you click.")
    can_baseline = bool(ss.frames) and has_key() and bool(q)
    if not ss.frames:
        st.info("Baseline needs the original frames — process an uploaded video in this session first.")
    n_base = st.number_input("Baseline questions to run (from your asked questions)", 1, 5, min(2, max(1, len(q))))
    if st.button("Run baseline (costs credits)", disabled=not can_baseline):
        client = get_client()
        for r in q[-int(n_base):]:
            with st.spinner(f"Baseline: {r['question']}"):
                try:
                    ss.baseline.append(run_baseline_query(client, ss.frames, r["question"],
                                                          model=vision_model, detail=detail))
                except Exception as exc:
                    st.error(friendly_error(exc))
                    break
    if ss.baseline:
        st.dataframe(pd.DataFrame(ss.baseline), hide_index=True, width="stretch")

    if ss.baseline and llm_q:
        st.subheader("Amortized cost / break-even (from measured tokens)")
        b_in = sum(b["input_tokens"] for b in ss.baseline) / len(ss.baseline)
        b_out = sum(b["output_tokens"] for b in ss.baseline) / len(ss.baseline)
        m_in = sum(r["input_tokens"] for r in llm_q) / len(llm_q)
        m_out = sum(r["output_tokens"] for r in llm_q) / len(llm_q)
        b_lat = sum(b["latency_s"] for b in ss.baseline) / len(ss.baseline)
        m_lat = sum(r["total_latency_s"] for r in llm_q) / len(llm_q)
        if pricing_set:
            unit = "USD"
            index_c = cost(idx_in, idx_out, price_in, price_out)
            base_c = cost(b_in, b_out, price_in, price_out)
            mem_c = cost(m_in, m_out, price_in, price_out)
        else:
            unit = "tokens (in+out)"
            index_c, base_c, mem_c = idx_in + idx_out, b_in + b_out, m_in + m_out
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Indexing cost ({unit})", f"{index_c:,.6g}")
        k2.metric(f"Baseline / question", f"{base_c:,.6g}")
        k3.metric(f"VideoMem / question", f"{mem_c:,.6g}")
        k4.metric("Avg latency baseline → memory", f"{b_lat:.2f}s → {m_lat:.2f}s")
        rows, n_star = break_even(index_c, mem_c, base_c)
        st.line_chart(pd.DataFrame(rows).set_index("questions"))
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if n_star:
            st.success(f"Break-even after ≈ **{n_star} questions** (index cost ÷ per-question saving). "
                       f"Based on {len(ss.baseline)} measured baseline and {len(llm_q)} measured memory queries.")
        else:
            st.warning("With these measurements the memory query is not cheaper per question.")
        if not pricing_set:
            st.caption("Pricing not configured: comparison uses raw token counts (input and output weighted equally).")

# ---------------- JSON ----------------
with tab_json:
    payload = json.dumps(mem.to_dict(), indent=2)
    st.download_button("Download memory JSON", payload, file_name="videomem_memory.json",
                       mime="application/json")
    st.json(mem.to_dict(), expanded=False)
