# VideoMem — Persistent Visual Memory for Video AI

**Index once. Remember forever.**

## Problem
Video AI apps re-send video frames to a vision model every time a user asks about something that already happened. Each question pays the full image-token cost and the full vision latency again, even though the video hasn't changed.

## Solution
VideoMem processes a video **once**:
1. OpenCV samples one frame every 2 s (resized to ≤640 px wide).
2. OpenAI Vision (Responses API, `gpt-4.1-mini`, strict JSON schema) turns each frame into a timestamped observation: `objects`, `description`, `actions`.
3. Observations go into a hash-map memory engine, persisted to local JSON.

Every later question is answered by **looking things up in memory** and making one small **text-only** call to synthesize a grounded answer with timestamps. No frames are re-sent.

## Architecture

```
MP4 ──► video.py (OpenCV sample @2s, resize 640px, JPEG)
          │
          ▼
        extract.py ── OpenAI Responses API (vision, batched 4 frames/call, strict JSON schema)
          │            validate JSON, snap timestamps to real sampled frames
          ▼
        memory.py  ── events_by_id / object_index / timestamp_index (+ sorted timestamps)
          │            persisted to data/memory_<videohash>_<interval>.json
          ▼
Question ► retrieve.py ── parse objects + times → hash lookups → evidence
                          → OpenAI text-only call (evidence JSON, no images)
                          → grounded answer with [timestamps]
          ▼
        app.py (Streamlit dashboard) + evaluate.py (optional baseline benchmark)
```

| File | Role |
|---|---|
| `video.py` | Metadata, frame sampling, resize, JPEG encode, file hash for caching |
| `extract.py` | OpenAI client, batched vision extraction, JSON validation, friendly API errors |
| `memory.py` | Hash-map memory engine + JSON persistence |
| `retrieve.py` | Question parsing, indexed retrieval, grounded answer (or deterministic fallback) |
| `evaluate.py` | Baseline query, cost from configured pricing, break-even math |
| `app.py` | Streamlit UI |
| `test_videomem.py` | Offline tests (mocked OpenAI client, synthetic video) |

## DSA concepts
```python
events_by_id    = {"event_003": {...}}              # dict: primary store
object_index    = {"bottle": {"event_003", ...}}    # dict[str, set]: inverted index
timestamp_index = {4.0: {"event_003"}}              # dict[float, set]: exact time lookup
_sorted_ts      = [0.0, 2.0, 4.0, ...]              # sorted list for range queries
```
- `get_events_by_object("bottle")` — hash lookup, **average-case O(1)** per key (plus O(k log k) to sort the k matching events). Plural variants (`bottles`→`bottle`) are a constant number of extra lookups.
- `get_events_by_timestamp(4.0)` — exact hash lookup, **average O(1)**.
- `get_events_by_timestamp(10, window=2)` — binary search (`bisect`) on sorted timestamps, **O(log T + k)**.
- `search_events(query)` — combines O(1) object-index hits with a keyword scan of descriptions/actions, which is **O(N)** in the number of events, then ranks candidates (O(m log m)). Exact-key lookups are O(1) on average; scanning or ranking across many events costs more.
- Sets deduplicate event IDs when an object appears multiple times or when several index paths hit the same event.

## System design notes
- **Cache key** = SHA-256 of the video file + sampling interval → re-uploading the same video loads memory with **0 vision calls**.
- **Batching**: 4 frames per vision request (configurable), each labeled with its timestamp; returned timestamps are snapped to real sampled times so the model can't invent times.
- **Grounding**: answer prompt only contains retrieved observations; it must cite `[t s]` and say when information is not in memory. If nothing matches, the model gets an explicit empty-evidence notice plus the list of all indexed objects.
- **Failure modes handled**: missing key, auth error, quota exhausted, rate limits, network errors, invalid JSON (per-batch, non-fatal), unreadable/empty video, no usable frames, empty search results. Without a key, answers fall back to a deterministic summary built straight from the index.
- **Demo Mode**: `data/demo_observations.json` holds hand-written observations of a fictional scene. The UI shows a warning banner whenever it is loaded; it is never presented as AI analysis.

## Install & run
```bash
cd videomem
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env: OPENAI_API_KEY=sk-...
streamlit run app.py
```
Run tests (no API key required): `python test_videomem.py`

Optional `.env` settings: `VIDEOMEM_VISION_MODEL`, `VIDEOMEM_TEXT_MODEL`, `PRICE_INPUT_PER_1M`, `PRICE_OUTPUT_PER_1M`.

## Benchmark methodology
Shown in the **Metrics & Benchmark** tab. All numbers come from measured API `usage` and wall-clock timing in the current session.

- **VideoMem indexing (one-time)**: vision calls, input/output tokens, latency for processing all sampled frames.
- **VideoMem per query**: hash-map retrieval time (ms), text-only LLM latency, text tokens, vision calls (always 0).
- **Baseline per query (optional, costs credits)**: for each of your last N questions, send *all* sampled frames + the question in one vision call; measure tokens and latency.
- **Amortized cost for N questions**:
  - Baseline total = N × baseline query cost
  - VideoMem total = indexing cost + N × memory query cost
  - Break-even N = ⌈indexing cost ÷ (baseline query cost − memory query cost)⌉
- Costs in USD are only shown if you enter pricing (sidebar or `.env`); otherwise the comparison uses raw token counts. Verify prices against OpenAI's pricing page. No accuracy score is reported — we did not build a labeled evaluation set.

## Limitations
- Sampling every 2 s can miss short events; the system only knows what was visible in sampled frames.
- Object names are model-generated strings; synonyms (`cup` vs `mug`) are not merged. Matching is lexical with simple plural handling.
- Time-range answers are only as fine-grained as the sampling interval.
- Single-video memory per session; no cross-video search yet.
- Vision accuracy depends on the model and frame quality; observations are not human-verified.

## Roadmap
- Synonym/embedding-based object normalization (still without a vector DB server)
- Scene-change–aware sampling instead of fixed intervals
- Multi-video and live-stream (CCTV) memory with rolling retention
- Object tracks (first seen / last seen intervals) and counting
- Re-query the original frames only for questions memory can't answer ("memory miss → vision fallback")
- Memory API for robotics and video agents

## 60–90 second demo script
> **(0–10 s)** "Video AI apps pay to re-watch the same video every time you ask a question. VideoMem indexes a video once and remembers it."
>
> **(10–30 s)** Upload a 30-second clip (someone picking up a bottle at a desk). Point at duration / FPS / frames. Click **Process Video**. "We sample a frame every 2 seconds and send them to GPT-4.1-mini Vision in batches, with a strict JSON schema." Show the progress bar, then the success line: N observations, X vision calls.
>
> **(30–45 s)** **Timeline** tab: timestamped observations. **Object Index** tab: "This is a hash map from object to a set of event IDs — average O(1) lookup. Here's `bottle` → 4s, 6s, 8s."
>
> **(45–70 s)** **Ask**: "When was the bottle visible?" → grounded answer with timestamps. Point at metrics: retrieval in well under a millisecond, **0 vision calls**, a few hundred text tokens. Ask "Was a laptop visible?" → it says it's not in memory instead of guessing. Ask "What happened around 10 seconds?" — no reprocessing.
>
> **(70–90 s)** **Metrics & Benchmark**: "Indexing is a one-time cost; every question after that is text-only. Here's the measured baseline that re-sends all frames per question, and the break-even point." Close: "Index once, query forever — persistent visual memory for CCTV, robots, and video agents."
