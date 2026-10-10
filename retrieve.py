"""Question -> indexed retrieval -> grounded text-only answer. No frames are resent."""
import json
import re
import time

from extract import DEFAULT_TEXT_MODEL, friendly_error, usage_of
from memory import tokenize, variants

GLOBAL_HINTS = ("summar", "overview", "everything", "all objects", "what objects",
                "which objects", "list", "describe the video", "what happens in", "whole video")

ANSWER_PROMPT = (
    "You answer questions about a video using ONLY the indexed observations provided. "
    "Each observation came from a single sampled frame at the given timestamp.\n"
    "- Cite supporting timestamps like [4.0s] for every claim.\n"
    "- If the evidence does not contain the answer, say clearly that it is not in the indexed memory.\n"
    "- Never invent objects, actions, or events. Do not assume what happened between sampled frames.\n"
    "- Be concise (2-5 sentences or a short list)."
)


def parse_times(question):
    """Find time references like '10 seconds', '10s', '0:10', 'at 12'."""
    q = question.lower()
    times = []
    for m, s in re.findall(r"\b(\d{1,2}):(\d{2})\b", q):
        times.append(int(m) * 60 + int(s))
    q = re.sub(r"\b\d{1,2}:\d{2}\b", " ", q)
    for n in re.findall(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)", q):
        times.append(float(n))
    if not times:
        for n in re.findall(r"\b(?:at|around|near|about)\s+(\d+(?:\.\d+)?)\b", q):
            times.append(float(n))
    return times


def retrieve(memory, question, window=None):
    """Return (events, plan) where plan explains which index paths were used."""
    interval = memory.meta.get("sample_interval", 2.0)
    window = window if window is not None else interval
    plan = {"object_hits": [], "time_hits": [], "keyword_search": False, "global": False}
    found = {}

    # 1. Object index: O(1) average lookup per question token (+ plural variants).
    for tok in set(tokenize(question)):
        evs = memory.get_events_by_object(tok)
        if evs:
            plan["object_hits"].append(tok)
            for e in evs:
                found[e["id"]] = e
    # Multi-word objects ("water bottle") — check index keys that appear in the question.
    q = question.lower()
    for obj in memory.object_index:
        if " " in obj and obj in q:
            plan["object_hits"].append(obj)
            for e in memory.get_events_by_object(obj):
                found[e["id"]] = e

    # 2. Timestamp index: exact hash hit or binary-searched window.
    for t in parse_times(question):
        evs = memory.get_events_by_timestamp(t, window=window)
        plan["time_hits"].append(t)
        for e in evs:
            found[e["id"]] = e

    # 3. Global questions get the whole (compact) timeline.
    if any(h in q for h in GLOBAL_HINTS) or (not found and not plan["time_hits"] and not tokenize(question)):
        plan["global"] = True
        for e in memory.all_events():
            found[e["id"]] = e

    # 4. Fallback: keyword search over descriptions/actions.
    if not found and not plan["time_hits"]:
        plan["keyword_search"] = True
        for e in memory.search_events(question):
            found[e["id"]] = e

    events = sorted(found.values(), key=lambda e: e["timestamp"])
    return events, plan


def _compact(events):
    return [{"t": e["timestamp"], "objects": e["objects"], "description": e["description"],
             "actions": e["actions"]} for e in events]


def fallback_answer(memory, question, events, plan):
    """Deterministic answer used when no API key is available (or the API fails)."""
    if not events:
        asked = [t for t in tokenize(question)]
        return ("No matching observations in the indexed memory"
                + (f" for: {', '.join(asked)}" if asked else "")
                + f". Indexed objects: {', '.join(memory.objects()) or 'none'}.")
    if plan["object_hits"] and not plan["global"]:
        lines = []
        for obj in plan["object_hits"]:
            ts = [f"{e['timestamp']}s" for e in memory.get_events_by_object(obj)]
            lines.append(f"'{obj}' visible at: {', '.join(ts)}")
        return "\n".join(lines)
    return "\n".join(f"[{e['timestamp']}s] {e['description']}" for e in events[:15])


def answer_question(memory, question, client=None, model=DEFAULT_TEXT_MODEL):
    t0 = time.perf_counter()
    events, plan = retrieve(memory, question)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    result = {"question": question, "evidence": events, "plan": plan,
              "retrieval_ms": round(retrieval_ms, 3), "vision_calls": 0, "text_calls": 0,
              "input_tokens": 0, "output_tokens": 0, "llm_latency_s": 0.0, "mode": "llm"}

    if client is None:
        result.update(answer=fallback_answer(memory, question, events, plan), mode="deterministic")
        result["total_latency_s"] = round(time.perf_counter() - t0, 3)
        return result

    if events:
        evidence = json.dumps(_compact(events))
    else:
        evidence = ("[] (no observation matched the question). Objects indexed anywhere in the video: "
                    + json.dumps(memory.objects()))
    duration = memory.meta.get("duration")
    prompt = (f"Video duration: {duration}s, sampled every {memory.meta.get('sample_interval', 2.0)}s.\n"
              f"Retrieved observations:\n{evidence}\n\nQuestion: {question}")
    t1 = time.perf_counter()
    try:
        resp = client.responses.create(model=model, instructions=ANSWER_PROMPT,
                                       input=prompt, temperature=0)
        result["answer"] = resp.output_text.strip()
        result["text_calls"] = 1
        result.update(usage_of(resp))
    except Exception as exc:
        result["answer"] = fallback_answer(memory, question, events, plan)
        result["mode"] = "deterministic"
        result["error"] = friendly_error(exc)
    result["llm_latency_s"] = round(time.perf_counter() - t1, 3)
    result["total_latency_s"] = round(time.perf_counter() - t0, 3)
    return result
