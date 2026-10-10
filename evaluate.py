"""Benchmark: per-question vision baseline vs. VideoMem (index once, query memory).

All numbers come from measured API usage. Costs are only computed from pricing
that is explicitly configured (USD per 1M tokens) — verify against OpenAI's
current pricing page before quoting them.
"""
import os
import time

from extract import DEFAULT_VISION_MODEL, _image_part, usage_of


def configured_pricing():
    """Return (input_per_1m, output_per_1m) from .env, or (None, None) if not configured."""
    try:
        return float(os.environ["PRICE_INPUT_PER_1M"]), float(os.environ["PRICE_OUTPUT_PER_1M"])
    except (KeyError, ValueError):
        return None, None


def cost(input_tokens, output_tokens, price_in, price_out):
    if price_in is None or price_out is None:
        return None
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


def run_baseline_query(client, frames, question, model=DEFAULT_VISION_MODEL, detail="auto"):
    """Baseline: answer one question by sending ALL sampled frames to the vision model."""
    content = [{"type": "input_text",
                "text": f"These are frames sampled from a video. Answer with timestamps.\nQuestion: {question}"}]
    for t, jpg in frames:
        content.append({"type": "input_text", "text": f"t={t}s"})
        content.append(_image_part(jpg, detail))
    t0 = time.perf_counter()
    resp = client.responses.create(model=model, input=[{"role": "user", "content": content}], temperature=0)
    out = {"question": question, "answer": resp.output_text.strip(), "vision_calls": 1,
           "frames_sent": len(frames), "latency_s": round(time.perf_counter() - t0, 3)}
    out.update(usage_of(resp))
    return out


def break_even(index_cost, memory_query_cost, baseline_query_cost, ns=(1, 5, 10, 25, 50, 100)):
    """Rows of cumulative cost for N questions. Returns (rows, break_even_n or None)."""
    rows = [{"questions": n,
             "baseline_total": n * baseline_query_cost,
             "videomem_total": index_cost + n * memory_query_cost} for n in ns]
    saving_per_q = baseline_query_cost - memory_query_cost
    n_star = None
    if saving_per_q > 0:
        n_star = max(1, int(-(-index_cost // saving_per_q)))  # ceil
    return rows, n_star
