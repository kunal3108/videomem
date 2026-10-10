"""OpenAI Vision extraction: sampled frames -> structured, timestamped observations."""
import base64
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

DEFAULT_VISION_MODEL = os.getenv("VIDEOMEM_VISION_MODEL", "gpt-4.1-mini")
DEFAULT_TEXT_MODEL = os.getenv("VIDEOMEM_TEXT_MODEL", "gpt-4.1-mini")


class ExtractionError(Exception):
    pass


def get_client():
    """Return an OpenAI client, or raise ExtractionError with a friendly message."""
    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ExtractionError("OPENAI_API_KEY is missing. Add it to .env and restart.")
    from openai import OpenAI
    return OpenAI(max_retries=2, timeout=90)


def friendly_error(exc):
    import openai
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI authentication failed — check OPENAI_API_KEY."
    if isinstance(exc, openai.RateLimitError):
        if "quota" in str(exc).lower():
            return "OpenAI quota exhausted (insufficient_quota). Add credits or use Demo Mode."
        return "OpenAI rate limit hit. Wait a moment and retry."
    if isinstance(exc, openai.NotFoundError):
        return "Model not found or not available for this key."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach the OpenAI API (network error)."
    if isinstance(exc, openai.APIError):
        return f"OpenAI API error: {getattr(exc, 'message', str(exc))[:200]}"
    return f"{type(exc).__name__}: {str(exc)[:200]}"


OBS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observations"],
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["timestamp", "objects", "description", "actions"],
                "properties": {
                    "timestamp": {"type": "number"},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "actions": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

SYSTEM_PROMPT = (
    "You are a precise visual indexer for a video memory system. For EACH frame you receive, "
    "return one observation with the exact timestamp given for that frame. Rules:\n"
    "- objects: concrete visible things as short lowercase singular nouns (e.g. 'person', 'bottle', 'laptop').\n"
    "- description: one factual sentence about what is visible in that single frame.\n"
    "- actions: only actions clearly visible in that frame (e.g. 'holding bottle'). Empty list if none.\n"
    "- Describe only what is visually supported. Never infer what happened between frames. Never guess."
)


def _image_part(jpeg_bytes, detail):
    b64 = base64.b64encode(jpeg_bytes).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": detail}


def usage_of(resp):
    u = getattr(resp, "usage", None)
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
    }


def _parse_batch(text, timestamps):
    """Validate model JSON and pin every observation to a real sampled timestamp."""
    try:
        data = json.loads(text)
        items = data["observations"]
        assert isinstance(items, list)
    except Exception as exc:
        raise ExtractionError(f"Model returned invalid JSON: {exc}") from exc

    by_ts = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        ts = item.get("timestamp")
        # Snap to nearest real timestamp; fall back to positional order.
        if isinstance(ts, (int, float)) and timestamps:
            ts = min(timestamps, key=lambda t: abs(t - ts))
        elif i < len(timestamps):
            ts = timestamps[i]
        else:
            continue
        if ts in by_ts:
            continue
        by_ts[ts] = {
            "timestamp": ts,
            "objects": [str(o) for o in item.get("objects", []) if str(o).strip()],
            "description": str(item.get("description", "")).strip(),
            "actions": [str(a) for a in item.get("actions", []) if str(a).strip()],
        }
    return [by_ts[t] for t in timestamps if t in by_ts]


def extract_observations(frames, model=DEFAULT_VISION_MODEL, batch_size=4, detail="auto",
                         client=None, progress=None):
    """frames: [(timestamp, jpeg_bytes)]. Returns (observations, stats)."""
    client = client or get_client()
    observations, errors = [], []
    stats = {"vision_calls": 0, "input_tokens": 0, "output_tokens": 0, "model": model,
             "frames_sent": 0, "batch_size": batch_size, "detail": detail}
    start = time.perf_counter()
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    for bi, batch in enumerate(batches):
        timestamps = [t for t, _ in batch]
        content = [{"type": "input_text",
                    "text": f"{len(batch)} frames follow. Timestamps (seconds): {timestamps}."}]
        for t, jpg in batch:
            content.append({"type": "input_text", "text": f"Frame at t={t}s:"})
            content.append(_image_part(jpg, detail))
        try:
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": content}],
                text={"format": {"type": "json_schema", "name": "frame_observations",
                                 "schema": OBS_SCHEMA, "strict": True}},
                temperature=0,
            )
            stats["vision_calls"] += 1
            stats["frames_sent"] += len(batch)
            u = usage_of(resp)
            stats["input_tokens"] += u["input_tokens"]
            stats["output_tokens"] += u["output_tokens"]
            observations.extend(_parse_batch(resp.output_text, timestamps))
        except ExtractionError as exc:
            errors.append(f"batch {bi + 1}: {exc}")
        except Exception as exc:
            msg = friendly_error(exc)
            # Auth/quota problems will fail every batch — stop early.
            if "authentication" in msg.lower() or "quota" in msg.lower() or "missing" in msg.lower():
                raise ExtractionError(msg) from exc
            errors.append(f"batch {bi + 1}: {msg}")
        if progress:
            progress((bi + 1) / len(batches), f"Analyzed batch {bi + 1}/{len(batches)}")
    stats["latency_s"] = round(time.perf_counter() - start, 2)
    stats["errors"] = errors
    if not observations:
        raise ExtractionError("No observations extracted. " + ("; ".join(errors) or ""))
    return observations, stats
