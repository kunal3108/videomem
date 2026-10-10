"""Offline tests (no API key needed). Run: python test_videomem.py"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from evaluate import break_even, cost
from extract import ExtractionError, _parse_batch, extract_observations
from memory import VideoMemory
from retrieve import answer_question, parse_times, retrieve
from video import VideoError, get_metadata, sample_frames

DEMO = Path(__file__).parent / "data" / "demo_observations.json"


def make_video(path, seconds=10, fps=10, size=(1280, 720)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), (i * 3) % 255, np.uint8)
        w.write(frame)
    w.release()


def demo_memory():
    demo = json.loads(DEMO.read_text())
    mem = VideoMemory()
    for o in demo["observations"]:
        mem.add_event(o["timestamp"], o["objects"], o["description"], o["actions"])
    mem.meta = {"sample_interval": 2.0, "duration": 30.0}
    return mem


def test_video_sampling():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v.mp4"
        make_video(p)
        meta = get_metadata(str(p))
        assert abs(meta["duration"] - 10) < 0.5, meta
        frames = sample_frames(str(p), interval=2.0, max_width=640)
        assert [t for t, _ in frames] == [0.0, 2.0, 4.0, 6.0, 8.0], [t for t, _ in frames]
        img = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == 640 and img.shape[0] == 360
        bad = Path(d) / "bad.mp4"
        bad.write_bytes(b"not a video")
        try:
            get_metadata(str(bad))
            assert False, "expected VideoError"
        except VideoError:
            pass


def test_memory_indexes_and_persistence():
    mem = demo_memory()
    assert mem.object_index["bottle"] >= {"event_003"}
    assert [e["timestamp"] for e in mem.get_events_by_object("bottles")][:2] == [4.0, 6.0]
    assert mem.get_events_by_timestamp(4.0)[0]["description"].startswith("A person is holding")
    assert [e["timestamp"] for e in mem.get_events_by_timestamp(10, window=2)] == [8.0, 10.0, 12.0]
    assert mem.search_events("drinking")[0]["timestamp"] == 20.0
    assert mem.get_events_by_object("elephant") == []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        mem.save(p)
        mem2 = VideoMemory.load(p)
        assert mem2.object_index == mem.object_index
        assert mem2.meta == mem.meta


def test_retrieval_and_deterministic_answers():
    mem = demo_memory()
    assert parse_times("What happened around 10 seconds?") == [10.0]
    assert parse_times("at 0:12") == [12]
    ev, plan = retrieve(mem, "When was the bottle visible?")
    assert plan["object_hits"] == ["bottle"] and {e["timestamp"] for e in ev} == {4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0}
    ev, plan = retrieve(mem, "Summarize the video.")
    assert plan["global"] and len(ev) == 15
    ev, plan = retrieve(mem, "Was an elephant visible?")
    assert ev == []
    res = answer_question(mem, "When was the bottle visible?", client=None)
    assert "4.0s" in res["answer"] and res["vision_calls"] == 0
    res = answer_question(mem, "Was an elephant visible?", client=None)
    assert "No matching observations" in res["answer"]


class FakeResponses:
    def __init__(self, payloads):
        self.payloads, self.calls = list(payloads), []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(output_text=self.payloads.pop(0),
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def test_extraction_with_fake_client():
    frames = [(0.0, b"x"), (2.0, b"x"), (4.0, b"x")]
    good = json.dumps({"observations": [
        {"timestamp": 0.1, "objects": ["Person"], "description": "a", "actions": []},
        {"timestamp": 2.0, "objects": ["bottle"], "description": "b", "actions": ["holding bottle"]}]})
    fake = SimpleNamespace(responses=FakeResponses([good, "not json"]))
    obs, stats = extract_observations(frames, batch_size=2, client=fake)
    assert [o["timestamp"] for o in obs] == [0.0, 2.0]  # 0.1 snapped to real timestamp
    assert stats["vision_calls"] == 2 and len(stats["errors"]) == 1  # bad JSON batch reported, not fatal
    img = fake.responses.calls[0]["input"][0]["content"][2]
    assert img["type"] == "input_image" and img["image_url"].startswith("data:image/jpeg;base64,")
    try:
        _parse_batch("{bad", [0.0])
        assert False
    except ExtractionError:
        pass


def test_extraction_retries_merged_frames():
    frames = [(0.0, b"x"), (2.0, b"x"), (4.0, b"x")]
    merged = json.dumps({"observations": [
        {"timestamp": 0.0, "objects": ["bottle"], "description": "a", "actions": []}]})
    single = lambda t, o: json.dumps({"observations": [
        {"timestamp": t, "objects": [o], "description": o, "actions": []}]})
    fake = SimpleNamespace(responses=FakeResponses([merged, single(2.0, "bottle"), single(4.0, "moon")]))
    obs, stats = extract_observations(frames, batch_size=3, client=fake)
    assert [o["timestamp"] for o in obs] == [0.0, 2.0, 4.0]
    assert obs[2]["objects"] == ["moon"]
    assert stats["vision_calls"] == 3 and stats["frames_retried"] == 2 and not stats["errors"]


def test_answer_uses_text_only_call():
    mem = demo_memory()
    fake = SimpleNamespace(responses=FakeResponses(["The bottle was visible at [4.0s]."]))
    res = answer_question(mem, "When was the bottle visible?", client=fake)
    call = fake.responses.calls[0]
    assert isinstance(call["input"], str) and "input_image" not in call["input"]
    assert res["text_calls"] == 1 and res["vision_calls"] == 0 and res["input_tokens"] == 100


def test_break_even():
    rows, n = break_even(index_cost=10, memory_query_cost=1, baseline_query_cost=3)
    assert n == 5 and rows[0]["baseline_total"] == 3
    assert cost(1_000_000, 0, 0.4, 1.6) == 0.4 and cost(1, 1, None, None) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
