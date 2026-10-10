"""Hash-map visual memory engine.

Three dictionaries form the index:
    events_by_id     : event_id   -> event dict          (primary store)
    object_index     : object     -> set(event_id)       (inverted index)
    timestamp_index  : timestamp  -> set(event_id)       (exact-time lookup)

Exact-key lookups are average-case O(1). Range queries over time use a sorted
list of timestamps + binary search (O(log T + k)).
"""
import bisect
import json
import re

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "of", "in", "on", "at", "to",
    "and", "or", "what", "when", "where", "who", "did", "does", "do", "any",
    "video", "visible", "appear", "appeared", "there", "seen", "see", "it",
    "with", "around", "seconds", "second", "sec", "s", "time", "happened",
}


def normalize(word):
    return word.strip().lower()


def variants(word):
    """Cheap plural/singular variants so 'bottles' finds 'bottle' and vice versa."""
    w = normalize(word)
    out = {w}
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    elif w.endswith("es") and len(w) > 4:
        out.update({w[:-2], w[:-1]})
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        out.add(w[:-1])
    else:
        out.update({w + "s", w + "es"})
    return out


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class VideoMemory:
    def __init__(self):
        self.events_by_id = {}
        self.object_index = {}
        self.timestamp_index = {}
        self._sorted_ts = []
        self.meta = {}  # video metadata + processing stats

    # ---------- writes ----------
    def add_event(self, timestamp, objects, description, actions=None):
        event_id = f"event_{len(self.events_by_id) + 1:03d}"
        ts = round(float(timestamp), 1)
        objs = sorted({normalize(o) for o in objects if o and o.strip()})
        event = {
            "id": event_id,
            "timestamp": ts,
            "objects": objs,
            "description": description.strip(),
            "actions": [a.strip() for a in (actions or []) if a and a.strip()],
        }
        self.events_by_id[event_id] = event
        for obj in objs:
            self.object_index.setdefault(obj, set()).add(event_id)
        if ts not in self.timestamp_index:
            bisect.insort(self._sorted_ts, ts)
        self.timestamp_index.setdefault(ts, set()).add(event_id)
        return event_id

    # ---------- reads ----------
    def _events(self, ids):
        return sorted((self.events_by_id[i] for i in ids), key=lambda e: e["timestamp"])

    def get_events_by_object(self, obj):
        """O(1) average per variant lookup in the inverted index."""
        ids = set()
        for v in variants(obj):
            ids |= self.object_index.get(v, set())
        return self._events(ids)

    def get_events_by_timestamp(self, timestamp, window=0.0):
        """Exact hash lookup when window == 0; otherwise binary-search a time range."""
        ts = round(float(timestamp), 1)
        if window <= 0:
            return self._events(self.timestamp_index.get(ts, set()))
        lo = bisect.bisect_left(self._sorted_ts, ts - window)
        hi = bisect.bisect_right(self._sorted_ts, ts + window)
        ids = set()
        for t in self._sorted_ts[lo:hi]:
            ids |= self.timestamp_index[t]
        return self._events(ids)

    def search_events(self, query, limit=20):
        """Keyword search: object-index hits score 3, action/description word hits score 1."""
        scores = {}
        for tok in tokenize(query):
            for v in variants(tok):
                for eid in self.object_index.get(v, ()):
                    scores[eid] = scores.get(eid, 0) + 3
        toks = set(tokenize(query))
        if toks:
            for eid, ev in self.events_by_id.items():
                words = set(tokenize(ev["description"] + " " + " ".join(ev["actions"])))
                hits = sum(1 for t in toks if variants(t) & words)
                if hits:
                    scores[eid] = scores.get(eid, 0) + hits
        ranked = sorted(scores, key=lambda e: (-scores[e], self.events_by_id[e]["timestamp"]))
        return [self.events_by_id[e] for e in ranked[:limit]]

    def all_events(self):
        return self._events(self.events_by_id.keys())

    def objects(self):
        return sorted(self.object_index)

    # ---------- persistence ----------
    def to_dict(self):
        return {
            "meta": self.meta,
            "events_by_id": self.events_by_id,
            "object_index": {k: sorted(v) for k, v in sorted(self.object_index.items())},
            "timestamp_index": {str(k): sorted(v) for k, v in sorted(self.timestamp_index.items())},
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data):
        mem = cls()
        mem.meta = data.get("meta", {})
        for ev in sorted(data.get("events_by_id", {}).values(), key=lambda e: e["timestamp"]):
            mem.add_event(ev["timestamp"], ev.get("objects", []), ev.get("description", ""), ev.get("actions", []))
        return mem

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
