"""Video loading and frame sampling with OpenCV."""
import hashlib

import cv2


class VideoError(Exception):
    pass


def file_hash(path, chunk=1 << 20):
    """SHA-256 of the video file; used as the memory cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def get_metadata(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open video. Is it a valid MP4?")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise VideoError("Video has no readable frames or an invalid frame rate.")
    return {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(frame_count / fps, 2),
        "width": width,
        "height": height,
    }


def _resize(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def sample_frames(path, interval=2.0, max_width=640, jpeg_quality=80):
    """Return [(timestamp_seconds, jpeg_bytes)] — one frame every `interval` seconds."""
    meta = get_metadata(path)
    fps = meta["fps"]
    cap = cv2.VideoCapture(path)
    frames = []
    t = 0.0
    while t < meta["duration"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            t += interval
            continue
        frame = _resize(frame, max_width)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            frames.append((round(t, 1), buf.tobytes()))
        t += interval
    cap.release()
    if not frames:
        raise VideoError("No usable frames could be decoded from this video.")
    return frames
