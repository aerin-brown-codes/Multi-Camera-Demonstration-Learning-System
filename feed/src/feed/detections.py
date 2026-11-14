from feed.messages import stamp_fields, stamp_of

DETECTIONS_TOPIC_PREFIX = "/detect/"
DETECTIONS = "detections"

ClassBoxConfidence = tuple[str, tuple[float, float, float, float], float]


def detections_topic(camera: str) -> str:
    return f"{DETECTIONS_TOPIC_PREFIX}{camera}/{DETECTIONS}"


def detections_message(camera: str, stamp: tuple[int, int], frame_id: str, found: list[ClassBoxConfidence]):
    fields = {
        "header": {"stamp": stamp_fields(stamp), "frame_id": frame_id},
        "detections": [{
            "results": [{"hypothesis": {"class_id": class_name, "score": confidence}}],
            "bbox": {"center": {"position": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2}, "theta": 0.0},
                     "size_x": x1 - x0, "size_y": y1 - y0},
        } for class_name, (x0, y0, x1, y1), confidence in found],
    }
    return detections_topic(camera), fields, None


def detections_of(topic: str, fields: dict) -> tuple[str, tuple[int, int], list[ClassBoxConfidence]]:
    camera, _, kind = topic.removeprefix(DETECTIONS_TOPIC_PREFIX).partition("/")
    if not topic.startswith(DETECTIONS_TOPIC_PREFIX) or not camera or kind != DETECTIONS:
        raise ValueError(f"{topic} is not a detections topic: expected {DETECTIONS_TOPIC_PREFIX}<camera>/{DETECTIONS}")
    found = []
    for detection in fields["detections"]:
        hypothesis, box = detection["results"][0]["hypothesis"], detection["bbox"]
        x, y = box["center"]["position"]["x"], box["center"]["position"]["y"]
        half_w, half_h = box["size_x"] / 2, box["size_y"] / 2
        found.append((hypothesis["class_id"], (x - half_w, y - half_h, x + half_w, y + half_h), hypothesis["score"]))
    return camera, stamp_of(fields), found
