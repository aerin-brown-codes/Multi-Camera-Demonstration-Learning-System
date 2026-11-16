from feed.messages import stamp_fields, stamp_of
from feed.transforms import TABLE_FRAME

DETECTIONS_TOPIC_PREFIX = "/detect/"
DETECTIONS = "detections"
POSITIONS = "detections_3d"
END_OF_A_POSITIONS_TOPIC = "/" + POSITIONS

ClassBoxConfidence = tuple[str, tuple[float, float, float, float], float]
IdClassPointConfidence = tuple[str, str, tuple[float, float, float], float]


def detections_topic(camera: str) -> str:
    return f"{DETECTIONS_TOPIC_PREFIX}{camera}/{DETECTIONS}"


def positions_topic(camera: str) -> str:
    return f"{DETECTIONS_TOPIC_PREFIX}{camera}/{POSITIONS}"


def camera_of(topic: str, kind: str) -> str:
    camera, _, kind_of_the_topic = topic.removeprefix(DETECTIONS_TOPIC_PREFIX).partition("/")
    if not topic.startswith(DETECTIONS_TOPIC_PREFIX) or not camera or kind_of_the_topic != kind:
        raise ValueError(f"{topic} is not a {kind} topic: expected {DETECTIONS_TOPIC_PREFIX}<camera>/{kind}")
    return camera


def detections_message(camera: str, stamp: tuple[int, int], frame_id: str, found: list[ClassBoxConfidence]):
    fields = {
        "header": {"stamp": stamp_fields(stamp), "frame_id": frame_id},
        "detections": [{
            "id": str(place_in_the_list),
            "results": [{"hypothesis": {"class_id": class_name, "score": confidence}}],
            "bbox": {"center": {"position": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2}, "theta": 0.0},
                     "size_x": x1 - x0, "size_y": y1 - y0},
        } for place_in_the_list, (class_name, (x0, y0, x1, y1), confidence) in enumerate(found)],
    }
    return detections_topic(camera), fields, None


def detections_of(topic: str, fields: dict) -> tuple[str, tuple[int, int], list[ClassBoxConfidence]]:
    camera = camera_of(topic, DETECTIONS)
    found = []
    for detection in fields["detections"]:
        hypothesis, box = detection["results"][0]["hypothesis"], detection["bbox"]
        x, y = box["center"]["position"]["x"], box["center"]["position"]["y"]
        half_w, half_h = box["size_x"] / 2, box["size_y"] / 2
        found.append((hypothesis["class_id"], (x - half_w, y - half_h, x + half_w, y + half_h), hypothesis["score"]))
    return camera, stamp_of(fields), found


def positions_message(camera: str, stamp: tuple[int, int], found: list[IdClassPointConfidence]):
    fields = {
        "header": {"stamp": stamp_fields(stamp), "frame_id": TABLE_FRAME},
        "detections": [{
            "id": id_in_the_boxes_message,
            "results": [{"hypothesis": {"class_id": class_name, "score": confidence},
                         "pose": {"pose": {"position": {"x": x, "y": y, "z": z}}}}],
        } for id_in_the_boxes_message, class_name, (x, y, z), confidence in found],
    }
    return positions_topic(camera), fields, None


def positions_of(topic: str, fields: dict) -> tuple[str, tuple[int, int], list[IdClassPointConfidence]]:
    camera = camera_of(topic, POSITIONS)
    found = []
    for detection in fields["detections"]:
        result = detection["results"][0]
        position = result["pose"]["pose"]["position"]
        found.append((detection["id"], result["hypothesis"]["class_id"],
                      (position["x"], position["y"], position["z"]), result["hypothesis"]["score"]))
    return camera, stamp_of(fields), found
