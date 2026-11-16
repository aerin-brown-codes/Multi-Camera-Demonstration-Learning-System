import numpy as np

CAMERA_TOPIC_PREFIX = "/camera/"
TRUTH_TOPIC = "truth"
CAMERA_INFO = "color/camera_info"
COLOR = "color/image_raw"
DEPTH = "aligned_depth_to_color/image_raw"

METRES_PER_DEPTH_UNIT = 0.001
NANOSECONDS_PER_SECOND = 1_000_000_000
DISTORTION_MODEL = "plumb_bob"
NO_LENS_DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]

ENCODING = {COLOR: "rgb8", DEPTH: "16UC1"}
PIXEL_TYPE = {COLOR: np.dtype("u1"), DEPTH: np.dtype("<u2")}
CHANNELS = {COLOR: 3, DEPTH: 1}


def stamp_from_seconds(t: float) -> tuple[int, int]:
    return divmod(round(t * NANOSECONDS_PER_SECOND), NANOSECONDS_PER_SECOND)


def seconds_from_stamp(stamp: tuple[int, int]) -> float:
    return stamp[0] + stamp[1] / NANOSECONDS_PER_SECOND


def stamp_fields(stamp: tuple[int, int]) -> dict:
    return {"sec": stamp[0], "nanosec": stamp[1]}


def stamp_from_fields(fields: dict) -> tuple[int, int]:
    return fields["sec"], fields["nanosec"]


def stamp_of(fields: dict) -> tuple[int, int]:
    return stamp_from_fields(fields["header"]["stamp"])


def camera_topic_prefix(camera: str) -> str:
    return f"{CAMERA_TOPIC_PREFIX}{camera}/"


def camera_info_topic(camera: str) -> str:
    return camera_topic_prefix(camera) + CAMERA_INFO


def color_topic(camera: str) -> str:
    return camera_topic_prefix(camera) + COLOR


def depth_topic(camera: str) -> str:
    return camera_topic_prefix(camera) + DEPTH


def camera_and_kind(topic: str) -> tuple[str, str]:
    if topic.startswith(CAMERA_TOPIC_PREFIX):
        camera, _, kind = topic[len(CAMERA_TOPIC_PREFIX):].partition("/")
        if camera and kind in (CAMERA_INFO, COLOR, DEPTH):
            return camera, kind
    raise ValueError(f"{topic} is not a camera topic: expected {CAMERA_TOPIC_PREFIX}<camera>/ followed by "
                     f"{CAMERA_INFO}, {COLOR} or {DEPTH}")


def color_optical_frame(camera: str) -> str:
    return f"{camera}_color_optical_frame"


def header(camera: str, stamp: tuple[int, int]) -> dict:
    return {"stamp": stamp_fields(stamp), "frame_id": color_optical_frame(camera)}


def camera_info_message(camera: str, stamp: tuple[int, int], K: np.ndarray, width: int, height: int):
    fields = {
        "header": header(camera, stamp), "height": height, "width": width,
        "distortion_model": DISTORTION_MODEL, "d": NO_LENS_DISTORTION, "k": np.asarray(K, float).ravel().tolist(),
    }
    return camera_info_topic(camera), fields, None


def K_of(fields: dict) -> np.ndarray:
    return np.array(fields["k"]).reshape(3, 3)


def image_message(camera: str, kind: str, stamp: tuple[int, int], image: np.ndarray):
    topic = camera_topic_prefix(camera) + kind
    pixel_type, channels = PIXEL_TYPE[kind], CHANNELS[kind]
    shape_fits = image.ndim == 2 if channels == 1 else image.ndim == 3 and image.shape[2] == channels
    if image.dtype != pixel_type or not shape_fits:
        raise ValueError(f"{topic} carries {ENCODING[kind]}: {pixel_type.itemsize * 8}-bit whole numbers, "
                         f"{channels} per pixel; got {image.dtype} with shape {image.shape}")
    height, width = image.shape[:2]
    fields = {
        "header": header(camera, stamp), "height": height, "width": width,
        "encoding": ENCODING[kind], "is_bigendian": 0, "step": width * channels * pixel_type.itemsize,
    }
    return topic, fields, image


def color_message(camera: str, stamp: tuple[int, int], rgb: np.ndarray):
    return image_message(camera, COLOR, stamp, rgb)


def depth_message(camera: str, stamp: tuple[int, int], depth: np.ndarray):
    return image_message(camera, DEPTH, stamp, depth)


def image_of(topic: str, fields: dict, data: bytes) -> np.ndarray:
    _, kind = camera_and_kind(topic)
    if kind == CAMERA_INFO:
        raise ValueError(f"{topic} carries camera info, not an image")
    pixel_type, channels = PIXEL_TYPE[kind], CHANNELS[kind]
    height, width, step = fields["height"], fields["width"], fields["step"]
    if fields["encoding"] != ENCODING[kind]:
        raise ValueError(f"{topic} carries {ENCODING[kind]}, but this image says {fields['encoding']}")
    if fields["is_bigendian"] != 0:
        raise ValueError(f"{topic}: this image is big-endian, only little-endian is read")
    if step != width * channels * pixel_type.itemsize:
        raise ValueError(f"{topic}: step is {step} bytes a row, but {width} pixels of {ENCODING[kind]} "
                         f"are {width * channels * pixel_type.itemsize}")
    if data is None or len(data) != step * height:
        raise ValueError(f"{topic}: {0 if data is None else len(data)} bytes of pixels, but step x height "
                         f"is {step} x {height} = {step * height}")
    shape = (height, width, channels) if channels > 1 else (height, width)
    return np.frombuffer(data, pixel_type).reshape(shape).copy()
