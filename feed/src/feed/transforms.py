import numpy as np

from feed.messages import stamp_fields

TF_STATIC_TOPIC = "/tf_static"
TABLE_FRAME = "table"


def quaternion_xyzw_of(R) -> np.ndarray:
    R = np.asarray(R, float)
    diagonal = np.diag(R)
    four_q_times_q = np.empty((4, 4))
    four_q_times_q[:3, :3] = R + R.T
    four_q_times_q[:3, 3] = four_q_times_q[3, :3] = R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]
    np.fill_diagonal(four_q_times_q, np.append(1 + 2 * diagonal - diagonal.sum(), 1 + diagonal.sum()))
    largest_part = int(np.argmax(np.diag(four_q_times_q)))
    return four_q_times_q[largest_part] / (2 * np.sqrt(four_q_times_q[largest_part, largest_part]))


def rotation_of(quaternion_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, float) / np.linalg.norm(quaternion_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def static_transforms_message(stamp: tuple[int, int], parent_frame: str,
                              T_parent_child_by_child_frame: dict[str, np.ndarray]):
    transforms = []
    for child_frame, T_parent_child in T_parent_child_by_child_frame.items():
        T_parent_child = np.asarray(T_parent_child, float)
        translation = dict(zip("xyz", T_parent_child[:3, 3].tolist()))
        rotation = dict(zip("xyzw", quaternion_xyzw_of(T_parent_child[:3, :3]).tolist()))
        transforms.append({"header": {"stamp": stamp_fields(stamp), "frame_id": parent_frame},
                           "child_frame_id": child_frame,
                           "transform": {"translation": translation, "rotation": rotation}})
    return TF_STATIC_TOPIC, {"transforms": transforms}, None


def static_transforms_of(fields: dict, parent_frame: str) -> dict[str, np.ndarray]:
    T_parent_child_by_child_frame = {}
    for entry in fields["transforms"]:
        child_frame, given_in = entry["child_frame_id"], entry["header"]["frame_id"]
        if given_in != parent_frame:
            raise ValueError(f"the pose of {child_frame} is given in '{given_in}', but only poses in "
                             f"'{parent_frame}' are read")
        translation, rotation = entry["transform"]["translation"], entry["transform"]["rotation"]
        T_parent_child = np.eye(4)
        T_parent_child[:3, :3] = rotation_of([rotation[part] for part in "xyzw"])
        T_parent_child[:3, 3] = [translation[axis] for axis in "xyz"]
        T_parent_child_by_child_frame[child_frame] = T_parent_child
    return T_parent_child_by_child_frame
