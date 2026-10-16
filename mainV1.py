import cv2
import math

from cvzone.HandTrackingModule import HandDetector


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0

WIDTH = 1280
HEIGHT = 720

PINCH_RATIO_ON = 0.40
PINCH_RATIO_OFF = 0.55

RESIZE_GRAB_RADIUS = 65
MIN_OBJECT_WIDTH = 120
MIN_OBJECT_HEIGHT = 70

RESIZE_DEADZONE = 0.04
RESIZE_SCALE_SMOOTHING = 0.30

SMOOTHING = 0.35
MAX_DRAWING_POINTS = 1500


# ============================================================
# OBJETS
# ============================================================

DEFAULT_OBJECTS = [
    (180, 180, 220, 130, "OBJECT 1", (255, 100, 80)),
    (490, 300, 240, 140, "OBJECT 2", (100, 220, 100)),
    (780, 150, 210, 120, "OBJECT 3", (100, 150, 255)),
]

objects = [
    {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "text": label,
        "color": color
    }
    for x, y, w, h, label, color in DEFAULT_OBJECTS
]


# ============================================================
# FUNCTIONS
# ============================================================

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_hand_center(hand):
    lm = hand["lmList"]
    ids = [0, 5, 9, 13, 17]

    x = sum(lm[i][0] for i in ids) / len(ids)
    y = sum(lm[i][1] for i in ids) / len(ids)

    return x, y


def point_inside_object(px, py, obj):
    return (
        obj["x"] <= px <= obj["x"] + obj["w"]
        and obj["y"] <= py <= obj["y"] + obj["h"]
    )


def draw_object(img, obj, selected=False, resizing=False):
    x = int(obj["x"])
    y = int(obj["y"])
    w = int(obj["w"])
    h = int(obj["h"])

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        obj["color"],
        -1
    )

    cv2.addWeighted(
        overlay, 0.55,
        img, 0.45,
        0, img
    )

    thickness = 5 if selected else 2

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        thickness
    )

    cv2.putText(
        img,
        obj["text"],
        (x + 20, y + min(70, h - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    handle_radius = 14 if resizing else 10
    handle_color = (
        (0, 255, 255) if resizing
        else (255, 255, 255)
    )

    for pt in [(x, y), (x + w, y + h)]:
        cv2.circle(
            img, pt, handle_radius,
            handle_color, -1
        )
        cv2.circle(
            img, pt, handle_radius,
            (0, 0, 0), 2
        )


def get_hand_info(hand):
    lm = hand["lmList"]

    thumb = lm[4]
    index = lm[8]
    wrist = lm[0]
    middle_mcp = lm[9]

    hand_size = max(
        distance(wrist, middle_mcp), 1
    )

    pinch_distance = distance(thumb, index)
    pinch_ratio = pinch_distance / hand_size

    pinch_center = (
        (thumb[0] + index[0]) / 2,
        (thumb[1] + index[1]) / 2
    )

    return {
        "hand": hand,
        "index": (int(index[0]), int(index[1])),
        "thumb": (int(thumb[0]), int(thumb[1])),
        "pinch_center": pinch_center,
        "palm": get_hand_center(hand),
        "ratio": pinch_ratio,
        "pinch_on": pinch_ratio < PINCH_RATIO_ON,
        "pinch_hold": pinch_ratio < PINCH_RATIO_OFF
    }


# ============================================================
# WEBCAM INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")


# ============================================================
# HAND DETECTOR
# ============================================================

detector = HandDetector(
    staticMode=False,
    maxHands=2,
    modelComplexity=1,
    detectionCon=0.7,
    minTrackCon=0.5
)


# ============================================================
# STATE VARIABLES 
# ============================================================

selected_object = None
active_palm = None

pinching = False

drag_offset_x = 0
drag_offset_y = 0

smooth_cursor = None

resizing = None

drawing_points = []


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:

        success, img = cap.read()

        if not success:
            break

        # Effet miroir
        img = cv2.flip(img, 1)

        height, width = img.shape[:2]

        # Detection des deux mains
        hands, img = detector.findHands(
            img,
            draw=True,
            flipType=False
        )

        hand_infos = [
            get_hand_info(hand)
            for hand in hands
        ]


        # ====================================================
        # CURSEURS INDEPENDANTS DES DEUX MAINS
        # ====================================================

        for i, info in enumerate(hand_infos, start=1):

            index_cursor = info["index"]
            thumb_cursor = info["thumb"]

            this_hand_pinching = info["pinch_hold"]

            if this_hand_pinching:
                cursor_color = (0, 255, 255)
                pinch_label = f"HAND {i} PINCH"

                cv2.line(
                    img,
                    thumb_cursor,
                    index_cursor,
                    cursor_color,
                    3
                )
            else:
                cursor_color = (0, 100, 255)
                pinch_label = f"HAND {i}"

            cv2.circle(
                img,
                index_cursor,
                22,
                (255, 255, 255),
                2
            )

            cv2.circle(
                img,
                index_cursor,
                11,
                cursor_color,
                -1
            )

            cv2.putText(
                img,
                pinch_label,
                (
                    index_cursor[0] + 26,
                    index_cursor[1] - 12
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                cursor_color,
                1
            )


        # ====================================================
        # RESIZE FOR 2 HANDS
        # ====================================================

        resize_corner_reserved = False

        if len(hand_infos) == 2:

            info_a = hand_infos[0]
            info_b = hand_infos[1]

            # Verification des coins
            for obj in reversed(objects):

                top_left = (
                    obj["x"],
                    obj["y"]
                )

                bottom_right = (
                    obj["x"] + obj["w"],
                    obj["y"] + obj["h"]
                )

                for info in hand_infos:

                    if info["pinch_hold"]:

                        pc = info["pinch_center"]

                        if (
                            distance(pc, top_left)
                            < RESIZE_GRAB_RADIUS
                            or
                            distance(pc, bottom_right)
                            < RESIZE_GRAB_RADIUS
                        ):
                            resize_corner_reserved = True
                            break

                if resize_corner_reserved:
                    break

            both_pinching = (
                info_a["pinch_on"]
                and info_b["pinch_on"]
            )


            # ================================================
            # RESIZE START
            # ================================================

            if resizing is None and both_pinching:

                for obj in reversed(objects):

                    top_left = (
                        obj["x"],
                        obj["y"]
                    )

                    bottom_right = (
                        obj["x"] + obj["w"],
                        obj["y"] + obj["h"]
                    )

                    a = info_a["pinch_center"]
                    b = info_b["pinch_center"]

                    normal_assignment = (
                        distance(a, top_left)
                        < RESIZE_GRAB_RADIUS
                        and
                        distance(b, bottom_right)
                        < RESIZE_GRAB_RADIUS
                    )

                    reversed_assignment = (
                        distance(b, top_left)
                        < RESIZE_GRAB_RADIUS
                        and
                        distance(a, bottom_right)
                        < RESIZE_GRAB_RADIUS
                    )

                    if not (
                        normal_assignment
                        or reversed_assignment
                    ):
                        continue

                    if normal_assignment:
                        tl_info = info_a
                        br_info = info_b
                    else:
                        tl_info = info_b
                        br_info = info_a

                    palm_tl = tl_info["palm"]
                    palm_br = br_info["palm"]

                    resizing = {
                        "object": obj,
                        "initial_distance": max(
                            distance(palm_tl, palm_br), 1
                        ),
                        "initial_w": obj["w"],
                        "initial_h": obj["h"],
                        "object_center": (
                            obj["x"] + obj["w"] / 2,
                            obj["y"] + obj["h"] / 2
                        ),
                        "hand_midpoint": (
                            (palm_tl[0] + palm_br[0]) / 2,
                            (palm_tl[1] + palm_br[1]) / 2
                        ),
                        "prev_tl": palm_tl,
                        "prev_br": palm_br,
                        "scale": 1.0
                    }

                    selected_object = None
                    pinching = False
                    active_palm = None

                    objects.remove(obj)
                    objects.append(obj)

                    print("RESIZE START")
                    break


            # ================================================
            # RESIZE IN PROGRESS
            # ================================================

            if resizing is not None:

                h0 = hand_infos[0]
                h1 = hand_infos[1]

                cost_normal = (
                    distance(
                        h0["palm"],
                        resizing["prev_tl"]
                    )
                    +
                    distance(
                        h1["palm"],
                        resizing["prev_br"]
                    )
                )

                cost_reversed = (
                    distance(
                        h1["palm"],
                        resizing["prev_tl"]
                    )
                    +
                    distance(
                        h0["palm"],
                        resizing["prev_br"]
                    )
                )

                if cost_normal <= cost_reversed:
                    tl_info = h0
                    br_info = h1
                else:
                    tl_info = h1
                    br_info = h0

                still_pinching = (
                    tl_info["pinch_hold"]
                    and br_info["pinch_hold"]
                )

                if still_pinching:

                    p1 = tl_info["palm"]
                    p2 = br_info["palm"]

                    resizing["prev_tl"] = p1
                    resizing["prev_br"] = p2

                    current_distance = distance(p1, p2)

                    raw_scale = (
                        current_distance
                        / resizing["initial_distance"]
                    )

                    # Zone morte
                    if abs(raw_scale - 1.0) <= RESIZE_DEADZONE:
                        target_scale = 1.0

                    elif raw_scale > 1.0:
                        target_scale = (
                            raw_scale - RESIZE_DEADZONE
                        )

                    else:
                        target_scale = (
                            raw_scale + RESIZE_DEADZONE
                        )

                    # Lissage du resize
                    resizing["scale"] = (
                        (1 - RESIZE_SCALE_SMOOTHING)
                        * resizing["scale"]
                        +
                        RESIZE_SCALE_SMOOTHING
                        * target_scale
                    )

                    initial_w = resizing["initial_w"]
                    initial_h = resizing["initial_h"]

                    min_scale = max(
                        MIN_OBJECT_WIDTH / initial_w,
                        MIN_OBJECT_HEIGHT / initial_h
                    )

                    max_scale = min(
                        (width - 10) / initial_w,
                        (height - 10) / initial_h
                    )

                    resizing["scale"] = max(
                        min_scale,
                        min(resizing["scale"], max_scale)
                    )

                    new_w = (
                        initial_w * resizing["scale"]
                    )

                    new_h = (
                        initial_h * resizing["scale"]
                    )

                    current_midpoint = (
                        (p1[0] + p2[0]) / 2,
                        (p1[1] + p2[1]) / 2
                    )

                    delta_x = (
                        current_midpoint[0]
                        - resizing["hand_midpoint"][0]
                    )

                    delta_y = (
                        current_midpoint[1]
                        - resizing["hand_midpoint"][1]
                    )

                    center_x = (
                        resizing["object_center"][0]
                        + delta_x
                    )

                    center_y = (
                        resizing["object_center"][1]
                        + delta_y
                    )

                    new_x = center_x - new_w / 2
                    new_y = center_y - new_h / 2

                    new_x = max(
                        0,
                        min(new_x, width - new_w)
                    )

                    new_y = max(
                        0,
                        min(new_y, height - new_h)
                    )

                    obj = resizing["object"]

                    obj["x"] = new_x
                    obj["y"] = new_y
                    obj["w"] = new_w
                    obj["h"] = new_h

                    # Ligne entre les deux pinchs
                    pc1 = tuple(
                        map(int, tl_info["pinch_center"])
                    )

                    pc2 = tuple(
                        map(int, br_info["pinch_center"])
                    )

                    cv2.line(
                        img,
                        pc1,
                        pc2,
                        (255, 0, 255),
                        4
                    )

                    cv2.circle(
                        img, pc1, 22,
                        (255, 0, 255), 4
                    )

                    cv2.circle(
                        img, pc2, 22,
                        (255, 0, 255), 4
                    )

                else:
                    print("RESIZE END")
                    resizing = None

        else:
            # Une des deux mains a disparu
            resizing = None


        # ====================================================
        # CHOICE OF THE CONTROLLING HAND 
        # ====================================================

        if hand_infos:

            control_hand = None

            # Garder la main qui effectue deja le drag
            if pinching and active_palm is not None:

                control_hand = min(
                    hand_infos,
                    key=lambda h: distance(
                        h["palm"],
                        active_palm
                    )
                )

            # Sinon choisir une main qui commence un pinch
            if control_hand is None:

                pinch_candidates = [
                    info
                    for info in hand_infos
                    if info["pinch_on"]
                ]

                if pinch_candidates:

                    control_hand = min(
                        pinch_candidates,
                        key=lambda h: h["ratio"]
                    )

            # Sinon main proche du curseur precedent
            if control_hand is None:

                if smooth_cursor is not None:

                    control_hand = min(
                        hand_infos,
                        key=lambda h: distance(
                            h["index"],
                            smooth_cursor
                        )
                    )

                else:
                    control_hand = hand_infos[0]


            # ================================================
            # CURSEUR SMOOTH
            # ================================================

            index_x, index_y = control_hand["index"]

            if smooth_cursor is None:

                smooth_cursor = (
                    index_x,
                    index_y
                )

            else:

                smooth_cursor = (
                    SMOOTHING * index_x
                    + (1 - SMOOTHING) * smooth_cursor[0],

                    SMOOTHING * index_y
                    + (1 - SMOOTHING) * smooth_cursor[1]
                )

            cursor_x = int(smooth_cursor[0])
            cursor_y = int(smooth_cursor[1])


            # ================================================
            # PINCH
            # ================================================

            was_pinching = pinching

            if resizing is not None:

                pinching = False
                selected_object = None

            elif not pinching:

                if control_hand["pinch_on"]:
                    pinching = True

            else:

                if not control_hand["pinch_hold"]:
                    pinching = False

            if pinching:
                active_palm = control_hand["palm"]
            else:
                active_palm = None

            pinch_started = (
                pinching and not was_pinching
            )

            pinch_released = (
                not pinching and was_pinching
            )


            # ================================================
            # DRAG & DROP
            # ================================================

            if (
                pinch_started
                and resizing is None
                and not resize_corner_reserved
            ):

                for obj in reversed(objects):

                    if point_inside_object(
                        cursor_x,
                        cursor_y,
                        obj
                    ):

                        selected_object = obj

                        drag_offset_x = (
                            cursor_x - obj["x"]
                        )

                        drag_offset_y = (
                            cursor_y - obj["y"]
                        )

                        objects.remove(obj)
                        objects.append(obj)

                        break


            # ================================================
            # MOVING OF THE OBJECT
            # ================================================

            if (
                resizing is None
                and pinching
                and selected_object is not None
            ):

                selected_object["x"] = (
                    cursor_x - drag_offset_x
                )

                selected_object["y"] = (
                    cursor_y - drag_offset_y
                )

                selected_object["x"] = max(
                    0,
                    min(
                        selected_object["x"],
                        width - selected_object["w"]
                    )
                )

                selected_object["y"] = max(
                    0,
                    min(
                        selected_object["y"],
                        height - selected_object["h"]
                    )
                )


            # ================================================
            # RELEASE
            # ================================================

            if pinch_released:
                selected_object = None


            # ================================================
            # HAND GESTURE RECOGNITION
            # ================================================

            fingers = detector.fingersUp(
                control_hand["hand"]
            )

            if fingers == [0, 0, 0, 0, 0]:
                gesture_text = "POING"

            elif fingers == [0, 1, 0, 0, 0]:
                gesture_text = "CURSEUR"

            elif fingers == [0, 1, 1, 0, 0]:
                gesture_text = "DESSIN"

            elif fingers == [1, 0, 0, 0, 0]:
                gesture_text = "LIKE"

            elif fingers == [1, 1, 1, 1, 1]:
                gesture_text = "MAIN OUVERTE"

            else:
                gesture_text = str(fingers)


            # ================================================
            # DRAWING MODE (this one is useless but I was having fun lol)
            # ================================================

            drawing_mode = (
                resizing is None
                and not pinching
                and fingers == [0, 1, 1, 0, 0]
            )

            if drawing_mode:

                drawing_points.append(
                    (cursor_x, cursor_y)
                )

                if len(drawing_points) > MAX_DRAWING_POINTS:
                    drawing_points.pop(0)

            else:

                if (
                    drawing_points
                    and drawing_points[-1] is not None
                ):
                    drawing_points.append(None)

            if resizing is not None:
                gesture_text = (
                    f"RESIZE x{resizing['scale']:.2f}"
                )


            # ================================================
            # PINCH VISUALIZATION
            # ================================================

            if pinching and resizing is None:

                thumb_x, thumb_y = control_hand["thumb"]
                index_x, index_y = control_hand["index"]

                cv2.line(
                    img,
                    (thumb_x, thumb_y),
                    (index_x, index_y),
                    (0, 255, 255),
                    4
                )

                center_x = int(
                    (thumb_x + index_x) / 2
                )

                center_y = int(
                    (thumb_y + index_y) / 2
                )

                cv2.circle(
                    img,
                    (center_x, center_y),
                    15,
                    (0, 255, 255),
                    -1
                )

        else:

            gesture_text = "Aucune main"

            pinching = False
            selected_object = None
            active_palm = None
            smooth_cursor = None
            resizing = None

            if (
                drawing_points
                and drawing_points[-1] is not None
            ):
                drawing_points.append(None)


        # ====================================================
        # LINES DISPLAY
        # ====================================================

        previous_point = None

        for point in drawing_points:

            if point is None:
                previous_point = None
                continue

            if previous_point is not None:

                cv2.line(
                    img,
                    previous_point,
                    point,
                    (255, 0, 255),
                    6
                )

            previous_point = point


        # ====================================================
        # BOXES DISPLAY
        # ====================================================

        for obj in objects:

            draw_object(
                img,
                obj,
                selected=(obj is selected_object),
                resizing=(
                    resizing is not None
                    and obj is resizing["object"]
                )
            )


        # ====================================================
        # INTERFACE
        # ====================================================

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (15, 15),
            (500, 150),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.6,
            img,
            0.4,
            0,
            img
        )

        cv2.putText(
            img,
            gesture_text,
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

        status = (
            "PINCH : ON"
            if pinching
            else "PINCH : OFF"
        )

        cv2.putText(
            img,
            status,
            (35, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            img,
            "C = clear | R = reset | Q = quitter",
            (35, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )


        # ====================================================
        # WEBCAM DISPLAY
        # ====================================================

        cv2.imshow(
            "Minority Report Hand Control",
            img
        )


        # ====================================================
        # CLAVIER
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            drawing_points.clear()

        if key == ord("r"):

            defaults = {
                label: (x, y, w, h)
                for x, y, w, h, label, color
                in DEFAULT_OBJECTS
            }

            for obj in objects:

                x, y, w, h = defaults[obj["text"]]

                obj["x"] = x
                obj["y"] = y
                obj["w"] = w
                obj["h"] = h

            selected_object = None
            resizing = None
            pinching = False
            active_palm = None


# ============================================================
# CLEANUP
# ============================================================

finally:
    cap.release()
    cv2.destroyAllWindows()