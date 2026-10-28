import os
import glob
import math
import csv
import statistics
import threading
import time
import multiprocessing as mproc

import cv2
import numpy as np
import mediapipe as mp

print("step5: script loaded", flush=True)

# ---------------- الإعدادات ----------------

FACE_MODEL = "models/face_landmarker.task"
OBJ_MODEL = "models/efficientdet_lite0.tflite"

VIDEOS_GLOB = "videos/*.mp4"
OUT_DIR = "output_smart"

NUM_FACES = 5

# عرض الفيديو أثناء حفظه
SHOW = True

# تشغيل كشف الموبايل
USE_PHONES = True

# تصغير الفريم قبل التحليل لتوفير الوقت والذاكرة
DETECT_MAX_W = 960

# كشف الموبايل كل فريمين
PHONE_STRIDE = 2

# أقصى مهلة لانتظار عملية كشف الموبايل، بالثواني
PHONE_TIMEOUT = 900

# ---------------- مستويات لفّة الرأس ----------------

YELLOW_YAW = 15.0
RED_YAW = 30.0
RED_YAW_OFF = 25.0

# صفر يعني تشغيل التنبيه من أول فريم
RED_SECONDS = 0.0
RECOVER_SECONDS = 0.3

# الرأس لأسفل: قراءة أو كتابة
PITCH_DROP = 10.0

# لو النظر لأسفل ينتج قيمة سالبة، غيّرها إلى -1.0
PITCH_SIGN = 1.0

# ---------------- إعدادات الموبايل ----------------

PHONE_SCORE = 0.25

# أقصى مسافة بين مركز الموبايل والوجه، بوحدات ارتفاع الوجه
PHONE_NEAR = 3.0

PHONE_SECONDS = 0.8
PHONE_RECOVER = 0.5

# الهامش حول الوجه، كنسبة من عرض الوجه
PHONE_FACE_MARGIN = 0.20

# استبعاد الموبايل لو أكثر من نصف مساحته داخل منطقة الوجه الموسّعة
PHONE_MAX_FACE_OVERLAP = 0.50

# ---------------- الوضع الطبيعي والتتبع ----------------

BASE_BAND = 8.0
BASE_MAX = 45.0

# أقل = استجابة أسرع
SMOOTH = 0.4

MAX_MATCH_DIST = 0.15
MAX_MISSED = 10
MIN_TRACK_FRAMES = 15

# ---------------- الألوان والرسم ----------------

GREEN = (0, 200, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX

NOSE = 1
LEFT_EDGE = 234
RIGHT_EDGE = 454

# ---------------- MediaPipe ----------------

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

RunningMode = mp.tasks.vision.RunningMode


# ---------------- القياس ----------------

def rotation_angles(matrix):
    """حساب yaw وpitch من مصفوفة تحويل الوجه."""
    R = np.asarray(matrix, dtype=np.float64)[:3, :3]

    sy = math.hypot(R[2, 1], R[2, 2])

    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))

    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return yaw, PITCH_SIGN * pitch


def shrink(frame):
    """تصغير الفريم عند الحاجة، وإرجاع نسبة التصغير."""
    h, w = frame.shape[:2]

    if w <= DETECT_MAX_W:
        return frame, 1.0

    sc = DETECT_MAX_W / w

    small = cv2.resize(
        frame,
        (DETECT_MAX_W, int(h * sc)),
        interpolation=cv2.INTER_AREA,
    )

    return small, sc


def to_pixels(face, w, h):
    """تحويل إحداثيات معالم الوجه النسبية إلى بكسلات."""
    return np.array(
        [[p.x * w, p.y * h] for p in face],
        dtype=np.float32,
    )


# ---------------- التتبع ----------------

def assign_tracks(tracks, centers, max_dist):
    pairs = []

    for fi, (cx, cy) in enumerate(centers):
        for tid, track in tracks.items():
            distance = math.hypot(
                cx - track["center"][0],
                cy - track["center"][1],
            )

            if distance < max_dist:
                pairs.append((distance, fi, tid))

    pairs.sort()

    face_to_tid = {}
    used = set()

    for distance, fi, tid in pairs:
        if fi in face_to_tid or tid in used:
            continue

        face_to_tid[fi] = tid
        used.add(tid)

    return face_to_tid


def safe_close(obj, timeout=10):
    """إغلاق MediaPipe بمهلة لأن close قد يعلق في بعض الإصدارات."""
    thread = threading.Thread(
        target=obj.close,
        daemon=True,
    )

    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(
            "  (close() timed out, continuing)",
            flush=True,
        )


# ---------------- كشف الموبايل ----------------

def detect_phones(path):
    """تشغيل كشف الموبايل داخل عملية منفصلة."""
    if not os.path.exists(OBJ_MODEL):
        print(
            "  [phones] model not found: %s -> phone detection disabled"
            % OBJ_MODEL,
            flush=True,
        )
        return None

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = ObjectDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=OBJ_MODEL,
        ),
        running_mode=RunningMode.VIDEO,
        score_threshold=PHONE_SCORE,
        max_results=10,
    )

    detector = ObjectDetector.create_from_options(options)

    out = []
    idx = 0
    last = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if idx % PHONE_STRIDE == 0:
            small, sc = shrink(frame)

            rgb = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2RGB,
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = detector.detect_for_video(
                image,
                int(idx * 1000 / fps),
            )

            last = []

            for detection in result.detections:
                if not detection.categories:
                    continue

                category = detection.categories[0]
                category_name = (
                    category.category_name or ""
                ).lower()

                if "phone" not in category_name:
                    continue

                box = detection.bounding_box

                last.append((
                    box.origin_x / sc,
                    box.origin_y / sc,
                    (box.origin_x + box.width) / sc,
                    (box.origin_y + box.height) / sc,
                    category.score,
                ))

        out.append(last)

        if idx % 100 == 0:
            print(
                "  [phones] frame %d / %d" % (idx, total),
                flush=True,
            )

        idx += 1

    cap.release()

    # لا نستدعي close هنا؛ العملية ستنتهي بعد إرسال النتائج.
    return out


def _phone_worker(path, conn):
    result = None

    for attempt in range(1, 4):
        try:
            result = detect_phones(path)
            break

        except Exception:
            print(
                "  [phones] attempt %d failed, retrying..."
                % attempt,
                flush=True,
            )
            time.sleep(2)

    conn.send(result)
    conn.close()

    os._exit(0)


def start_phone_process(path):
    if not USE_PHONES:
        return None, None

    parent, child = mproc.Pipe(False)

    process = mproc.Process(
        target=_phone_worker,
        args=(path, child),
        daemon=True,
    )

    process.start()
    child.close()

    return process, parent


def collect_phones(process, conn, n_frames):
    if process is None:
        return []

    result = None

    try:
        if conn.poll(PHONE_TIMEOUT):
            result = conn.recv()
        else:
            print(
                "  phone detector timed out, skipping it",
                flush=True,
            )

    except (EOFError, OSError):
        result = None

    process.join(3)

    if process.is_alive():
        process.terminate()

    return result


# ---------------- استبعاد الموبايلات فوق الوجه ----------------

def drop_face_phones(phones, faces):
    """
    استبعاد اكتشافات الموبايل التي تتداخل مع منطقة الوجه الموسّعة.

    نسبة التداخل محسوبة من مساحة مربع الموبايل،
    وليس من مساحة الوجه.
    """
    face_regions = []

    for face in faces:
        pts = np.asarray(
            face["pts"],
            dtype=np.float32,
        )

        if (
            pts.ndim != 2
            or pts.shape[0] == 0
            or pts.shape[1] != 2
            or not np.isfinite(pts).all()
        ):
            continue

        fx1, fy1 = pts.min(axis=0)
        fx2, fy2 = pts.max(axis=0)

        face_width = float(fx2 - fx1)
        face_height = float(fy2 - fy1)

        if face_width <= 0 or face_height <= 0:
            continue

        margin = PHONE_FACE_MARGIN * face_width

        face_regions.append((
            fx1 - margin,
            fy1 - margin,
            fx2 + margin,
            fy2 + margin,
        ))

    kept = []

    for phone in phones:
        x1, y1, x2, y2, score = map(float, phone)

        if not np.isfinite(
            [x1, y1, x2, y2, score]
        ).all():
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        phone_area = (x2 - x1) * (y2 - y1)

        on_face = False

        for fx1, fy1, fx2, fy2 in face_regions:
            overlap_width = max(
                0.0,
                min(x2, fx2) - max(x1, fx1),
            )

            overlap_height = max(
                0.0,
                min(y2, fy2) - max(y1, fy1),
            )

            overlap_ratio = (
                overlap_width * overlap_height
            ) / phone_area

            if overlap_ratio > PHONE_MAX_FACE_OVERLAP:
                on_face = True
                break

        if not on_face:
            kept.append(phone)

    return kept


# ---------------- تحليل الفيديو ----------------

def analyze(path):
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=FACE_MODEL,
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=NUM_FACES,
        output_facial_transformation_matrixes=True,
    )

    process, conn = start_phone_process(path)

    landmarker = FaceLandmarker.create_from_options(
        face_options
    )

    tracks = {}
    next_id = 1
    frames = []
    idx = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        small, _ = shrink(frame)

        rgb = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB,
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp = int(idx * 1000 / fps)

        if idx % 30 == 0:
            print(
                "  analyzing frame %d / %d" % (idx, total),
                end="\r",
                flush=True,
            )

        result = landmarker.detect_for_video(
            image,
            timestamp,
        )

        matrices = (
            result.facial_transformation_matrixes or []
        )

        faces = []

        for fi, face in enumerate(result.face_landmarks):
            if fi >= len(matrices):
                continue

            pts = to_pixels(face, w, h)

            yaw, pitch = rotation_angles(
                matrices[fi]
            )

            faces.append((
                pts,
                yaw,
                pitch,
                pts.mean(axis=0),
            ))

        matches = assign_tracks(
            tracks,
            [face[3] for face in faces],
            MAX_MATCH_DIST * w,
        )

        used = set()
        found = []

        for fi, (pts, yaw, pitch, center) in enumerate(faces):
            tid = matches.get(fi)

            if tid is None:
                tid = next_id
                next_id += 1

            tracks[tid] = {
                "center": (
                    float(center[0]),
                    float(center[1]),
                ),
                "missed": 0,
            }

            used.add(tid)

            found.append({
                "id": tid,
                "pts": pts.astype(np.int32),
                "yaw": yaw,
                "pitch": pitch,
                "center": (
                    float(center[0]),
                    float(center[1]),
                ),
                "fh": float(
                    pts[:, 1].max() - pts[:, 1].min()
                ),
            })

        for tid in list(tracks.keys()):
            if tid not in used:
                tracks[tid]["missed"] += 1

                if tracks[tid]["missed"] > MAX_MISSED:
                    del tracks[tid]

        frames.append({
            "faces": found,
            "phones": [],
        })

        idx += 1

    print(" " * 50, end="\r")

    cap.release()
    safe_close(landmarker)

    print(
        "  waiting for phone detector...",
        flush=True,
    )

    phones_all = collect_phones(
        process,
        conn,
        len(frames),
    )

    if (
        phones_all is None
        and USE_PHONES
        and os.path.exists(OBJ_MODEL)
    ):
        print(
            "  phone detector crashed, retrying once...",
            flush=True,
        )

        process, conn = start_phone_process(path)

        phones_all = collect_phones(
            process,
            conn,
            len(frames),
        )

    if phones_all is None:
        print(
            "  phone detection unavailable for this video",
            flush=True,
        )
        phones_all = []

    # الفلترة هنا قبل اتخاذ القرار والرسم وتسجيل الأحداث.
    for i, frame_data in enumerate(frames):
        raw_phones = (
            phones_all[i]
            if i < len(phones_all)
            else []
        )

        frame_data["phones"] = drop_face_phones(
            raw_phones,
            frame_data["faces"],
        )

    return frames, fps, w, h


# ---------------- الوضع الطبيعي لكل طالب ----------------

def robust_baseline(values):
    baseline = statistics.median(values)

    for _ in range(3):
        near = [
            value
            for value in values
            if abs(value - baseline) <= BASE_BAND
        ]

        if len(near) < max(
            MIN_TRACK_FRAMES,
            0.3 * len(values),
        ):
            break

        baseline = statistics.median(near)

    return max(
        -BASE_MAX,
        min(BASE_MAX, baseline),
    )


def compute_baselines(frames):
    yaws = {}
    pitches = {}

    for frame_data in frames:
        for detection in frame_data["faces"]:
            tid = detection["id"]

            yaws.setdefault(tid, []).append(
                detection["yaw"]
            )

            pitches.setdefault(tid, []).append(
                detection["pitch"]
            )

    baselines = {}

    for tid, values in yaws.items():
        if len(values) >= MIN_TRACK_FRAMES:
            baselines[tid] = (
                robust_baseline(values),
                robust_baseline(pitches[tid]),
            )

    return baselines


# ---------------- منطق القرار ----------------

class Trigger:
    """
    عدّاد تنبيه.

    يبدأ بعد need_on_s،
    وينتهي بعد need_off_s من غياب الشرط.
    """

    def __init__(self, kind, fps, need_on_s, need_off_s):
        self.kind = kind
        self.fps = fps

        self.need_on = max(1, need_on_s * fps)
        self.need_off = need_off_s * fps

        self.on = 0
        self.off = 0

        self.flag = False
        self.start = 0
        self.peak = 0.0

        self.open = None
        self.events = []
        self.flag_frames = 0

    def update(self, active, idx, value=0.0):
        if active:
            if self.on == 0:
                self.start = idx
                self.peak = 0.0

            self.on += 1
            self.off = 0

            self.peak = max(self.peak, value)

        else:
            self.off += 1

            if self.off >= self.need_off:
                self.on = 0

        was_flagged = self.flag

        self.flag = self.on >= self.need_on

        if self.flag and not was_flagged:
            self.open = self.start

        if was_flagged and not self.flag:
            self._close(
                (idx - self.off) / self.fps
            )

        if self.flag:
            self.flag_frames += 1

        return self.flag

    def _close(self, end_s):
        self.events.append([
            self.kind,
            round(self.open / self.fps, 2),
            round(end_s, 2),
            round(self.peak, 2),
        ])

        self.open = None

    def finish(self, total_frames):
        if self.open is not None:
            self._close(
                total_frames / self.fps
            )

    @property
    def seconds(self):
        return self.on / self.fps


def link_phones(phones, faces, base):
    """ربط كل موبايل بأقرب وجه، بوحدات ارتفاع الوجه."""
    links = {}
    linked_ids = []

    for x1, y1, x2, y2, score in phones:
        phone_center_x = (x1 + x2) / 2
        phone_center_y = (y1 + y2) / 2

        best = None
        best_distance = PHONE_NEAR

        for detection in faces:
            tid = detection["id"]

            if tid not in base:
                continue

            distance = math.hypot(
                phone_center_x - detection["center"][0],
                phone_center_y - detection["center"][1],
            ) / max(detection["fh"], 1.0)

            if distance < best_distance:
                best = tid
                best_distance = distance

        linked_ids.append(best)

        if best is not None:
            links[best] = max(
                links.get(best, 0.0),
                score,
            )

    return links, linked_ids


def ema(previous, new):
    if previous is None:
        return new

    return (
        SMOOTH * previous
        + (1 - SMOOTH) * new
    )


def decide(frames, base, fps):
    states = {}
    rows = []
    phone_rows = []

    for idx, frame_data in enumerate(frames):
        links, linked_ids = link_phones(
            frame_data["phones"],
            frame_data["faces"],
            base,
        )

        phone_rows.append(
            list(zip(
                frame_data["phones"],
                linked_ids,
            ))
        )

        row = []

        for detection in frame_data["faces"]:
            tid = detection["id"]

            if tid not in base:
                continue

            state = states.get(tid)

            if state is None:
                state = {
                    "yaw": None,
                    "pitch": None,
                    "max_yaw": 0.0,
                    "warn_frames": 0,
                    "turn": Trigger(
                        "HEAD_TURN",
                        fps,
                        RED_SECONDS,
                        RECOVER_SECONDS,
                    ),
                    "phone": Trigger(
                        "PHONE",
                        fps,
                        PHONE_SECONDS,
                        PHONE_RECOVER,
                    ),
                }

                states[tid] = state

            base_yaw, base_pitch = base[tid]

            state["yaw"] = ema(
                state["yaw"],
                detection["yaw"] - base_yaw,
            )

            state["pitch"] = ema(
                state["pitch"],
                detection["pitch"] - base_pitch,
            )

            absolute_yaw = abs(state["yaw"])

            state["max_yaw"] = max(
                state["max_yaw"],
                absolute_yaw,
            )

            down = state["pitch"] > PITCH_DROP

            threshold = (
                RED_YAW_OFF
                if state["turn"].flag
                else RED_YAW
            )

            state["turn"].update(
                absolute_yaw > threshold and not down,
                idx,
                absolute_yaw,
            )

            warn = (
                absolute_yaw >= YELLOW_YAW
                and not down
                and not state["turn"].flag
            )

            if warn:
                state["warn_frames"] += 1

            state["phone"].update(
                tid in links,
                idx,
                links.get(tid, 0.0),
            )

            row.append({
                "id": tid,
                "pts": detection["pts"],
                "yaw": state["yaw"],
                "pitch": state["pitch"],
                "down": down,
                "warn": warn,
                "turn_flag": state["turn"].flag,
                "phone_t": state["phone"].seconds,
                "phone_flag": state["phone"].flag,
                "alarm_s": (
                    state["turn"].flag_frames / fps
                    + state["phone"].flag_frames / fps
                ),
            })

        rows.append(row)

    for state in states.values():
        state["turn"].finish(len(frames))
        state["phone"].finish(len(frames))

    return rows, phone_rows, states


# ---------------- الرسم ----------------

def draw_face(frame, detection):
    if detection["phone_flag"]:
        color = RED
        status = "SUSPICIOUS: PHONE"

    elif detection["turn_flag"]:
        color = RED
        status = "SUSPICIOUS: HEAD TURN"

    elif detection["warn"] or detection["phone_t"] > 0:
        color = YELLOW
        status = "WARNING"

    else:
        color = GREEN
        status = "FOCUSED"

    pts = detection["pts"]

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    face_width = max(int(x2 - x1), 1)
    radius = max(1, face_width // 110)

    # رسم معالم الوجه دفعة واحدة
    padding = radius + 2

    origin_x = max(int(x1) - padding, 0)
    origin_y = max(int(y1) - padding, 0)

    end_x = min(
        int(x2) + padding,
        frame.shape[1],
    )

    end_y = min(
        int(y2) + padding,
        frame.shape[0],
    )

    roi = frame[
        origin_y:end_y,
        origin_x:end_x,
    ]

    if roi.size:
        mask = np.zeros(
            roi.shape[:2],
            dtype=np.uint8,
        )

        xs = np.clip(
            pts[:, 0] - origin_x,
            0,
            roi.shape[1] - 1,
        )

        ys = np.clip(
            pts[:, 1] - origin_y,
            0,
            roi.shape[0] - 1,
        )

        mask[ys, xs] = 255

        if radius > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    2 * radius - 1,
                    2 * radius - 1,
                ),
            )

            mask = cv2.dilate(
                mask,
                kernel,
            )

        roi[mask > 0] = color

    for landmark_id in (
        NOSE,
        LEFT_EDGE,
        RIGHT_EDGE,
    ):
        point = tuple(
            int(value)
            for value in pts[landmark_id]
        )

        cv2.circle(
            frame,
            point,
            radius + 3,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            point,
            radius + 1,
            color,
            -1,
            cv2.LINE_AA,
        )

    label_x = int(x1)
    label_y = max(int(y1) - 12, 40)

    cv2.putText(
        frame,
        "ID%d %s" % (
            detection["id"],
            status,
        ),
        (label_x, label_y - 22),
        FONT,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    info = "yaw=%+.0f pitch=%+.0f%s" % (
        detection["yaw"],
        detection["pitch"],
        "  DOWN" if detection["down"] else "",
    )

    if detection["alarm_s"] > 0:
        info += "  alarm %.1fs" % detection["alarm_s"]

    cv2.putText(
        frame,
        info,
        (label_x, label_y - 6),
        FONT,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_phone(frame, box, tid):
    x1, y1, x2, y2, score = box

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        MAGENTA,
        2,
    )

    who = (
        "ID%d" % tid
        if tid is not None
        else "-"
    )

    cv2.putText(
        frame,
        "PHONE -> %s" % who,
        (
            int(x1),
            max(int(y1) - 6, 12),
        ),
        FONT,
        0.5,
        MAGENTA,
        1,
        cv2.LINE_AA,
    )


def render(path, rows, phone_rows, fps, w, h, out_path):
    cap = cv2.VideoCapture(path)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    idx = 0
    quit_all = False

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        row = (
            rows[idx]
            if idx < len(rows)
            else []
        )

        phone_row = (
            phone_rows[idx]
            if idx < len(phone_rows)
            else []
        )

        flagged = sum(
            1
            for detection in row
            if (
                detection["turn_flag"]
                or detection["phone_flag"]
            )
        )

        for detection in row:
            draw_face(frame, detection)

        for box, tid in phone_row:
            draw_phone(frame, box, tid)

        cv2.rectangle(
            frame,
            (0, 0),
            (400, 34),
            DARK,
            -1,
        )

        cv2.putText(
            frame,
            "Faces: %d  Flagged: %d  Phones: %d  %.1fs" % (
                len(row),
                flagged,
                len(phone_row),
                idx / fps,
            ),
            (10, 24),
            FONT,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if SHOW:
            cv2.imshow(
                "Smart Exam Monitoring",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_all = True
                break

        idx += 1

    cap.release()
    writer.release()

    return quit_all


# ---------------- معالجة فيديو وحفظ التقارير ----------------

def process_video(path, report, event_log):
    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    name = os.path.basename(path)

    out_path = os.path.join(
        OUT_DIR,
        "out_" + name,
    )

    frames, fps, w, h = analyze(path)

    base = compute_baselines(frames)

    rows, phone_rows, states = decide(
        frames,
        base,
        fps,
    )

    for tid, state in sorted(states.items()):
        turn = state["turn"]
        phone = state["phone"]

        print(
            "  ID%d: base_yaw=%+.0f base_pitch=%+.0f "
            "max_yaw_dev=%.0f "
            "| head: %d ev %.1fs "
            "| phone: %d ev %.1fs" % (
                tid,
                base[tid][0],
                base[tid][1],
                state["max_yaw"],
                len(turn.events),
                turn.flag_frames / fps,
                len(phone.events),
                phone.flag_frames / fps,
            )
        )

        report.append([
            name,
            tid,
            round(base[tid][0], 1),
            round(base[tid][1], 1),
            round(state["max_yaw"], 1),
            len(turn.events),
            round(turn.flag_frames / fps, 1),
            len(phone.events),
            round(phone.flag_frames / fps, 1),
        ])

        for event in turn.events + phone.events:
            event_log.append(
                [name, tid] + event
            )

    quit_all = render(
        path,
        rows,
        phone_rows,
        fps,
        w,
        h,
        out_path,
    )

    print("Saved:", out_path)

    return quit_all


def write_csv(path, header, data):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(header)
        writer.writerows(data)

    print("Report:", path)


# ---------------- التشغيل ----------------

if __name__ == "__main__":
    mproc.freeze_support()

    report = []
    event_log = []

    videos = sorted(
        glob.glob(VIDEOS_GLOB)
    )

    print(
        "step5: found %d videos in %s" % (
            len(videos),
            VIDEOS_GLOB,
        ),
        flush=True,
    )

    for video_path in videos:
        print(
            "Processing:",
            video_path,
            flush=True,
        )

        if process_video(
            video_path,
            report,
            event_log,
        ):
            break

    cv2.destroyAllWindows()

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    write_csv(
        os.path.join(
            OUT_DIR,
            "report.csv",
        ),
        [
            "video",
            "student_id",
            "base_yaw",
            "base_pitch",
            "max_yaw_dev",
            "head_events",
            "head_seconds",
            "phone_events",
            "phone_seconds",
        ],
        report,
    )

    event_log.sort(
        key=lambda event: (
            event[0],
            event[3],
        )
    )

    write_csv(
        os.path.join(
            OUT_DIR,
            "events.csv",
        ),
        [
            "video",
            "student_id",
            "type",
            "start_s",
            "end_s",
            "peak",
        ],
        event_log,
    )
