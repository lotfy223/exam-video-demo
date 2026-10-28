import csv
import hashlib
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st


ROOT = Path(__file__).resolve().parent
MAX_SECONDS = 30
MAX_MB = 20

st.set_page_config(
    page_title="ExamLens",
    page_icon="◉",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
        color: #e2e8f0;
    }

    .block-container {
        max-width: 960px;
        padding-top: 3rem;
        padding-bottom: 2rem;
        direction: ltr;
        text-align: left;
    }

    .brand {
        color: #38bdf8;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 3px;
        margin-bottom: 12px;
    }

    .heading {
        color: #f8fafc;
        font-size: clamp(30px, 5vw, 42px);
        font-weight: 700;
        letter-spacing: -1.5px;
        line-height: 1.2;
        margin-bottom: 10px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 30px;
    }

    [data-testid="stFileUploader"] {
        background: #111b2e;
        border: 1px solid #25334b;
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stMetric"] {
        background: #111b2e;
        border: 1px solid #25334b;
        border-radius: 14px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px;
        min-height: 44px;
    }

    div.stButton > button[kind="primary"] {
        background: #0284c7;
        border: 1px solid #0284c7;
        color: white;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #0369a1;
        border-color: #0369a1;
    }

    .footer {
        margin-top: 28px;
        padding-top: 16px;
        border-top: 1px solid #25334b;
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand">EXAMLENS</div>
    <div class="heading">Video review, simplified.</div>
    <div class="subtitle">
        Phone detection & head movement analysis.
    </div>
    """,
    unsafe_allow_html=True,
)


RUNNER = """
import sys
import multiprocessing as mproc
from pathlib import Path
import step5_detector as detector

if __name__ == "__main__":
    mproc.freeze_support()

    source = sys.argv[1]
    output = Path(sys.argv[2])

    detector.SHOW = False
    detector.OUT_DIR = str(output)

    report = []
    events = []

    detector.process_video(source, report, events)

    detector.write_csv(
        str(output / "report.csv"),
        [
            "video", "student_id", "base_yaw", "base_pitch",
            "max_yaw_dev", "head_events", "head_seconds",
            "phone_events", "phone_seconds"
        ],
        report,
    )

    events.sort(key=lambda event: (event[0], event[3]))

    detector.write_csv(
        str(output / "events.csv"),
        [
            "video", "student_id", "type",
            "start_s", "end_s", "peak"
        ],
        events,
    )
"""


def analyze_video(video_bytes):
    required = [
        ROOT / "step5_detector.py",
        ROOT / "models" / "face_landmarker.task",
        ROOT / "models" / "efficientdet_lite0.tflite",
    ]

    missing = [
        str(path.relative_to(ROOT))
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]

    if missing:
        raise RuntimeError(
            "Missing or empty files: " + ", ".join(missing)
        )

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is unavailable. Add ffmpeg to packages.txt."
        )

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        source = work / "input.mp4"
        output = work / "results"
        output.mkdir()

        source.write_bytes(video_bytes)

        cap = cv2.VideoCapture(str(source))

        try:
            opened = cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            readable, first_frame = cap.read()
        finally:
            cap.release()

        if (
            not opened
            or not readable
            or fps <= 0
            or count <= 0
        ):
            raise ValueError(
                "Unable to read this video. Please try another MP4."
            )

        duration = count / fps

        if duration > MAX_SECONDS:
            raise ValueError(
                f"Video duration is {duration:.1f}s. "
                f"Maximum allowed: {MAX_SECONDS}s."
            )

        height, width = first_frame.shape[:2]

        if max(width, height) > 1920 or fps > 60:
            raise ValueError(
                "Please use a video up to 1920px "
                "on its longest side and up to 60 FPS."
            )

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                RUNNER,
                str(source),
                str(output),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=180,
        )

        log = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(log[-16000:])

        raw_video = output / "out_input.mp4"
        report = output / "report.csv"
        events = output / "events.csv"

        for path in (raw_video, report, events):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    "Processing did not produce all required files."
                )

        browser_video = output / "result.mp4"

        conversion = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i", str(raw_video),
                "-map", "0:v:0",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags", "+faststart",
                "-an",
                str(browser_video),
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )

        if (
            conversion.returncode != 0
            or not browser_video.is_file()
            or browser_video.stat().st_size == 0
        ):
            raise RuntimeError(
                "Video conversion failed.\n"
                + conversion.stderr[-6000:]
            )

        phone_unavailable = any(
            message in log
            for message in [
                "phone detection unavailable",
                "phone detection disabled",
                "phone detector timed out",
            ]
        )

        return {
            "video": browser_video.read_bytes(),
            "report": report.read_bytes(),
            "events": events.read_bytes(),
            "duration": duration,
            "phone_unavailable": phone_unavailable,
            "log": log,
        }


uploaded = st.file_uploader(
    "Upload video",
    type=["mp4"],
    help=f"MP4 · Up to {MAX_SECONDS} seconds · {MAX_MB} MB",
)

st.caption(f"MP4 · {MAX_SECONDS}s max · {MAX_MB} MB max")

video_bytes = uploaded.getvalue() if uploaded else None

file_id = (
    hashlib.sha256(video_bytes).hexdigest()
    if video_bytes is not None
    else None
)

if st.session_state.get("file_id") != file_id:
    st.session_state.pop("results", None)
    st.session_state["file_id"] = file_id

consent = st.checkbox(
    "I have permission to upload and process this video."
)

if uploaded:
    with st.expander("Preview original"):
        st.video(video_bytes)

start = st.button(
    "Analyze Video",
    type="primary",
    use_container_width=True,
    disabled=uploaded is None or not consent,
)

if start:
    st.session_state.pop("results", None)

    if len(video_bytes) > MAX_MB * 1024 * 1024:
        st.error(f"File size exceeds {MAX_MB} MB.")
    else:
        try:
            with st.spinner("Analyzing video…"):
                st.session_state["results"] = analyze_video(
                    video_bytes
                )

        except ValueError as error:
            st.error(str(error))

        except subprocess.TimeoutExpired:
            st.error(
                "Processing timed out. Please try a shorter video."
            )

        except Exception as error:
            st.error("Processing failed.")
            with st.expander("Technical details"):
                st.code(str(error))


if "results" in st.session_state:
    data = st.session_state["results"]

    event_rows = list(
        csv.DictReader(
            io.StringIO(
                data["events"].decode("utf-8-sig")
            )
        )
    )

    head_events = sum(
        row.get("type") == "HEAD_TURN"
        for row in event_rows
    )

    phone_events = sum(
        row.get("type") == "PHONE"
        for row in event_rows
    )

    st.divider()
    st.subheader("Results")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Duration",
        f'{data["duration"]:.1f}s',
    )

    metric2.metric(
        "Head-turn events",
        head_events,
    )

    metric3.metric(
        "Phone events",
        "Unavailable" if data["phone_unavailable"] else phone_events,
    )

    st.write("")

    if data["phone_unavailable"]:
        st.warning(
            "Phone detection was incomplete. "
            "Phone counts are not reliable for this run."
        )

    st.video(
        data["video"],
        format="video/mp4",
    )

    st.caption(
        "Event counts are detection intervals, not unique objects. "
        "Output video has no audio."
    )

    download1, download2, download3 = st.columns(3)

    with download1:
        st.download_button(
            "Download Video",
            data=data["video"],
            file_name="examlens_result.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    with download2:
        st.download_button(
            "Summary CSV",
            data=data["report"],
            file_name="summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download3:
        st.download_button(
            "Events CSV",
            data=data["events"],
            file_name="events.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if event_rows:
        with st.expander("Event details"):
            st.dataframe(
                event_rows,
                hide_index=True,
                use_container_width=True,
            )

    with st.expander("Processing log"):
        st.code(data["log"])

    if st.button("Clear Results"):
        st.session_state.pop("results", None)
        st.rerun()


st.markdown(
    """
    <div class="footer">
        Experimental tool. Human review required;
        detections do not establish cheating or attention.
        <br>
        Videos are processed on the server.
        Temporary working files are deleted after processing;
        results remain in session memory until cleared or the session ends.
    </div>
    """,
    unsafe_allow_html=True,
)
