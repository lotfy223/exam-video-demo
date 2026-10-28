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
MAX_MB = 20
MAX_SECONDS = 20

st.set_page_config(
    page_title="مراجعة الفيديو",
    page_icon="🎥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        direction: rtl;
        text-align: right;
        padding: 26px;
        border: 1px solid #334155;
        border-radius: 20px;
        background: linear-gradient(120deg, #0f172a, #164e63);
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 10px;
        font-size: 2rem;
    }
    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎥 مراجعة الفيديو بالذكاء الاصطناعي</h1>
        <p>
            رصد الموبايلات وحركة الرأس،
            مع عرض الفيديو الناتج وتحميل تقارير المراجعة.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "نسخة تجريبية للمراجعة البشرية فقط. "
    "وجود موبايل أو التفات الرأس مش دليل على الغش، "
    "ولا يثبت التركيز أو عدمه."
)

with st.expander("حدود التجربة والخصوصية"):
    st.write(
        "• أقصى مدة: ٢٠ ثانية.\n\n"
        "• أقصى حجم: ٢٠ ميجابايت.\n\n"
        "• المعالجة بتحصل على السيرفر، مش على جهازك فقط.\n\n"
        "• ارفع فقط فيديو عندك إذن باستخدامه ومعالجته.\n\n"
        "• التطبيق لا يضيف الفيديوهات لبيانات تدريب. "
        "ملفات العمل المؤقتة تُحذف عند انتهاء المحاولة، "
        "والنتيجة تُحفظ في ذاكرة جلسة الصفحة لعرضها وتحميلها.\n\n"
        "• الرصد ممكن يخطئ أو يفوّت موبايل، "
        "خصوصًا قرب الوجه أو مع الحجب."
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


def run_analysis(video_bytes):
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
            "ملفات ناقصة أو فاضية: " + ", ".join(missing)
        )

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "أداة ffmpeg مش متثبتة. "
            "أضف ffmpeg لملف packages.txt "
            "واستنى إعادة تجهيز الموقع."
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
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            readable, first_frame = cap.read()
        finally:
            cap.release()

        if (
            not opened
            or not readable
            or fps <= 0
            or frame_count <= 0
        ):
            raise RuntimeError(
                "مش قادر أقرأ الفيديو. جرّب ملف MP4 تاني."
            )

        duration = frame_count / fps

        if duration > MAX_SECONDS:
            raise RuntimeError(
                f"مدة الفيديو {duration:.1f} ثانية. "
                "اختار مقطع مدته ٢٠ ثانية أو أقل."
            )

        # حد إضافي لتقليل استهلاك ذاكرة السيرفر.
        height, width = first_frame.shape[:2]

        if width > 1920 or height > 1920 or fps > 60:
            raise RuntimeError(
                "الفيديو كبير على النسخة التجريبية. "
                "استخدم دقة 1080p أو أقل ومعدل ٦٠ فريم أو أقل."
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
                    "التحليل ما أنتجش كل الملفات المطلوبة."
                )

        # تحويل الفيديو لصيغة مناسبة للتشغيل في المتصفح.
        browser_video = output / "result.mp4"

        converted = subprocess.run(
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

        if converted.returncode != 0:
            raise RuntimeError(
                "التحليل خلص، لكن تحويل الفيديو للعرض فشل:\n"
                + converted.stderr[-6000:]
            )

        return {
            "video": browser_video.read_bytes(),
            "report": report.read_bytes(),
            "events": events.read_bytes(),
            "duration": duration,
            "log": log,
        }


st.subheader("١. اختار الفيديو")

uploaded = st.file_uploader(
    "ارفع فيديو بصيغة MP4",
    type=["mp4"],
)

# مسح النتيجة القديمة عند تغيير الملف حتى لا تختلط النتائج.
video_bytes = uploaded.getvalue() if uploaded else None
file_id = (
    hashlib.sha256(video_bytes).hexdigest()
    if video_bytes is not None
    else None
)

if st.session_state.get("file_id") != file_id:
    st.session_state.pop("results", None)
    st.session_state["file_id"] = file_id

if uploaded:
    st.caption(
        f"الملف: {uploaded.name} — "
        f"الحجم: {uploaded.size / (1024 * 1024):.1f} ميجابايت"
    )

    with st.expander("معاينة الفيديو الأصلي"):
        st.video(video_bytes)

consent = st.checkbox(
    "عندي إذن باستخدام الفيديو ومعالجته على السيرفر."
)

start = st.button(
    "▶ ابدأ تحليل الفيديو",
    type="primary",
    disabled=uploaded is None or not consent,
)

if start:
    st.session_state.pop("results", None)

    if len(video_bytes) > MAX_MB * 1024 * 1024:
        st.error("حجم الفيديو أكبر من ٢٠ ميجابايت.")
    else:
        try:
            with st.spinner(
                "جاري التحليل وتجهيز الفيديو للعرض…"
            ):
                st.session_state["results"] = run_analysis(
                    video_bytes
                )

        except subprocess.TimeoutExpired:
            st.error(
                "المعالجة تجاوزت المهلة. جرّب مقطع أقصر."
            )

        except Exception as error:
            st.error("حصل خطأ أثناء المعالجة.")
            with st.expander("تفاصيل الخطأ", expanded=True):
                st.code(str(error))


if "results" in st.session_state:
    data = st.session_state["results"]

    st.divider()
    st.subheader("٢. نتيجة التحليل")
    st.success(
        "انتهى التشغيل. شغّل الفيديو هنا وراجع العلامات."
    )

    phone_unavailable = any(
        message in data["log"]
        for message in [
            "phone detection unavailable",
            "phone detection disabled",
            "phone detector timed out",
        ]
    )

    if phone_unavailable:
        st.warning(
            "كشف الموبايل ما اشتغلش بشكل كامل. "
            "غياب أحداث الموبايل هنا مش دليل على عدم وجوده."
        )

    event_text = data["events"].decode("utf-8-sig")
    event_rows = list(
        csv.DictReader(io.StringIO(event_text))
    )

    head_events = sum(
        row.get("type") == "HEAD_TURN"
        for row in event_rows
    )

    phone_events = sum(
        row.get("type") == "PHONE"
        for row in event_rows
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "مدة الفيديو",
        f'{data["duration"]:.1f} ثانية',
    )
    col2.metric("أحداث التفات الرأس", head_events)
    col3.metric(
        "أحداث الموبايل",
        "غير مكتمل" if phone_unavailable else phone_events,
    )

    st.caption(
        "الأرقام أحداث رصد متتابعة، "
        "مش عدد موبايلات فريدة ولا أحكام على الأشخاص."
    )

    # عرض الفيديو الناتج داخل الصفحة.
    st.video(data["video"], format="video/mp4")

    st.caption(
        "الفيديو الناتج من غير صوت. "
        "العبارات المرسومة داخله جاية من كود الكاشف، "
        "ومش إثبات للتركيز أو الغش."
    )

    st.subheader("تحميل النتائج")

    download1, download2, download3 = st.columns(3)

    with download1:
        st.download_button(
            "⬇ الفيديو الناتج",
            data=data["video"],
            file_name="result.mp4",
            mime="video/mp4",
        )

    with download2:
        st.download_button(
            "⬇ ملخص النتائج",
            data=data["report"],
            file_name="report.csv",
            mime="text/csv",
        )

    with download3:
        st.download_button(
            "⬇ سجل الأحداث",
            data=data["events"],
            file_name="events.csv",
            mime="text/csv",
        )

    with st.expander("تفاصيل الأحداث"):
        if event_rows:
            st.dataframe(
                event_rows,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info(
                "مفيش أحداث مسجّلة؛ "
                "ده مش ضمان إن كل الحالات اتكشفت."
            )

    with st.expander("سجل التشغيل"):
        st.code(data["log"])

    if st.button("مسح النتيجة من الجلسة"):
        st.session_state.pop("results", None)
        st.rerun()

st.divider()
st.caption(
    "مشروع تجريبي • النتائج للمراجعة البشرية • "
    "لا تستخدمه منفردًا لاتخاذ قرارات بحق الطلاب"
)
