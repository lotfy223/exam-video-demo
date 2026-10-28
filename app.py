import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="تجربة تحليل الفيديو",
    page_icon="🎥",
)

st.title("🎥 تجربة رصد الموبايل وحركة الرأس")

st.write(
    "ارفع فيديو قصير، وبعد التحليل حمّل الفيديو الناتج والتقارير."
)

st.warning(
    "مشروع تجريبي: وجود موبايل أو التفات الرأس مش دليل على الغش. "
    "النتائج محتاجة مراجعة بشرية."
)

st.info(
    "الحد التجريبي: فيديو مدته ٢٠ ثانية وحجمه ٢٠ ميجابايت. "
    "ارفع فقط فيديو عندك إذن باستخدامه، ويفضل من غير بيانات شخصية."
)

uploaded = st.file_uploader(
    "اختار فيديو بصيغة MP4",
    type=["mp4"],
)

consent = st.checkbox(
    "عندي إذن باستخدام الفيديو ومعالجته على السيرفر."
)

# التحليل في عملية مستقلة، بملفات مؤقتة خاصة بكل تجربة.
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

if st.button(
    "تحليل الفيديو",
    disabled=uploaded is None or not consent,
):
    st.session_state.pop("results", None)

    if uploaded.size > 20 * 1024 * 1024:
        st.error("الفيديو أكبر من ٢٠ ميجابايت.")
        st.stop()

    required_files = [
        ROOT / "step5_detector.py",
        ROOT / "models" / "face_landmarker.task",
        ROOT / "models" / "efficientdet_lite0.tflite",
    ]

    missing = [
        str(path.relative_to(ROOT))
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        st.error("ملفات ناقصة: " + ", ".join(missing))
        st.stop()

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        source = work / "input.mp4"
        output = work / "results"
        output.mkdir()

        source.write_bytes(uploaded.getvalue())

        cap = cv2.VideoCapture(str(source))

        try:
            readable = cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            ok, _ = cap.read()
        finally:
            cap.release()

        if not readable or not ok or fps <= 0 or count <= 0:
            st.error("مش قادر أقرأ الفيديو. جرّب ملف MP4 تاني.")
            st.stop()

        duration = count / fps

        if duration > 20:
            st.error(
                f"مدة الفيديو {duration:.1f} ثانية. "
                "اختار مقطع مدته ٢٠ ثانية أو أقل."
            )
            st.stop()

        try:
            with st.spinner("جاري التحليل، استنى لحد ما يخلص…"):
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

        except subprocess.TimeoutExpired:
            st.error(
                "التحليل تجاوز مهلة ٣ دقائق. جرّب فيديو أقصر."
            )
            st.stop()

        if result.returncode != 0:
            st.error("حصل خطأ أثناء التحليل.")
            with st.expander("تفاصيل الخطأ"):
                st.code(result.stdout + "\n" + result.stderr)
            st.stop()

        video = output / "out_input.mp4"
        report = output / "report.csv"
        events = output / "events.csv"

        if not all(path.is_file() for path in [video, report, events]):
            st.error("التحليل ما أنتجش كل الملفات المطلوبة.")
            st.stop()

        st.session_state["results"] = {
            "video": video.read_bytes(),
            "report": report.read_bytes(),
            "events": events.read_bytes(),
            "log": result.stdout + "\n" + result.stderr,
        }

# الملفات المؤقتة بتتمسح، ونسخة التحميل بتفضل في جلسة الصفحة.
if "results" in st.session_state:
    data = st.session_state["results"]

    st.success(
        "انتهى التشغيل. حمّل الفيديو وراجعه للتأكد من الكشف."
    )

    if (
        "phone detection unavailable" in data["log"]
        or "phone detection disabled" in data["log"]
    ):
        st.warning(
            "كشف الموبايل ما اشتغلش في التجربة دي؛ "
            "ده مش معناه إن الفيديو مفيهوش موبايلات."
        )

    st.download_button(
        "تحميل الفيديو الناتج",
        data=data["video"],
        file_name="result.mp4",
        mime="video/mp4",
    )

    st.download_button(
        "تحميل ملخص النتائج",
        data=data["report"],
        file_name="report.csv",
        mime="text/csv",
    )

    st.download_button(
        "تحميل سجل الأحداث",
        data=data["events"],
        file_name="events.csv",
        mime="text/csv",
    )

    with st.expander("سجل التشغيل"):
        st.code(data["log"])

    if st.button("مسح النتيجة من الجلسة"):
        del st.session_state["results"]
        st.rerun()
