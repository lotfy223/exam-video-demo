# ExamLens

A video review tool that helps a human proctor find moments when a student turns their head away during an exam.

**Live demo:** [Try ExamLens on Streamlit](https://exam-video-demo-huxph9rvf9jnyft2auxwhn.streamlit.app/)

## What it does
Upload a short exam video. ExamLens tracks each face, measures head rotation, and returns an annotated video with a label on each student, so a reviewer knows where to look instead of watching every second.

## How it works
- **Face tracking:** MediaPipe Face Landmarker finds each face and estimates head yaw and pitch.
- **Per-student baseline:** each student's own neutral head position is used as the reference, instead of one fixed angle for everyone.
- **Time thresholds:** an alert starts only after a head turn lasts long enough, and ends only after recovery, so labels do not flicker.
- **Video output:** the annotated video is converted with FFmpeg (H.264) so it plays in the browser.
- **Web app:** a Streamlit interface for upload, preview, processing and download.

## Demo limits
- Format: MP4
- Length: up to 30 seconds
- Size: up to 20 MB
- Resolution: up to 1920 px on the longest side, up to 60 FPS
- The output video has no audio.

## Run locally
```bash
git clone https://github.com/lotfy223/exam-video-demo.git
cd exam-video-demo
pip install -r requirements.txt
streamlit run app.py
```
FFmpeg must be installed. On Streamlit Cloud, add `ffmpeg` to `packages.txt`.

The model file `models/face_landmarker.task` must be present in the repo.

## Limitations
- This is a learning prototype, **not an automated cheating detector**.
- Head movement does not prove cheating or lack of attention. A human must always review the result.
- Camera angle, blocked faces and missed faces can change the results.
- Accuracy has not been measured on a labeled evaluation set yet.
- Videos are processed on the server. Temporary files are deleted after processing, and results stay in session memory until cleared or the session ends.
- Only upload videos you have permission to process.

## Next steps
- Test on labeled videos and measure false alerts
- Improve handling of crowded scenes and partly hidden faces
- Collect feedback on the user experience

## Author
Ahmed Lotfy

Built as part of my training at Instant Software Solutions. Thanks to Jana Hatem and Rawan Mahmoud for their guidance and feedback.
