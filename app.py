import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
from ultralytics import YOLO

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Moving Object Detection Dashboard",
    page_icon="🎥",
    layout="wide",
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


def load_demo_dataframe():
    demo_csv_path = os.path.join(os.path.dirname(__file__), "detections.csv")
    if not os.path.exists(demo_csv_path):
        return pd.DataFrame(columns=["Frame", "Confidence", "Class", "Object"])

    df = pd.read_csv(demo_csv_path)
    df["Object"] = df["Class"].map(CLASS_NAMES).fillna("Other")
    return df


def build_detection_dataframe(results):
    rows = []

    for frame_idx, result in enumerate(results, start=1):
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            rows.append(
                {
                    "Frame": frame_idx,
                    "Confidence": round(float(conf), 4),
                    "Class": int(cls_id),
                    "Object": CLASS_NAMES.get(int(cls_id), "Other"),
                }
            )

    if rows:
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["Frame", "Confidence", "Class", "Object"])


def find_processed_video_path(save_dir, source_path):
    if not save_dir or not os.path.isdir(save_dir):
        return None

    source_name = os.path.splitext(os.path.basename(source_path))[0].lower()
    candidates = []

    for entry in os.listdir(save_dir):
        full_path = os.path.join(save_dir, entry)
        if os.path.isfile(full_path) and entry.lower().endswith((".mp4", ".avi", ".mov")):
            candidates.append(full_path)

    if not candidates:
        return None

    for candidate in candidates:
        candidate_name = os.path.splitext(os.path.basename(candidate))[0].lower()
        if source_name in candidate_name:
            return candidate

    return candidates[0]


def run_yolo(video_path, confidence):
    results = model.track(
        source=video_path,
        save=True,
        persist=True,
        conf=confidence,
    )

    save_dir = results[0].save_dir if results else None
    processed_video_path = find_processed_video_path(save_dir, video_path)
    detection_df = build_detection_dataframe(results)

    return processed_video_path, detection_df


def render_dashboard(df):
    if df.empty:
        st.info("No detections available yet.")
        return

    total = len(df)
    frames = int(df["Frame"].nunique()) if "Frame" in df.columns else 0
    avg_conf = round(float(df["Confidence"].mean()) * 100, 2) if "Confidence" in df.columns else 0.0
    classes = int(df["Object"].nunique()) if "Object" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Objects Detected", total)
    c2.metric("🎬 Frames Processed", frames)
    c3.metric("📦 Object Classes", classes)
    c4.metric("⭐ Avg Confidence", f"{avg_conf}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        objects = df.groupby("Frame").size().reset_index(name="Objects")
        fig = px.line(
            objects,
            x="Frame",
            y="Objects",
            title="Objects Detected Per Frame",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.pie(df, names="Object", title="Object Distribution", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.histogram(df, x="Confidence", nbins=25, title="Confidence Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top = (
            df.groupby("Frame")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="Objects")
        )
        fig = px.bar(
            top,
            x="Frame",
            y="Objects",
            color="Objects",
            text="Objects",
            title="Top 10 Busiest Frames",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📌 Detection Summary")
    left, right = st.columns([2, 1])

    with left:
        st.dataframe(df.head(20), use_container_width=True)

    with right:
        st.info(
            f"""
### Project Details

**Model**
YOLOv8

**Dataset**
MOT17 / Uploaded Video

**Framework**
Ultralytics

**Language**
Python

**Detections**
{total}

**Frames**
{frames}
"""
        )


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown(
    """
<h1 style='text-align:center;'>🎥 Moving Object Detection Dashboard</h1>
<h4 style='text-align:center;color:gray;'>
YOLOv8 • MOT17 Dataset • Computer Vision Project
</h4>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

st.subheader("🎥 Demo Detection")
st.write("This is a YOLOv8 object detection and tracking demo using a preprocessed sample video with bounding boxes.")

demo_video_path = os.path.join(os.path.dirname(__file__), "15781298_1920_1080_60fps.avi")
if os.path.exists(demo_video_path):
    st.video(demo_video_path)
else:
    st.warning("Demo video is not available in the project folder yet.")

demo_df = load_demo_dataframe()
render_dashboard(demo_df)

st.markdown("---")

st.subheader("📤 Upload Your Own Video")

uploaded_video = st.file_uploader("Choose a Video", type=["mp4", "avi", "mov"])
run = st.button("🚀 Run Detection")
confidence = st.slider("Confidence Threshold", 0.10, 1.00, 0.30, 0.05)

uploaded_df = pd.DataFrame(columns=["Frame", "Confidence", "Class", "Object"])
processed_video_path = None

if uploaded_video is not None and run:
    extension = os.path.splitext(uploaded_video.name)[1] or ".mp4"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp_file.write(uploaded_video.read())
    temp_file.close()

    temp_path = temp_file.name
    st.success("Video uploaded successfully!")

    with st.spinner("Running YOLO Detection..."):
        try:
            processed_video_path, uploaded_df = run_yolo(temp_path, confidence)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            uploaded_df = pd.DataFrame(columns=["Frame", "Confidence", "Class", "Object"])
            processed_video_path = None

    st.success("✅ Detection Completed!")

    if processed_video_path and os.path.exists(processed_video_path):
        st.subheader("🎥 Detection Result")
        st.video(processed_video_path)

        with open(processed_video_path, "rb") as video_file:
            video_bytes = video_file.read()

        st.download_button(
            "📥 Download Processed Video",
            data=video_bytes,
            file_name="detected_video.mp4",
            mime="video/mp4",
        )

        render_dashboard(uploaded_df)
    else:
        st.error("Processed video not found.")
        render_dashboard(uploaded_df)
else:
    st.info("Upload a video and click Run Detection to generate fresh YOLOv8 results.")

# -------------------------------------------------------
# DOWNLOAD
# -------------------------------------------------------

download_df = uploaded_df if not uploaded_df.empty else demo_df
csv = download_df.to_csv(index=False)

st.download_button(
    "📥 Download Detection Results",
    csv,
    "detections.csv",
    "text/csv",
)

st.markdown("---")

st.caption("Developed using Python • YOLOv8 • OpenCV • Plotly • Streamlit")