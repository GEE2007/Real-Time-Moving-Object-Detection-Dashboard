import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Moving Object Detection Dashboard",
    page_icon="🎥",
    layout="wide"
)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

df = pd.read_csv("detections.csv")

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

df["Object"] = df["Class"].map(CLASS_NAMES).fillna("Other")

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
unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------

total = len(df)
frames = df["Frame"].nunique()
avg_conf = round(df["Confidence"].mean()*100,2)
classes = df["Object"].nunique()

c1,c2,c3,c4 = st.columns(4)

c1.metric("🎯 Objects Detected", total)
c2.metric("🎬 Frames Processed", frames)
c3.metric("📦 Object Classes", classes)
c4.metric("⭐ Avg Confidence", f"{avg_conf}%")

st.markdown("---")

# -------------------------------------------------------
# VIDEO
# -------------------------------------------------------
st.subheader("🎥 Processed Detection Video")

st.info(
    "The processed detection video has been successfully generated using the YOLOv8 object detection model. "
    "Click the button below to download and view the output locally."
)

with open("mot17_video.mp4", "rb") as video_file:
    st.download_button(
        label="📥 Download Processed Video",
        data=video_file,
        file_name="mot17_video.mp4",
        mime="video/mp4"
    )

st.markdown("---")

# -------------------------------------------------------
# CHARTS
# -------------------------------------------------------

left,right = st.columns(2)

with left:

    objects = (
        df.groupby("Frame")
        .size()
        .reset_index(name="Objects")
    )

    fig = px.line(
        objects,
        x="Frame",
        y="Objects",
        title="Objects Detected Per Frame",
        markers=True
    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    fig = px.pie(
        df,
        names="Object",
        title="Object Distribution",
        hole=0.45
    )

    st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------------

left,right = st.columns(2)

with left:

    fig = px.histogram(
        df,
        x="Confidence",
        nbins=25,
        title="Confidence Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

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
        title="Top 10 Busiest Frames"
    )

    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

# -------------------------------------------------------
# QUICK SUMMARY
# -------------------------------------------------------

st.subheader("📌 Detection Summary")

left,right = st.columns([2,1])

with left:

    st.dataframe(df.head(20),use_container_width=True)

with right:

    st.info(f"""
### Project Details

**Model**
YOLOv8

**Dataset**
MOT17

**Framework**
Ultralytics

**Language**
Python

**Detections**
{total}

**Frames**
{frames}
""")

st.markdown("---")

# -------------------------------------------------------
# DOWNLOAD
# -------------------------------------------------------

csv = df.to_csv(index=False)

st.download_button(
    "📥 Download Detection Results",
    csv,
    "detections.csv",
    "text/csv"
)

st.markdown("---")

st.caption("Developed using Python • YOLOv8 • OpenCV • Plotly • Streamlit")