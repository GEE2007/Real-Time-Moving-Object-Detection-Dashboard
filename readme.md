# 🎥 Moving Object Detection Dashboard

An interactive Streamlit dashboard for video-based object detection and multi-object tracking using YOLOv8. Users can upload videos, detect objects frame by frame, explore analytics, and download processed results.

🌐 **Live Demo:** https://moving-object-detection-g.streamlit.app/

---

## Features

- Object detection using YOLOv8
- Multi-object tracking with persistent Track IDs
- Upload custom videos (MP4, AVI, MOV)
- Built-in demo video
- Interactive analytics dashboard
- Download processed video
- Export detection results as CSV

---

## Dashboard

**Summary Metrics**

- Total objects detected
- Frames processed
- Object classes detected
- Average detection confidence

**Visualizations**

- Objects detected per frame
- Object class distribution
- Confidence distribution
- Top 10 busiest frames
- Detection summary table

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python |
| Framework | Streamlit |
| Object Detection | Ultralytics YOLOv8 |
| Video Processing | OpenCV |
| Data Analysis | Pandas |
| Visualization | Plotly |

---

## Project Structure

```text
moving-object-detection/
│
├── app.py
├── convert.py
├── detections.csv
├── demo.mp4
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/GEE2007/moving-object-detection.git

cd moving-object-detection

pip install -r requirements.txt

streamlit run app.py
```

Or try the live demo:

https://moving-object-detection-g.streamlit.app/

---

## Notes

- Uses YOLOv8's built-in tracking for assigning persistent object IDs.
- Supports MP4, AVI, and MOV videos.
- Detection analytics are generated automatically after processing.
- Processed videos and detection results can be downloaded from the dashboard.

---

## Future Improvements

- Live webcam detection
- Support for additional YOLO models
- Object filtering by class
- Performance optimization for larger videos

---

## Author

**Geetika Bhardwaj**

GitHub: https://github.com/GEE2007