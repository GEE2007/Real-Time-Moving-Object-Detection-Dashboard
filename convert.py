import cv2
import imageio

input_path = "mot17_video.avi"
output_path = "mot17_video.mp4"

# Open input video
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps != fps: # Handle invalid FPS
    fps = 30.0

# Initialize imageio FFmpeg writer with browser-safe arguments
writer = imageio.get_writer(
    output_path, 
    fps=fps, 
    codec="libx264", 
    pixelformat="yuv420p",
    ffmpeg_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] # Fix odd dimension issue for H.264
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert OpenCV BGR to RGB (Crucial for correct encoding)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    writer.append_data(frame_rgb)

cap.release()
writer.close()

print("Converted successfully with H.264 & yuv420p!")