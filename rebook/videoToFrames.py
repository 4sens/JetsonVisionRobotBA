import cv2
from pathlib import Path

# -------- CONFIG --------
VIDEO_DIR = Path(r"C:\Users\noovelUser\Documents\YOLO\rebook\recordings")
OUTPUT_DIR = Path(r"C:\Users\noovelUser\Documents\YOLO\rebook\frames4")

FPS_TARGET = 3           # frames per second
PREFIX = "frame2_"
START_INDEX = 1
# ------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

idx = START_INDEX

for video_path in VIDEO_DIR.glob("*.mp4"):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback

    frame_interval = int(round(fps / FPS_TARGET))
    frame_count = 0

    print(f"Processing {video_path.name} @ {fps:.2f} FPS → every {frame_interval} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out_path = OUTPUT_DIR / f"{PREFIX}{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            idx += 1

        frame_count += 1

    cap.release()

print(f"\n✅ Done. Extracted images: {idx - START_INDEX}")