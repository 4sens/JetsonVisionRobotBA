import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

DATASET_ROOT = Path(r"C:\Users\noovelUser\Documents\YOLO\rebook\dataset")
SPLIT = "val"  # train/val/test

IMAGES_DIR = DATASET_ROOT / "images" / SPLIT
LABELS_DIR = DATASET_ROOT / "labels" / SPLIT

WIN = "GT Label Viewer (EXIF+Aspect OK)"
WIN_W, WIN_H = 1400, 900

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_yolo_labels(label_path: Path):
    if not label_path.exists():
        return []
    out = []
    for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        out.append((cls_id, x, y, w, h))
    return out

def yolo_xywhn_to_xyxy(x, y, w, h, W, H):
    x1 = int((x - w/2) * W)
    y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W)
    y2 = int((y + h/2) * H)
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
    return x1, y1, x2, y2

def load_image_exif_ok(path: Path):
    # PIL reads EXIF orientation; exif_transpose "bakes" it into pixels
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    arr = np.array(im)  # RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def draw_gt(frame, labels):
    H, W = frame.shape[:2]
    for cls_id, x, y, w, h in labels:
        x1, y1, x2, y2 = yolo_xywhn_to_xyxy(x, y, w, h, W, H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"GT:{cls_id}", (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def letterbox_to_window(img, win_w, win_h):
    H, W = img.shape[:2]
    scale = min(win_w / W, win_h / H)
    new_w, new_h = int(W * scale), int(H * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x0 = (win_w - new_w) // 2
    y0 = (win_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def main():
    images = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WIN_W, WIN_H)

    i = 0
    while True:
        img_path = images[i]
        frame = load_image_exif_ok(img_path)

        label_path = LABELS_DIR / (img_path.stem + ".txt")
        labels = read_yolo_labels(label_path)

        vis = draw_gt(frame.copy(), labels)
        cv2.putText(vis, f"{SPLIT} {i+1}/{len(images)} | {img_path.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        show = letterbox_to_window(vis, WIN_W, WIN_H)
        cv2.imshow(WIN, show)

        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("n"), 32, 13):  # next
            i = (i + 1) % len(images)
        elif k == ord("p"):          # prev
            i = (i - 1) % len(images)
        else:
            i = (i + 1) % len(images)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
