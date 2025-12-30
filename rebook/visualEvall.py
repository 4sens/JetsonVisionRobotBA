import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
DATA_YAML = r"C:\Users\noovelUser\Documents\YOLO\rebook\dataset\data.yaml"
WEIGHTS   = r"C:\Users\noovelUser\Documents\YOLO\rebook\runs\rebook_shoe_v2\weights\best.pt"

CONF_THRES = 0.10   # show predictions above this conf
IOU_THRES  = 0.50
WINDOW_W, WINDOW_H = 1600, 900
# ----------------------------------------

def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    root = Path(d.get("path", Path(path).parent))

    val_rel = d.get("val", "images/val")
    val_images_dir = (root / val_rel).resolve()

    # assume labels mirror images folder structure:
    # images/val  -> labels/val
    # valid/images -> valid/labels (Roboflow style) would also work if set accordingly in yaml
    val_labels_dir = Path(str(val_images_dir).replace("\\images\\", "\\labels\\")).resolve()

    names = d.get("names", None)
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}

    return val_images_dir, val_labels_dir, names

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(imgs)

def read_yolo_labels(label_path: Path):
    """Returns list of (cls_id, x, y, w, h) normalized."""
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for ln in lines:
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

def xywhn_to_xyxy(px, py, pw, ph, img_w, img_h):
    x1 = int((px - pw / 2) * img_w)
    y1 = int((py - ph / 2) * img_h)
    x2 = int((px + pw / 2) * img_w)
    y2 = int((py + ph / 2) * img_h)
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2

def draw_gt(frame, labels, names=None):
    h, w = frame.shape[:2]
    for (cls_id, x, y, bw, bh) in labels:
        x1, y1, x2, y2 = xywhn_to_xyxy(x, y, bw, bh, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
        cv2.putText(frame, f"GT: {cls_name}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def draw_preds(frame, result):
    if result.boxes is None or len(result.boxes) == 0:
        return frame
    boxes = result.boxes
    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        if conf < CONF_THRES:
            continue
        cls_id = int(boxes.cls[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cls_name = result.names.get(cls_id, str(cls_id))
        cv2.putText(frame, f"PRED: {cls_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

def main():
    val_images_dir, val_labels_dir, names_yaml = load_data_yaml(DATA_YAML)
    images = list_images(val_images_dir)
    if not images:
        raise RuntimeError(f"No images found in {val_images_dir}")

    model = YOLO(WEIGHTS)

    win = "VAL Review (Left=GT, Right=PRED)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WINDOW_W, WINDOW_H)

    idx = 0
    while True:
        img_path = images[idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("Could not read:", img_path)
            idx = (idx + 1) % len(images)
            continue

        # label file with same stem
        label_path = val_labels_dir / (img_path.stem + ".txt")
        gt_labels = read_yolo_labels(label_path)

        # Predict
        res = model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

        left = frame.copy()
        right = frame.copy()

        # names: prefer model names (what it actually uses)
        names = res.names if hasattr(res, "names") else names_yaml

        left = draw_gt(left, gt_labels, names=names_yaml or names)
        right = draw_preds(right, res)

        # headers
        cv2.putText(left, f"GROUND TRUTH | {img_path.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(right, f"PREDICTIONS | conf>={CONF_THRES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # concat side-by-side (resize if needed to fit nicely)
        # Keep same height for concat
        h = min(left.shape[0], right.shape[0])
        left2 = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
        right2 = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
        vis = cv2.hconcat([left2, right2])

        cv2.imshow(win, vis)

        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27):  # q or esc
            break
        elif k in (ord('n'), 32, 13):  # n or space or enter
            idx = (idx + 1) % len(images)
        elif k == ord('p'):
            idx = (idx - 1) % len(images)
        else:
            # any other key -> next
            idx = (idx + 1) % len(images)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
