from pathlib import Path

DATASET = Path(r"C:\Users\noovelUser\Documents\YOLO\rebook\dataset")

splits = ["train", "val", "test"]

for split in splits:
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split

    if not img_dir.exists():
        continue

    total_images = 0
    positive_images = 0
    total_boxes = 0

    for img in img_dir.glob("*.*"):
        total_images += 1
        label_file = lbl_dir / (img.stem + ".txt")

        if label_file.exists():
            with open(label_file, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) > 0:
                    positive_images += 1
                    total_boxes += len(lines)

    negatives = total_images - positive_images

    print(f"\n--- {split.upper()} ---")
    print(f"Images:     {total_images}")
    print(f"Positives:  {positive_images}")
    print(f"Negatives:  {negatives}")
    print(f"Total boxes:{total_boxes}")