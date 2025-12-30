from pathlib import Path
from PIL import Image, ImageOps

# ---------- CONFIG ----------
INPUT_DIR = Path(r"C:\Users\noovelUser\Documents\YOLO\rebook\z")
OUTPUT_DIR = Path(r"C:\Users\noovelUser\Documents\YOLO\rebook\z_formatted")

PREFIX = "img_"
START_INDEX = 1
DIGITS = 6

MAX_SIZE = 2000          # max width or height
QUALITY = 95
RECURSIVE = True
KEEP_ORIGINALS = True    # False = delete originals after conversion
# --------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def iter_images(root: Path):
    pattern = "**/*" if RECURSIVE else "*"
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def to_rgb(img: Image.Image) -> Image.Image:
    # handle alpha/palette/etc.
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def resize_keep_aspect(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    scale = min(max_size / w, max_size / h, 1.0)  # never upscale
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(iter_images(INPUT_DIR))
    if not files:
        print(f"No images found in: {INPUT_DIR}")
        return

    idx = START_INDEX
    for src in files:
        dst_name = f"{PREFIX}{idx:0{DIGITS}d}.jpg"
        dst = OUTPUT_DIR / dst_name

        try:
            with Image.open(src) as im:
                # 1) Apply EXIF rotation (makes image "native" upright)
                im = ImageOps.exif_transpose(im)

                # 2) Convert + resize
                im = to_rgb(im)
                im = resize_keep_aspect(im, MAX_SIZE)

                # 3) Save to JPG
                im.save(dst, format="JPEG", quality=QUALITY, optimize=True)

            if not KEEP_ORIGINALS:
                src.unlink(missing_ok=True)

            print(f"OK  {src.name} -> {dst_name}")
            idx += 1

        except Exception as e:
            print(f"FAIL {src} ({e})")

    print(f"\nDone. Wrote {idx - START_INDEX} images to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()