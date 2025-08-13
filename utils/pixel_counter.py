#!/usr/bin/env python3
"""
Count non-white pixels in one or more images.

A pixel is considered "white" if ALL (RGB/BGR/gray) channels are >= THRESHOLD.
Defaults to 250 (on 0–255 scale). For 16-bit images, the threshold is scaled.

Usage examples:
  python count_nonwhite.py img1.jpg img2.png
  python count_nonwhite.py /path/to/dir -r
  python count_nonwhite.py /path/to/dir -r -t 245 -e jpg png jpeg tif
  python count_nonwhite.py imgs -r --csv results.csv
"""

import argparse
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import csv

COMMON_EXTS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp")

def list_images(paths, exts, recursive):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for ext in exts:
                files.extend(p.glob(f"{pattern}.{ext}"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[WARN] Not found: {p}", file=sys.stderr)
    return sorted({f.resolve() for f in files})

def load_bgr_drop_alpha(path):
    # Load unchanged, then drop alpha if present
    img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("OpenCV failed to read image.")
    if img.ndim == 3 and img.shape[2] == 4:  # BGRA -> BGR
        img = img[:, :, :3]
    return img

def count_nonwhite_pixels(img, threshold_8bit):
    """
    Returns (nonwhite_count, total_pixels, dtype_bits)
    Works for grayscale (H,W), color (H,W,3), and 16-bit images.
    """
    if img.dtype == np.uint16:
        thr = int(threshold_8bit * 257)  # 255 -> 65535 mapping
    elif img.dtype == np.uint8:
        thr = threshold_8bit
    else:
        # Convert other types to uint8 safely
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        thr = threshold_8bit

    if img.ndim == 2:
        white_mask = img >= thr
        total = img.size
    else:
        # Use all channels (B,G,R). Pixel is white only if ALL channels >= thr.
        white_mask = np.all(img >= thr, axis=2)
        total = img.shape[0] * img.shape[1]

    white_count = int(np.count_nonzero(white_mask))
    nonwhite_count = int(total - white_count)
    bits = 16 if img.dtype == np.uint16 else 8
    return nonwhite_count, total, bits

def main():
    ap = argparse.ArgumentParser(description="Count non-white pixels in images.")
    ap.add_argument("paths", nargs="+", help="Image files and/or directories")
    ap.add_argument("-t", "--threshold", type=int, default=250,
                    help="8-bit whiteness threshold (0-255). Default: 250")
    ap.add_argument("-e", "--ext", nargs="*", default=list(COMMON_EXTS),
                    help="Allowed file extensions (no dots). Default: common image types")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Recurse into subfolders for directories")
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional CSV output path")
    args = ap.parse_args()

    # Validate threshold
    if not (0 <= args.threshold <= 255):
        print("[ERROR] --threshold must be in [0, 255].", file=sys.stderr)
        sys.exit(2)

    files = list_images(args.paths, [e.lower() for e in args.ext], args.recursive)
    if not files:
        print("[ERROR] No images found with given paths/extensions.", file=sys.stderr)
        sys.exit(1)

    rows = []
    grand_nonwhite = 0
    grand_total = 0

    for f in files:
        try:
            img = load_bgr_drop_alpha(f)
            nonwhite, total, bits = count_nonwhite_pixels(img, args.threshold)
            pct = (nonwhite / total) * 100.0 if total else 0.0
            grand_nonwhite += nonwhite
            grand_total += total
            print(f"{f.name}: non-white={nonwhite:,}  total={total:,}  "
                  f"pct={pct:6.2f}%  depth={bits}-bit")
            rows.append([str(f), nonwhite, total, f"{pct:.4f}", bits])
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}", file=sys.stderr)

    if grand_total > 0:
        overall_pct = (grand_nonwhite / grand_total) * 100.0
        print("-" * 60)
        print(f"TOTAL: non-white={grand_nonwhite:,}  total={grand_total:,}  "
              f"pct={overall_pct:6.2f}%")

    if args.csv and rows:
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["path", "nonwhite_pixels", "total_pixels", "percent_nonwhite", "bit_depth"])
            writer.writerows(rows)
        print(f"[INFO] Wrote CSV -> {outp}")

if __name__ == "__main__":
    main()
