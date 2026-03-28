#!/usr/bin/env python3
"""Merge OpenImages image metadata and human annotations into a combined CSV.

Produces rows: ImageID,LabelName,OriginalURL suitable for tools/build_imagefolder.py

Usage:
    python3 tools/merge_openimages_csvs.py \
        --images oidv6-train-images-boxable.csv \
        --annotations oidv6-train-annotations-human-imagelabels.csv \
        --out train-openimages-combined.csv
"""
import csv
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="OpenImages images CSV (contains ImageID, OriginalURL)")
    p.add_argument("--annotations", required=True, help="OpenImages annotations CSV (ImageID, LabelName)")
    p.add_argument("--out", required=True, help="Output combined CSV path")
    p.add_argument("--min-confidence", type=float, default=1.0, help="Minimum Confidence value to accept (default 1.0)")
    p.add_argument("--multi-label", choices=["first", "skip", "multi"], default="first", help="How to handle images with multiple labels")
    args = p.parse_args()

    # load image id -> original url map
    img_map = {}
    with open(args.images, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = (row.get("ImageID") or row.get("image_id") or "").strip()
            url = (row.get("OriginalURL") or row.get("Original_URL") or row.get("OriginalUrl") or row.get("image_url") or "").strip()
            if image_id and url:
                img_map[image_id] = url

    # collect annotations grouped by image
    image_labels = {}
    missing_url = 0
    total_rows = 0
    kept_rows = 0
    with open(args.annotations, newline="") as annfh:
        r = csv.DictReader(annfh)
        for row in r:
            total_rows += 1
            image_id = (row.get("ImageID") or row.get("image_id") or "").strip()
            label = (row.get("LabelName") or row.get("Label") or "").strip()
            conf_raw = (row.get("Confidence") or row.get("confidence") or "").strip()
            if not image_id or not label:
                continue
            # parse confidence if present
            if conf_raw:
                try:
                    conf = float(conf_raw)
                except Exception:
                    # non-numeric (e.g., 'true') -> treat as 1.0
                    conf = 1.0
                if conf < args.min_confidence:
                    continue
            # ensure we have an image URL mapping
            if image_id not in img_map:
                missing_url += 1
                continue
            kept_rows += 1
            image_labels.setdefault(image_id, []).append(label)

    # write output according to multi-label policy
    out_path = Path(args.out)
    with open(out_path, "w", newline="") as outfh:
        w = csv.writer(outfh)
        w.writerow(["ImageID", "LabelName", "OriginalURL"])
        skipped_multi = 0
        written = 0
        for image_id, labels in image_labels.items():
            url = img_map.get(image_id)
            if not url:
                continue
            if args.multi_label == "skip" and len(labels) > 1:
                skipped_multi += 1
                continue
            if args.multi_label == "first":
                chosen = [labels[0]]
            else:
                chosen = labels
            for lab in chosen:
                w.writerow([image_id, lab, url])
                written += 1

    print(f"Total annotation rows read: {total_rows}")
    print(f"Rows kept after confidence/url filtering: {kept_rows}")
    print(f"Images missing URL mapping: {missing_url}")
    if args.multi_label == "skip":
        print(f"Images skipped due to multi-label: {skipped_multi}")
    print(f"Wrote {written} rows to: {out_path}")


if __name__ == "__main__":
    main()
