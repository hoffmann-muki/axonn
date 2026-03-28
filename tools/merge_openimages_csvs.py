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
    args = p.parse_args()

    img_map = {}
    with open(args.images, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = row.get("ImageID") or row.get("image_id")
            url = row.get("OriginalURL") or row.get("Original_URL") or row.get("OriginalUrl") or row.get("image_url")
            if image_id and url:
                img_map[image_id] = url

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as outfh:
        w = csv.writer(outfh)
        w.writerow(["ImageID", "LabelName", "OriginalURL"])
        with open(args.annotations, newline="") as annfh:
            r = csv.DictReader(annfh)
            for row in r:
                image_id = row.get("ImageID") or row.get("image_id")
                label = row.get("LabelName") or row.get("Label")
                url = img_map.get(image_id)
                if image_id and label and url:
                    w.writerow([image_id, label, url])

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
