#!/usr/bin/env python3
# tools/build_imagefolder.py
# Usage: python tools/build_imagefolder.py --csv annotations.csv --out /data/open_images --split train --max-per-class 1000

import argparse
import csv
import os
import requests
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--csv", help="CSV with columns: url,label (header names 'url' and 'label')")
parser.add_argument("--openimages-csv", help="Official OpenImages CSV with columns ImageID,LabelName,OriginalURL")
parser.add_argument("--label-map-csv", help="Optional class-descriptions CSV mapping LabelName to human-readable label")
parser.add_argument("--out", required=True, help="Output root (will create <out>/<split>/<label>/...)")
parser.add_argument("--split", default="train", help="Dataset split folder name (train/validation)")
parser.add_argument("--max-per-class", type=int, default=0, help="0 = unlimited")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--max-images", type=int, default=0, help="Limit total images processed (0 = unlimited)")
parser.add_argument("--multi-label", choices=["first", "skip", "multi"], default="first", help="How to handle images with multiple labels: first=use first label, skip=ignore multi-label images, multi=place image in all label folders")
parser.add_argument("--save-csv", help="Optional path to write the converted url,label CSV")
parser.add_argument("--skip-existing", action="store_true", help="Skip images that already exist in target folders")
args = parser.parse_args()

out_root = Path(args.out)
split_dir = out_root / args.split
split_dir.mkdir(parents=True, exist_ok=True)

# Build mapping: label -> list of (image_id, url)
by_label = defaultdict(list)

# Optional label name mapping (LabelName -> display name)
label_name_map = {}
if args.label_map_csv:
    with open(args.label_map_csv, newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            key = row.get("LabelName") or row.get("label") or row.get("id")
            name = row.get("DisplayName") or row.get("display_name") or row.get("Label") or row.get("label_name")
            if key and name:
                label_name_map[key] = name

image_labels = defaultdict(list)  # ImageID -> list of LabelName
image_urls = {}  # ImageID -> URL

if args.openimages_csv:
    with open(args.openimages_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = row.get("ImageID") or row.get("Image_Id") or row.get("ImageID")
            label = row.get("LabelName") or row.get("Label") or row.get("LabelName")
            url = row.get("OriginalURL") or row.get("OriginalURL") or row.get("OriginalURL") or row.get("URL") or row.get("ImageURL")
            if not image_id or not label or not url:
                continue
            image_labels[image_id].append(label)
            # Prefer the first URL if multiple rows reference same image
            if image_id not in image_urls:
                image_urls[image_id] = url

    # Convert image_labels + image_urls into by_label according to multi-label policy
    count = 0
    for image_id, labels in image_labels.items():
        if args.max_images and count >= args.max_images:
            break
        url = image_urls.get(image_id)
        if not url:
            continue
        if args.multi_label == "skip" and len(labels) > 1:
            continue
        if args.multi_label == "first":
            use_labels = [labels[0]]
        else:
            use_labels = labels
        for lab in use_labels:
            out_label = label_name_map.get(lab, lab)
            by_label[out_label].append((image_id, url))
        count += 1

elif args.csv:
    # Generic url,label CSV
    with open(args.csv, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = row.get("url") or row.get("image_url") or row.get("image_url_original")
            label = row.get("label") or row.get("LabelName") or row.get("class")
            if not url or not label:
                continue
            by_label[label].append((None, url))

else:
    raise SystemExit("Provide either --openimages-csv or --csv to build the dataset")

# enforce max per class
for label, items in list(by_label.items()):
    if args.max_per_class > 0:
        by_label[label] = items[: args.max_per_class]

cache_dir = out_root / ".cache_images"
cache_dir.mkdir(parents=True, exist_ok=True)

def _ext_from_url(url):
    p = urlparse(url).path
    name = Path(p).name
    ext = Path(name).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        return ext
    return ".jpg"

def download_one(item):
    """Download image into cache and return cache path or None on failure.

    item: (image_id, url)
    """
    image_id, url = item
    # Use image_id if available for stable filenames, else use hash of URL
    if image_id:
        fname = image_id
    else:
        fname = str(abs(hash(url)))
    ext = _ext_from_url(url)
    cache_path = cache_dir / f"{fname}{ext}"
    if cache_path.exists():
        return cache_path
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(cache_path, "wb") as out:
            for chunk in resp.iter_content(1024 * 32):
                out.write(chunk)
        return cache_path
    except Exception:
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception:
                pass
        return None

from shutil import copy2

tasks = []
for label, items in by_label.items():
    safe_label = label.replace("/", "_")
    dest_dir = split_dir / safe_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        tasks.append((label, it))

total = len(tasks)
print(f"Preparing to download {total} images into {split_dir}")

def worker(task):
    label, (image_id, url) = task
    safe_label = label.replace("/", "_")
    dest_dir = split_dir / safe_label
    # Determine final path
    ext = _ext_from_url(url)
    filename = (image_id if image_id else str(abs(hash(url)))) + ext
    dest_path = dest_dir / filename
    if args.skip_existing and dest_path.exists():
        return True

    cache_path = download_one((image_id, url))
    if cache_path is None:
        return False

    try:
        # copy cached image into class folder
        copy2(cache_path, dest_path)
        return True
    except Exception:
        return False

success = 0
fail = 0
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for ok in ex.map(worker, tasks):
        if ok:
            success += 1
        else:
            fail += 1

print(f"Download complete: {success} succeeded, {fail} failed")

if args.save_csv:
    with open(args.save_csv, "w", newline='') as outfh:
        w = csv.writer(outfh)
        w.writerow(["url", "label"])
        for label, items in by_label.items():
            for image_id, url in items:
                w.writerow([url, label])
    print("Wrote converted CSV to:", args.save_csv)

print("Done. ImageFolder located at:", split_dir)