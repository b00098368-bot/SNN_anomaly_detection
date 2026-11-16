# prepare_data.py : the ONE step that needs the raw event files, so it is run once, by whoever has them.
# Self-contained: needs only Python, numpy and the three official files from the dataset authors'
# repo (train_split.txt, test_split.txt, gt-ucf-dic.pickle). It never changes the raw data.
#
# What it does
#   1. Splits the NORMAL videos into train / val / test (70/15/15, whole videos, fixed seed).
#   2. Takes the official test videos of the chosen anomaly classes as the anomalous test set.
#   3. Converts the official frame labels into one small file per test video.
#   4. Turns every selected video into 533 ms windows: counts of ON and OFF events on a 64x64 grid.
#
# Usage (Windows cmd, from the folder holding this file):
#   python prepare_data.py --data dataset --out work
#   --data : folder with the event .npz files (searched recursively) and the three official files
#   --out  : output folder (default work). THIS FOLDER (small) is what gets shared for training.
# Safe to stop and run again: videos that are already converted are skipped.
import argparse
import csv
import json
import pickle
import re
import time
from pathlib import Path
import numpy as np

# ---------------- settings (the training code must use the same values) ----------------
CLASSES = ["Fighting", "Assault"]     # anomaly classes used for testing
WINDOW_US = 533_328                   # one window = 533,328 microseconds = 16 video frames
FRAMES_PER_WINDOW = 16
SIZE = 64                             # events are summed into a 64 x 64 grid
SENSOR_W, SENSOR_H = 1280, 720
SPLIT_SEED = 0
VAL_FRACTION = 0.15
TEST_NORMAL_FRACTION = 0.15
CHUNK = 10_000_000                    # events processed at a time (keeps memory small)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--out", default="work")
ap.add_argument("--classes", nargs="+", default=CLASSES)
args = ap.parse_args()
DATA, OUT, CLASSES = Path(args.data), Path(args.out), args.classes


# ---------------- helpers ----------------
def video_class(name):
    """'Fighting003_x264' -> 'Fighting', 'Normal_Videos_003_x264' -> 'Normal'."""
    stem = re.sub(r"[_\d]+$", "", name.replace("_x264", ""))
    return "Normal" if stem.startswith("Normal") else stem


def find_one(filename):
    hits = sorted(DATA.rglob(filename))
    if not hits:
        raise SystemExit(f"Could not find {filename} under {DATA}. Put the three official files in that folder.")
    return hits[0]


def read_list(path):
    """One video per line, e.g. 'Fighting003_x264.npy'. Keeps only the name."""
    with open(path) as f:
        return [Path(line.split()[0]).stem for line in f if line.split()]


class PlainDataOnly(pickle.Unpickler):
    """Refuses to import anything, so the label pickle can only hold plain data."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"pickle wants to import {module}.{name}, refusing")


def events_to_counts(path):
    """One video -> uint16 array (n_windows, 2, SIZE, SIZE). Channel 0 = OFF events, 1 = ON events."""
    d = np.load(path)
    t, x, y, p = d["t"].astype(np.int64), d["x"], d["y"], d["p"]
    t = t - t.min()                                     # windows start at the video's first event
    n = int(t.max() // WINDOW_US) + 1
    counts = np.zeros(n * 2 * SIZE * SIZE, dtype=np.int64)
    for i in range(0, len(t), CHUNK):
        w = t[i:i + CHUNK] // WINDOW_US
        pol = (p[i:i + CHUNK] > 0).astype(np.int64)
        xs = x[i:i + CHUNK].astype(np.int64) * SIZE // SENSOR_W   # int64 first: uint16 would overflow
        ys = y[i:i + CHUNK].astype(np.int64) * SIZE // SENSOR_H
        counts += np.bincount(((w * 2 + pol) * SIZE + ys) * SIZE + xs, minlength=counts.size)
    return np.clip(counts, 0, 65535).astype(np.uint16).reshape(n, 2, SIZE, SIZE), len(t)


# ---------------- find the data ----------------
paths = {}
for p in DATA.rglob("*.npz"):
    if p.stem in paths:
        print("WARNING: duplicate file name, using the first one:", p.stem)
    else:
        paths[p.stem] = p
train_all, test_all = read_list(find_one("train_split.txt")), read_list(find_one("test_split.txt"))
print(f"event files found under {DATA}: {len(paths)}  (examples: {sorted(paths)[:2]})")
if not paths:
    raise SystemExit("No .npz event files found. Check --data.")

# ---------------- 1. split the normal videos ----------------
normal = sorted(n for n in set(train_all + test_all) if n in paths and video_class(n) == "Normal")
n_off_test = sum(n in test_all for n in normal)
print(f"normal videos on disk: {len(normal)} ({n_off_test} in the official test list, "
      f"{len(normal) - n_off_test} in the official train list)")
order = np.random.RandomState(SPLIT_SEED).permutation(len(normal))
n_val, n_test = round(VAL_FRACTION * len(normal)), round(TEST_NORMAL_FRACTION * len(normal))
rows = [(normal[i], "val" if r < n_val else "test" if r < n_val + n_test else "train", "Normal")
        for r, i in enumerate(order)]

# ---------------- 2. anomalous test videos ----------------
missing = []
for c in CLASSES:
    wanted = [n for n in test_all if video_class(n) == c]
    found = [n for n in wanted if n in paths]
    missing += [n for n in wanted if n not in paths]
    print(f"{c}: {len(found)} of {len(wanted)} official test videos found")
    rows += [(n, "test", c) for n in found]
if missing:
    print("MISSING anomalous test videos:", missing)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(exist_ok=True)
(OUT / "cache").mkdir(exist_ok=True)
with open(OUT / "split_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video", "role", "class"])
    w.writerows(rows)
summary = {}
for _, role, c in rows:
    summary[f"{role}/{c}"] = summary.get(f"{role}/{c}", 0) + 1
print("split:", dict(sorted(summary.items())))

# ---------------- 3. frame labels (only official test videos have them) ----------------
with open(find_one("gt-ucf-dic.pickle"), "rb") as f:
    labels = PlainDataOnly(f).load()
n_lab = 0
for video, role, _ in rows:
    if role == "test" and video in labels:
        np.save(OUT / "labels" / f"{video}.npy", np.asarray(labels[video], dtype=np.uint8))
        n_lab += 1
print(f"wrote {n_lab} label files (normals from the official TRAIN list have none: they are all normal)")

# ---------------- 4. events -> windows ----------------
start = time.time()
n_events = {}
for k, (video, _, _) in enumerate(rows, 1):
    out = OUT / "cache" / f"{video}.npz"
    if out.exists():
        continue
    counts, n_events[video] = events_to_counts(paths[video])
    np.savez_compressed(out, w=counts)
    print(f"[{k}/{len(rows)}] {video}: {n_events[video]:,} events -> {len(counts)} windows "
          f"({time.time() - start:.0f} s elapsed)")

# ---------------- checks and report ----------------
bad = []
for video, _, _ in rows:
    lab = OUT / "labels" / f"{video}.npy"
    if lab.exists():
        n_win, n_frames = len(np.load(OUT / "cache" / f"{video}.npz")["w"]), len(np.load(lab))
        if abs(n_win * FRAMES_PER_WINDOW - n_frames) > FRAMES_PER_WINDOW:
            bad.append((video, n_win, n_frames))
size_mb = sum(p.stat().st_size for p in (OUT / "cache").glob("*.npz")) / 1e6
json.dump(dict(classes=CLASSES, window_us=WINDOW_US, size=SIZE, split_seed=SPLIT_SEED,
               val_fraction=VAL_FRACTION, test_normal_fraction=TEST_NORMAL_FRACTION),
          open(OUT / "prep_info.json", "w"), indent=2)

print("\n===== REPORT (send this block back) =====")
print("event files:", len(paths), "| normals on disk:", len(normal),
      f"({n_off_test} official test, {len(normal) - n_off_test} official train)")
print("split:", dict(sorted(summary.items())))
print("missing anomalous test videos:", missing or "none")
print(f"windows vs label length mismatches: {len(bad)}", bad[:5])
print(f"output folder: {OUT.resolve()}  size: {size_mb:.0f} MB  -> share this folder")
