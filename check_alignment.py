# check_alignment.py : small diagnostic  (needs the raw event files). Run after prepare_data.py.
#   python check_alignment.py --data dataset --out work
# For every TEST video it prints when the event stream starts and ends, next to the length of the
# official frame labels (30 frames per second). Send the printed table back.
# Why: the cache has 1 to 5 fewer windows than the labels, and we need to know whether the
# event stream starts late or ends early before we trust the frame-level scores.
import argparse
import csv
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--out", default="work")
a = ap.parse_args()
paths = {p.stem: p for p in Path(a.data).rglob("*.npz")}
rows = [r for r in csv.DictReader(open(Path(a.out) / "split_manifest.csv")) if r["role"] == "test"]

print(f"{'video':28} {'t.min s':>8} {'t.max s':>8} {'stream s':>9} {'label s':>8} {'label-stream s':>14}")
for r in rows:
    v = r["video"]
    lab = Path(a.out) / "labels" / f"{v}.npy"
    if v not in paths or not lab.exists():
        continue
    t = np.load(paths[v])["t"]
    t0, t1 = t.min() / 1e6, t.max() / 1e6
    label_s = len(np.load(lab)) / 30.0
    print(f"{v:28} {t0:8.3f} {t1:8.3f} {t1 - t0:9.3f} {label_s:8.3f} {label_s - (t1 - t0):14.3f}")
