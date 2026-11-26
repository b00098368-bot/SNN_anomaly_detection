# One-Class Anomaly Detection on UCF-Crime-DVS: Spiking vs Convolutional Autoencoder

COE 476 Neural Networks and Deep Learning, Fall 2026 (Track A, Neuromorphic AI)
Authors: Zohair Syed (00098429), Kevin Alex (00098368)

## What this project does

We train a convolutional autoencoder (CNN-AE) and a spiking autoencoder (SNN-AE) on **normal** event-camera surveillance clips only. The reconstruction error of a window is its anomaly score, and we evaluate it against the official frame-level labels of five anomaly classes (Arson, Assault, Burglary, Explosion, Fighting).

We also test one advancement, **per-window event-rate normalization**, and a clip-length ablation (T = 2, 4, 8), with 3 seeds each. Two no-learning references (event count and a zero predictor) show how much of the score is just "how busy the window is".

Main result (T = 4, pooled frame-level AUROC, mean over 3 seeds): SNN-AE 0.694, CNN-AE 0.687, event count 0.667, zero predictor 0.651. Normalization lowered both models (SNN-AE 0.671, CNN-AE 0.629). See the report for the full analysis, including bootstrap intervals.

## Repository contents

```
COE476_proj.ipynb     main notebook (data inspection, models, training, evaluation, all experiments)
data/ (needs to be added)
  cache.zip                   preprocessed event-count windows, one file per video
  work_meta.zip               split manifest, frame labels, preprocessing info 
work/                      saved outputs (results.csv, summary.csv, cost.csv, per_class_T4.csv,
                              environment.txt and figures)
requirements.txt              package versions used
README.md
```

## Data

The dataset is **UCF-Crime-DVS** (Qian et al., AAAI 2025), real event recordings of the UCF-Crime videos, released under the MIT licence: https://github.com/YBQian-Roy/UCF-Crime-DVS. Frame-level labels come from the official `gt-ucf-dic.pickle`. Please credit the dataset authors and keep the licence notice if you reuse the data files.

The raw event files are large and are **not** included. All experiments run from the two preprocessed archives in `data/`:

| Archive | Unpacks to | Contents |
|---|---|---|
| `cache.zip` | `cache/<video>.npz` (198 files) | Key `w`: uint16 array of shape (windows, 2, 64, 64). Event counts per window, channel 0 = OFF, channel 1 = ON. Raw counts only; scaling is done in the notebook. |
| `work_meta.zip` | `split_manifest.csv` | One row per video: name, role (train / val / test), class. |
| | `labels/<video>.npy` | 0/1 anomaly label per video frame (30 fps). A video without a label file is treated as all normal. |
| | `prep_info.json` | Preprocessing settings. The notebook reads `window_us`. |

**How they were made (done once, outside the notebook):** each video's events were shifted so the first event is at time 0 and cut into consecutive windows of 533,328 microseconds (16 video frames at 30 fps). Sensor coordinates (1280x720) were mapped to a 64x64 grid by integer scaling, and events were counted per window, polarity and cell. The normal videos (Normal_Videos_003 to 944; 147 used) were shuffled with seed 0 and split by whole video into 103 train, 22 validation and 22 test. The 51 anomalous videos (Arson 9, Assault 3, Burglary 13, Explosion 21, Fighting 5) come from the official test list and are test-only.

## Setup

Tested with Python 3.13.17 on Windows 11 and an NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.8).

```bash
pip install -r requirements.txt
```

A GPU makes training much faster. The notebook also runs on CPU, only slowly. For a CUDA build of PyTorch, install it from https://pytorch.org first, then the rest of the requirements.

## How to run

1. Keep `cache.zip` and `work_meta.zip` in a folder called `data` next to the notebook.
2. Open `COE476_proj.ipynb` and run the cells **from top to bottom**.
3. The first cell unpacks the archives into `work/` (set the environment variable `WORK_DIR` to use another folder). Unpacking is skipped if `work/cache` already exists.

The experiments cell trains 3 seeds for each configuration and saves results after every run. If the notebook stops, run the setup cells again and then the experiments cell: finished runs are skipped.

### Settings

All settings are in one cell near the top.

| Setting | Value |
|---|---|
| Seeds | 0, 1, 2 (they change weight initialization and batch order; clip sampling and the split are fixed) |
| Epochs, batch, learning rate | at most 30, 64, 1e-3, with early stopping (patience 3) on validation loss |
| Clips per training video | at most 200 |
| `USE_BN`, `SNN_LEARN`, `MATCH_LATENT` | True, True, True |
| `RUN_ABLATION` | True (adds T = 2 and 8) |
| `CODE_VERSION` | v2 (bump it if you change the model or training code, so old results are never mixed in) |

## Rerunning from scratch

The experiments cell skips every run that is already listed in `work/results.csv`, so running the notebook again on top of existing results retrains nothing and only rebuilds the tables and figures from the saved scores.

To retrain everything, delete `work/results.csv` and the `work/scores/` folder, then run the notebook from top to bottom. The notebook recreates both. Seeds, the split and the clip sampling are fixed, so a rerun uses the same data and settings. GPU training is not exactly deterministic, so expect results close to ours but not identical to the last decimal.

If the notebook stops partway, run the setup cells again and then the experiments cell. Finished runs are skipped.

## Outputs (written to `work/`)

| File | Content |
|---|---|
| `results.csv` | one row per run and scope (pooled and each class) |
| `summary.csv` | mean and std over seeds |
| `scores/*.npz` | per-video window scores of every run (used by the bootstrap and plots) |
| `cost.csv` | parameters, MACs per clip, inference and training time, peak GPU memory |
| `per_class_T4.csv` | per-class AUROC table (pooled / within-video) |
| `roc_T4.png`, `score_over_time.png`, `auroc_vs_T.png` | evaluation figures |
| `environment.txt` | package versions and hardware of the run |

## Notes and limitations

- No threshold is tuned: AUROC and AUPRC need none, and validation loss is used only for early stopping. No setting was chosen using test results.
- The seed standard deviation measures training randomness only. The video-level bootstrap intervals in the report are the honest uncertainty, since the test set has only 73 videos.
- Spike activity and power were not measured, so we make no power-saving claim. The SNN-AE is simulated densely on a GPU and costs more compute than the CNN-AE (see `cost.csv`).

## Reused and original work

Reused: PyTorch, snnTorch, scikit-learn, and the UCF-Crime-DVS data. Original: the models and training loop, the per-window normalization, the event-count and zero-predictor references, the per-video AUROC, the window-to-frame scoring, the video-level bootstrap and the cost measurements.

