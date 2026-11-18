# 🚁 Finding People in Drone Images (No Neural Networks)

Can you spot a lost person in a drone photo using only classical machine learning: handcrafted features, no deep learning?

Turns out, mostly yes. This project scans aerial search-and-rescue images from the **SARD 1** dataset and flags the ones that contain a person. It catches **~95% of images with a human in them** on the held-out test set.

The goal is **recall, not accuracy**. In search and rescue, a false alarm costs a few seconds of a reviewer's time. A missed survivor costs far more.

## How it works

1. **Cut patches.** Crop people out of the labelled boxes (HUMAN) and sample random background windows (NOT HUMAN). Each patch is 64×128 px, since people in these images are tall and small.
2. **Turn patches into numbers.** HOG (shape), HSV colour histograms, and LBP (texture).
3. **Train classifiers.** Nine models compared; Random Forest wins on recall.
4. **Scan whole images.** Slide a window over the photo at 4 sizes (~1,137 windows). The image's score is its single highest window score.
5. **Flag it.** If the score clears a threshold (picked on validation data only), the image is flagged as "has a human".

## Results

Final model: Random Forest on HOG + colour + LBP, with one round of hard-negative mining. The threshold (0.35) was chosen on validation to hit ≥95% recall and frozen before the test set was touched.

| | Validation (1,144 imgs) | **Test (570 imgs)** |
|---|---|---|
| Recall | 0.950 | **0.955** |
| Precision | 0.937 | 0.939 |
| F2 | 0.948 | 0.952 |
| False alarms on human-free images | 35.0% | 35.7% |

Moving the threshold trades recall against false alarms:

| Threshold | Recall | Human-free images flagged |
|---|---|---|
| 0.50 | 0.886 | 10.7% |
| **0.35** | **0.950** | **35.0%** |
| 0.22 | 0.992 | 76.3% |

> About 85% of images contain a person, so image-level precision looks flattering. The false-alarm rate on human-free images is the more honest cost, though with only 84 such test images it's noisy.

### Things we learned

- **Colour does the heavy lifting.** HOG alone gets Random Forest to F2 = 0.66. Add colour and it jumps to 0.95. LBP texture adds nothing. (Caveat: most negatives are grass and dirt, so colour separates them easily. It may matter less against rocks or water.)
- **Fancy tuning barely helped.** Optuna-tuned models were no better than defaults, and the tuned Random Forest was actually worse. The default 300-tree forest stayed as the main model.
- **Hard-negative mining is worth it.** Retraining with the worst false positives from human-free photos cut false-alarm windows per human-free image from 3.67 to 0.55.
- **Accuracy is a trap.** ~80% of patches are "not human", so a model can look great while missing people. KNN and Extra Trees have ~99% precision but miss a third or more of humans.

### Where it fails

**Missed people** are mostly small (median 48 px vs 83 px overall), often lying down (the tall window suits standing people), and hidden in dense grass, dark forest, or shadow. Many misses are near-misses scoring 0.30–0.34, just under the threshold.

**False alarms** are mostly rock piles, then vehicles, water edges, and shadowy vegetation. Their scores are low, typically just above the threshold.

## Run it

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The web app lets you upload images or point at a folder. Each image is scanned, flagged if its best window clears the threshold (adjustable slider, default 0.35), and shown with red boxes on the likely people. You also get a results table and CSV export. Everything runs locally and uses all your CPU cores (~1 s/image on 8 logical cores).

Put the dataset in a `SARD1/` folder in the project root.

## Reproducing the pipeline

Run everything from the project root (scripts use relative paths). Seed 42 throughout. The full pipeline takes a few hours on 8 cores, mostly the sliding-window inference.

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_analyze.py` | Dataset analysis, box-size plots |
| 2 | `step2_patches.py` | Build human / background patches |
| 3 | `step3_features.py` | HOG + colour + LBP features |
| 4 | `step4_baselines.py` | Compare 9 models |
| 5 | `step5_tune.py` | Optuna tuning |
| 6 | `step6_ablation.py` | Feature ablation |
| 7 | `step7_benchmark.py`, `step7_mine.py` | Speed test, hard-negative mining |
| 8 | `step8_infer.py valid`, `step8_threshold.py` | Validation scoring and threshold choice |
| 9 | `step8_infer.py test`, `step9_test.py` | One-time test evaluation |
| 10 | `step10_errors.py` | Error galleries and stats |

```
scripts/   all .py files        charts/   all .png figures
csvs/      result tables        models/   saved .joblib models
optuna/    tuning databases     patches, features and scores (numpy arrays)/   .npz / .npy data
```

## About the data

SARD 1: 640×640 drone images, one class (human). 5,755 images split into train (4,041), valid (1,144), and test (570), with ~15% containing no people.

The files came from Roboflow with augmentation, so each original photo appears ~3 times. No original photo crosses a split boundary, so there's no train/test leakage, but the effective sample size is smaller than the image counts suggest.

## Limitations

- **Image-level only.** A flagged image might be flagged by the wrong window. There's no box-level matching (IoU, NMS) yet.
- **Small and hidden people** remain the main weakness: roughly 1 in 20 images with a person is still missed.
- **Single seed, limited tuning** (12–25 Optuna trials on subsampled data). Variance across seeds wasn't measured.
- **One dataset.** Generalisation to SARD 2 and SeaDronesSee wasn't tested.

This is meant as **decision support for a human reviewer, not a replacement for one.**

## Ideas for next steps

Bounding boxes with NMS and IoU evaluation · SARD 2 and SeaDronesSee tests · a second mining round · multi-seed results · image-level recall for the unmined model
