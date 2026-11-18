# Human Detection in Search & Rescue Aerial Images (classical ML)

Team: Amitoj Singh Karir, Priyanka Nathani, Jaskaran Singh Panjetta, Pranav Bhoj

Find people in drone images (SARD 1 dataset) using handcrafted features and classical ML only (no neural networks).
Main goal: **recall**, because a missed survivor is worse than a false alarm.

## How it works

1. Cut each image into small patches: HUMAN (from the labelled boxes) and NOT HUMAN (random background windows).
2. Turn each 64x128 patch into numbers: HOG (shape), HSV colour histogram, LBP (texture).
3. Train classifiers on the patches.
4. On a full image, slide a window over it (4 sizes, ~1,137 windows). **The image's score is its highest window score.**
5. If the score is above a threshold, the image is flagged as "has a human". The threshold is picked on validation only.

## Files and run order

| Step | File | What it does |
|---|---|---|
| 1 | `step1_analyze.py` | dataset analysis, box-size histograms |
| 2 | `step2_patches.py` | builds patches -> `patches_*.npz` |
| 3 | `step3_features.py` | HOG + colour + LBP -> `features_*.npz` |
| 4 | `step4_baselines.py` | compares 9 models (results_baselines.csv, 3 charts) |
| 5 | `step5_tune.py` | Optuna tuning of 3 models (results_tuned.csv) |
| 6 | `step6_ablation.py` | feature ablation (results_ablation.csv) |
| 7 | `sliding.py`, `step7_benchmark.py`, `step7_mine.py` | sliding window, speed test, hard-negative mining |
| 8 | `step8_infer.py valid`, `step8_threshold.py` | validation inference and threshold choice |
| 9 | `step8_infer.py test`, `step9_test.py` | one-time test evaluation |
| 10 | `step10_errors.py` | error galleries and statistics |

Folders: `scripts/` (all `.py`), `charts/` (all `.png`), `csvs/` (result tables), `patches, features and scores (numpy arrays)/` (every `.npz`/`.npy`: patches, features, hard negatives, window scores, `test_errors.npz`, `threshold.npy`), `models/` (saved `.joblib` models), `optuna/` (tuning databases `.db`). In the main folder: `app.py` and `requirements.txt`.
**Run every script from the main `ML` folder**, e.g. `python scripts/step1_analyze.py`, because the scripts find the data by relative path.

## Using the trained model (web app)

```
python -m streamlit run app.py
```

Opens a webpage where you upload images or give a folder path. Each image is scanned with the sliding window, flagged if its best window score reaches the threshold (slider, default 0.35), and shown with red boxes on the likely people. It also gives a results table and a CSV download. It runs entirely on your own computer and spreads the scanning over all CPU cores (adjustable with the "CPU workers" slider; worker code is in `scripts/parallel.py`). About 1 s per image on 8 logical cores, against about 4.4 s on one core. `requirements.txt` lists the libraries.

Needs: numpy, opencv-python, scikit-image, scikit-learn, matplotlib, pandas, optuna, joblib. Dataset folder `SARD1/` in the main folder.
Seed 42 everywhere. The full pipeline takes a few hours on 8 CPU cores (the sliding-window inference is the slow part, about 23 min for validation and 12 min for test).

## Dataset (SARD 1)

| Split | Images | Without humans | Boxes |
|---|---|---|---|
| train | 4,041 | 571 (14.1%) | 5,229 |
| valid | 1,144 | 177 (15.5%) | 1,463 |
| test | 570 | 84 (14.7%) | 732 |

- Images are 640x640, one class (human). No box is zero-size or outside the image.
- Person boxes are tall and small: median 36x82 px, longer side median 86 px (90th percentile 164 px). That is why patches are 64 wide x 128 tall. Plot: `box_sizes.png`.
- **Near-duplicates:** the files come from Roboflow with augmentation. Train has 4,041 files but only 1,387 original photos (about 3 copies each). No original photo appears in more than one split, so there is no train/test leakage. The test set also contains repeated photos, so it holds fewer than 570 independent scenes.

Patches: 28,429 train (5,229 human), 8,023 valid, 3,996 test, about 4 negatives per human. Each patch's features: 3,780 HOG + 512 colour + 10 LBP.

## Results

All patch-level tables are on the **validation** set. The test set was used once, at the end.

### T1. Nine models, same features (HOG + colour + LBP), validation patches

| Model | Accuracy | Recall | Precision | F2 | Missed humans (of 1,463) |
|---|---|---|---|---|---|
| Random Forest | 0.976 | **0.958** | 0.916 | 0.949 | 62 |
| RBF SVM | **0.977** | 0.940 | 0.935 | 0.939 | 88 |
| Linear SVM | 0.955 | 0.934 | 0.836 | 0.913 | 96 |
| Logistic Regression | 0.954 | 0.929 | 0.835 | 0.908 | 104 |
| Gradient Boosting | 0.947 | 0.878 | 0.840 | 0.870 | 179 |
| Decision Tree | 0.932 | 0.814 | 0.813 | 0.814 | 272 |
| KNN | 0.955 | 0.765 | 0.983 | 0.800 | 344 |
| Extra Trees | 0.944 | 0.699 | 0.992 | 0.743 | 440 |
| Naive Bayes | 0.912 | 0.644 | 0.834 | 0.675 | 521 |

Charts: `chart_accuracy.png`, `chart_recall.png`, `chart_precision.png`.

- Random Forest has the best recall. RBF SVM has the best accuracy and precision.
- KNN and Extra Trees are very precise but miss many people (cautious models, bad for rescue).
- Accuracy is misleading: about 80% of patches are "not human".
- RBF SVM, KNN and Gradient Boosting used PCA (150 components, fitted on train, keeps 44% of variance). Random Forest and the linear models used all features.
- The first six are the models from the original plan; Extra Trees, Gradient Boosting and the linear/RBF SVM split were added.

### T1b. Optuna tuning (objective: F2 on validation)

| Model | Default F2 | Tuned F2 | Missed (default -> tuned) | Best parameters |
|---|---|---|---|---|
| Linear SVM | 0.913 | 0.918 | 96 -> 83 | C = 0.00032 |
| RBF SVM | 0.939 | 0.939 | 88 -> 82 | C = 88.5, gamma = 4.9e-5 |
| Random Forest | **0.949** | 0.941 | **62** -> 75 | 105 trees, depth 29, leaf 5 |

Tuning helped little. The tuned Random Forest was worse than the default because the search used a 10,000-patch subset, few trials (12 to 25) and limited depth. **The default Random Forest (300 trees) is the main model.**

### T2. Feature ablation (validation patches)

| Features | RF F2 | RF missed | RBF SVM F2 | RBF SVM missed |
|---|---|---|---|---|
| F1 HOG only | 0.659 | 572 | 0.883 | 172 |
| F2 HOG + colour | **0.949** | 61 | **0.939** | 84 |
| F3 HOG + LBP | 0.720 | 474 | 0.885 | 170 |
| F4 HOG + colour + LBP | **0.949** | 62 | **0.939** | 82 |

Chart: `chart_ablation.png`. **Colour is what makes the models work; LBP adds nothing.** Caveat: most negative patches are grass and ground, so colour separates them easily. It may help less against rocks, vehicles and water (see error analysis).

### T3. Hard-negative mining (one round)

Scanned 400 human-free train photos: 20,910 windows scored above 0.3 (about 52 per image). Kept the 10 worst per image = 1,960 hard negatives, added to the negatives, Random Forest retrained.

| | Before | After |
|---|---|---|
| Patch accuracy | 0.976 | 0.978 |
| Patch recall | 0.958 | 0.942 |
| Patch precision | 0.916 | 0.937 |
| Patch F2 | 0.949 | 0.941 |
| False-alarm windows per human-free validation image (score > 0.5) | 3.67 | 0.55 |
| Human-free validation images with at least one false alarm (score > 0.5) | 28.2% | 8.5% |

The before/after on full images is for false alarms only. An image-level recall comparison for the unmined model was not run.

### T4. Final image-level results

Model: mined Random Forest on HOG + colour + LBP. Image score = max window score. Threshold chosen on validation to reach at least 95% recall: **0.35**, frozen before the test run.

| | Validation (1,144 images) | **Test (570 images)** |
|---|---|---|
| Recall | 0.950 | **0.955** |
| Precision | 0.937 | 0.939 |
| F1 | n/a | 0.947 |
| F2 | 0.948 | 0.952 |
| Accuracy | 0.904 | 0.909 |
| TP / FP / FN / TN | 919 / 62 / 48 / 115 | 464 / 30 / 22 / 54 |
| False alarms on human-free images | 35.0% | 35.7% |

Charts: `chart_val_image_level.png` (PR curve and recall vs false-alarm rate), `chart_test_confusion.png`.

Validation operating points (the recall/false-alarm trade-off):

| Threshold | Recall | Precision | Human-free images flagged |
|---|---|---|---|
| 0.50 | 0.886 | 0.978 | 10.7% |
| 0.473 | 0.901 | 0.973 | 13.6% |
| **0.350** | **0.950** | **0.937** | **35.0%** |
| 0.287 | 0.981 | 0.910 | 53.1% |
| 0.220 | 0.992 | 0.877 | 76.3% |

About 85% of images contain a person, so image-level precision looks high even for a weak detector. The false-alarm rate on human-free images is the more honest cost. Only 177 validation and 84 test images are human-free, so that rate is noisy.

### Error analysis (test)

Galleries: `errors_missed.png` (22 images), `errors_false_alarms.png` (30). Green = real person, red = highest-scoring window.

**Missed people**
- Smaller: median longer side 48 px in missed images vs 83 px overall. 48% are under 48 px (11% overall).
- More lying-down people: 30% of misses vs 12% overall (tall 1:2 windows suit standing people).
- By eye: dense dry grass, dark forest or deep shadow, dense foliage.
- Many near-misses: scores 0.30 to 0.34, just under the threshold, with the red window often on the person.
- Some people sit at the image edge, and some misses are augmented copies of the same photo.

**False alarms**
- By eye: rock piles (most common), vehicles, water edges and reflections, shadowy vegetation and tracks.
- Scores are low (median 0.43, just above the 0.35 threshold). Only one is high (0.79, a dirt path).

## Limitations

- Image-level detection only: a flagged image may be flagged by the wrong window. Box-level matching (IoU, NMS) was not done.
- Small and hidden people are the main weakness. About 1 in 20 images with a person is still missed, and about 1 in 3 human-free images is falsely flagged at the chosen threshold.
- Train and test contain augmented copies of the same photos. Splits don't share photos, but the effective sample size is smaller than the image counts.
- Tuning was limited (12 to 25 trials, subsampled). Results are from a single seed; variance across seeds was not measured.
- Only SARD 1 was used. SARD 2 and SeaDronesSee (generalisation tests) were not run.
- Framing: this is **decision support for a human reviewer, not a replacement**.

## Not done yet

Bounding boxes with NMS and IoU matching; SARD 2 and SeaDronesSee tests; an image-level recall comparison for the unmined model; a second mining round; results across several seeds.
