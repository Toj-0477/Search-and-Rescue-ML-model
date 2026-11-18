# Step 7b: hard-negative mining
# 1. run the main model over train images that contain NO humans -> every window it flags is a false alarm
# 2. add those windows to the negatives and retrain the Random Forest
# 3. compare before/after: patch metrics on validation + false alarms on human-free validation images
import os
import time
import joblib
import numpy as np
import cv2
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, fbeta_score, accuracy_score
from sliding import make_windows, window_features

SEED = 42
STRIDE = 0.375
N_MINE_IMAGES = 400     # human-free train images to scan
THRESH = 0.3            # windows scoring above this count as false alarms
MAX_PER_IMAGE = 10
N_PROC = 5

wins = make_windows(stride_frac=STRIDE)
scaler = model = None


def init(path):
    global scaler, model
    scaler, model = joblib.load(path)
    model.n_jobs = 1


def scan(path):
    """Return (raw features, scores) for every window of one image."""
    img = cv2.imread(path)
    X = window_features(img, wins)
    return X, model.predict_proba(scaler.transform(X))[:, 1]


def empty_images(split):
    lab = f"SARD1/{split}/labels"
    return [f for f in sorted(os.listdir(lab)) if os.path.getsize(f"{lab}/{f}") == 0]


def img_path(split, label_name):
    d = f"SARD1/{split}/images"
    stem = os.path.splitext(label_name)[0]
    return f"{d}/{stem}.jpg"


if __name__ == "__main__":
    scaler, _ = joblib.load("models/model_rf_default.joblib")

    if os.path.exists("patches, features and scores (numpy arrays)/hard_negatives.npy"):      # mining was already done in an earlier run
        hard = np.load("patches, features and scores (numpy arrays)/hard_negatives.npy")
        print(f"loaded {len(hard)} saved hard negatives", flush=True)
    else:
        # one image per original photo (train has ~3 augmented copies of each photo)
        seen, picked = set(), []
        for f in empty_images("train"):
            base = f.split("_jpg.rf.")[0]
            if base not in seen:
                seen.add(base)
                picked.append(img_path("train", f))
        rng = np.random.default_rng(SEED)
        picked = [picked[i] for i in rng.permutation(len(picked))[:N_MINE_IMAGES]]
        print(f"{len(picked)} human-free train photos to mine, {len(wins)} windows each", flush=True)

        t = time.time()
        with Pool(N_PROC, initializer=init, initargs=("models/model_rf_default.joblib",)) as pool:
            res = pool.map(scan, picked, chunksize=4)
        hard = []
        n_fp = 0
        for X, s in res:
            idx = np.where(s > THRESH)[0]
            n_fp += len(idx)
            idx = idx[np.argsort(-s[idx])][:MAX_PER_IMAGE]
            hard.append(X[idx])
        hard = np.vstack(hard)
        print(f"mined {len(hard)} hard negatives ({n_fp} windows above {THRESH} in total, "
              f"{n_fp/len(picked):.1f} per image) in {time.time()-t:.0f}s", flush=True)
        np.save("patches, features and scores (numpy arrays)/hard_negatives.npy", hard)

    # retrain with the extra negatives (scaler stays fitted on the original train set)
    tr, va = np.load("patches, features and scores (numpy arrays)/features_train.npz"), np.load("patches, features and scores (numpy arrays)/features_valid.npz")
    Xtr = np.hstack([tr["hog"], tr["color"], tr["lbp"]])
    Xva = np.hstack([va["hog"], va["color"], va["lbp"]])
    ytr, yva = tr["y"], va["y"]
    X2 = np.vstack([Xtr, hard])
    y2 = np.concatenate([ytr, np.zeros(len(hard), dtype=np.uint8)])
    t = time.time()
    new = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=6, random_state=SEED)
    new.fit(scaler.transform(X2), y2)
    joblib.dump((scaler, new), "models/model_rf_mined.joblib")
    print(f"retrained in {time.time()-t:.0f}s", flush=True)

    # before / after
    old_scaler, old = joblib.load("models/model_rf_default.joblib")
    print("\nPatch level (validation):")
    for name, m in [("before mining", old), ("after mining", new)]:
        p = m.predict(old_scaler.transform(Xva))
        print(f"  {name:14s} acc {accuracy_score(yva, p):.3f} recall {recall_score(yva, p):.3f} "
              f"precision {precision_score(yva, p):.3f} F2 {fbeta_score(yva, p, beta=2):.3f}")

    val_empty = [img_path("valid", f) for f in empty_images("valid")]
    print(f"\nFalse alarms on {len(val_empty)} human-free validation images "
          f"(windows scoring > 0.5, {len(wins)} windows per image):")
    for name, path in [("before mining", "models/model_rf_default.joblib"), ("after mining", "models/model_rf_mined.joblib")]:
        with Pool(N_PROC, initializer=init, initargs=(path,)) as pool:
            out = pool.map(scan, val_empty, chunksize=4)
        mx = np.array([s.max() for _, s in out])
        cnt = np.array([(s > 0.5).sum() for _, s in out])
        print(f"  {name:14s} windows flagged per image {cnt.mean():.2f}, "
              f"images with at least one flag {(cnt > 0).mean():.1%}, median max score {np.median(mx):.2f}",
              flush=True)
