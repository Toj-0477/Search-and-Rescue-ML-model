# Step 8a: run the sliding window over every image of a split and save all window scores
# usage: python step8_infer.py valid
import os
import sys
import time
import joblib
import numpy as np
import cv2
from multiprocessing import Pool
from sliding import make_windows, window_features

STRIDE = 0.375
N_PROC = 5
MODEL = "models/model_rf_mined.joblib"
wins = make_windows(stride_frac=STRIDE)
scaler = model = None


def init():
    global scaler, model
    scaler, model = joblib.load(MODEL)
    model.n_jobs = 1


def scan(path):
    img = cv2.imread(path)
    X = scaler.transform(window_features(img, wins))
    return model.predict_proba(X)[:, 1].astype(np.float32)


if __name__ == "__main__":
    split = sys.argv[1]
    names = sorted(os.listdir(f"SARD1/{split}/images"))
    paths = [f"SARD1/{split}/images/{n}" for n in names]
    t = time.time()
    scores = []
    with Pool(N_PROC, initializer=init) as pool:
        for i, s in enumerate(pool.imap(scan, paths, chunksize=4)):
            scores.append(s)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(paths)} images, {time.time()-t:.0f}s", flush=True)
    np.savez_compressed(f"patches, features and scores (numpy arrays)/scores_{split}.npz", scores=np.array(scores), names=np.array(names), wins=wins)
    print(f"saved scores_{split}.npz ({len(paths)} images x {len(wins)} windows) in {time.time()-t:.0f}s")
