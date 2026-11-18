# Step 3: turn each patch into numbers (HOG, colour histogram, LBP)
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from joblib import Parallel, delayed


def hog_feat(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys")


def color_feat(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    return hist / hist.sum()


def lbp_feat(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")   # values 0..9
    hist = np.bincount(lbp.astype(int).ravel(), minlength=10).astype(float)
    return hist / hist.sum()


def all_feats(patch):
    return hog_feat(patch), color_feat(patch), lbp_feat(patch)


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        d = np.load(f"patches, features and scores (numpy arrays)/patches_{split}.npz")
        X, y = d["X"], d["y"]
        out = Parallel(n_jobs=4)(delayed(all_feats)(p) for p in X)
        hog_f = np.array([o[0] for o in out])
        col_f = np.array([o[1] for o in out])
        lbp_f = np.array([o[2] for o in out])
        # F1 = HOG, F2 = HOG+colour, F3 = HOG+LBP, F4 = all three
        np.savez_compressed(f"patches, features and scores (numpy arrays)/features_{split}.npz", hog=hog_f, color=col_f, lbp=lbp_f, y=y)
        print(f"{split}: HOG {hog_f.shape[1]} + colour {col_f.shape[1]} + LBP {lbp_f.shape[1]} "
              f"features for {len(y)} patches")
