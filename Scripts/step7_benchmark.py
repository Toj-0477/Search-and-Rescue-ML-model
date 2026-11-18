# Step 7a: how long does one window take? (decides stride and scales)
import os
import time
import joblib
import numpy as np
import cv2
from sliding import make_windows, window_features

t = time.time()
scaler, model = joblib.load("models/model_rf_default.joblib")
print(f"model load {time.time()-t:.1f}s, file {os.path.getsize('models/model_rf_default.joblib')/1e6:.0f} MB")
model.n_jobs = 1

name = sorted(os.listdir("SARD1/valid/images"))[0]
img = cv2.imread(f"SARD1/valid/images/{name}")

for frac in (0.5, 0.375, 0.25):
    wins = make_windows(stride_frac=frac)
    t = time.time()
    X = scaler.transform(window_features(img, wins))
    t_feat = time.time() - t
    t = time.time()
    model.predict_proba(X)
    t_pred = time.time() - t
    print(f"stride {frac}: {len(wins)} windows/image, features {t_feat:.1f}s, RF predict {t_pred:.1f}s "
          f"per image (1 core)")
