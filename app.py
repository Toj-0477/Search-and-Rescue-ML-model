# Web app: pick images (or a folder of images) and find the people in them.
# Run from the ML folder:  streamlit run app.py
import os
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd
import cv2
import streamlit as st

sys.path.insert(0, "scripts")
from sliding import make_windows
from parallel import init, score_chunk
from detect import fit_to_square, windows_inside, draw_result

MODEL_PATH = "models/model_rf_mined.joblib"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")

st.set_page_config(page_title="SAR Human Detector", layout="wide")


def default_threshold():
    try:
        return float(np.load("patches, features and scores (numpy arrays)/threshold.npy")[0])
    except Exception:
        return 0.35


@st.cache_resource
def get_pool(n_workers):
    """One pool of worker processes, each holding its own copy of the model. Kept between button clicks."""
    ctx = multiprocessing.get_context("spawn")
    return ctx.Pool(n_workers, initializer=init, initargs=(MODEL_PATH,))


st.title("Human detection in aerial search-and-rescue images")
st.write("Classical ML only: HOG + colour + LBP features and a Random Forest, applied with a sliding window. "
         "An image is flagged when its best window score reaches the threshold. "
         "**Decision support for a human reviewer, not a replacement.**")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection threshold", 0.10, 0.80, default_threshold(), 0.01,
                          help="Lower = finds more people but more false alarms. 0.35 gives about 95% recall on our test set.")
    speed = st.radio("Scan", ["Fast", "Thorough"], index=1,
                     help="Thorough checks more window positions (about 1,137 per image vs 668).")
    stride = 0.5 if speed == "Fast" else 0.375
    n_cpu = os.cpu_count() or 1
    n_workers = st.slider("CPU workers", 1, n_cpu, n_cpu,
                          help=f"Your computer has {n_cpu} logical cores. All of them are used by default.")
    source = st.radio("Images from", ["Upload files", "Folder on this computer"])

images = []   # (name, BGR array)
if source == "Upload files":
    files = st.file_uploader("Choose one or more images", type=["jpg", "jpeg", "png", "bmp"],
                             accept_multiple_files=True)
    for f in files or []:
        arr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            images.append((f.name, arr))
else:
    folder = st.text_input("Folder path", placeholder=r"C:\Users\you\Desktop\ML\SARD1\test\images")
    if folder:
        if os.path.isdir(folder):
            for n in sorted(os.listdir(folder)):
                if n.lower().endswith(IMAGE_TYPES):
                    arr = cv2.imread(os.path.join(folder, n))
                    if arr is not None:
                        images.append((n, arr))
            st.caption(f"{len(images)} images found")
        else:
            st.error("That folder doesn't exist.")

if images and st.button(f"Process {len(images)} image(s)", type="primary"):
    t0 = time.time()
    bar = st.progress(0.0, text=f"Starting {n_workers} workers (the first run loads the model in each)...")
    pool = get_pool(n_workers)
    all_wins = make_windows(stride_frac=stride)

    # Scale each image to fit 640 x 640 and scan only its real area (never the padding).
    squares, valid, img_wins = [], [], []
    for _, img in images:
        sq, vw, vh = fit_to_square(img)
        squares.append(sq)
        valid.append((vw, vh))
        img_wins.append(windows_inside(all_wins, vw, vh))

    # one task = one image, or a slice of one image's windows when there are fewer images than workers
    n_slices = 1 if len(images) >= n_workers else max(1, n_workers // len(images))
    tasks, n_parts = [], []
    for sq, w_i in zip(squares, img_wins):
        pieces = [p for p in np.array_split(w_i, n_slices) if len(p)]
        n_parts.append(len(pieces))
        tasks += [(sq, p) for p in pieces]

    parts = []
    for k, scores in enumerate(pool.imap(score_chunk, tasks, chunksize=1)):
        parts.append(scores)
        bar.progress((k + 1) / len(tasks), text=f"Scanned {k + 1} of {len(tasks)} chunks")
    elapsed = time.time() - t0
    bar.progress(1.0, text=f"Done in {elapsed:.1f} s ({elapsed / len(images):.2f} s per image)")

    results, shown = [], []
    start = 0
    for i, (name, _) in enumerate(images):
        if n_parts[i] == 0:        # picture too small for even the smallest window
            results.append({"image": name, "best score": 0.0, "result": "too small"})
            continue
        scores = np.concatenate(parts[start:start + n_parts[i]])
        start += n_parts[i]
        vw, vh = valid[i]
        best, flagged, vis = draw_result(squares[i], img_wins[i], scores, threshold)
        vis = vis[:vh, :vw]        # show only the real picture, not the padding
        results.append({"image": name, "best score": round(best, 3),
                        "result": "HUMAN FOUND" if flagged else "no human"})
        shown.append((name, best, flagged, vis))

    df = pd.DataFrame(results)
    n_flag = int((df["result"] == "HUMAN FOUND").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Images processed", len(df))
    c2.metric("Flagged (human found)", n_flag)
    c3.metric("Not flagged", len(df) - n_flag)

    st.download_button("Download results (CSV)", df.to_csv(index=False), "results.csv", "text/csv")
    st.dataframe(df, width="stretch", hide_index=True)

    st.subheader("Images")
    shown.sort(key=lambda t: -t[1])           # most likely humans first
    cols = st.columns(3)
    for i, (name, best, flagged, vis) in enumerate(shown):
        with cols[i % 3]:
            st.image(vis, caption=f"{name}: {'HUMAN FOUND' if flagged else 'no human'} (score {best:.2f})")
