# Step 10: error analysis - look at the images the system got wrong
import os
import numpy as np
import cv2

th = float(np.load("patches, features and scores (numpy arrays)/threshold.npy")[0])
sc = np.load("patches, features and scores (numpy arrays)/scores_test.npz")
err = np.load("patches, features and scores (numpy arrays)/test_errors.npz")
names, scores, wins = list(sc["names"]), sc["scores"], sc["wins"]


def boxes(name):
    out = []
    for line in open(f"SARD1/test/labels/{os.path.splitext(name)[0]}.txt").read().strip().splitlines():
        if line.strip():
            _, cx, cy, w, h = map(float, line.split()[:5])
            out.append((int((cx - w/2) * 640), int((cy - h/2) * 640), int((cx + w/2) * 640), int((cy + h/2) * 640)))
    return out


def draw(name):
    img = cv2.imread(f"SARD1/test/images/{name}")
    i = names.index(name)
    for x1, y1, x2, y2 in boxes(name):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)           # ground truth: green
    best = scores[i].argmax()
    x1, y1, x2, y2 = wins[best]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)               # best window: red
    cv2.putText(img, f"max {scores[i].max():.2f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return cv2.resize(img, (320, 320))


def gallery(file_names, out, cols=6):
    tiles = [draw(n) for n in file_names]
    while len(tiles) % cols:
        tiles.append(np.zeros((320, 320, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows))


fn_names, fp_names = list(err["fn"]), list(err["fp"])
gallery(fn_names, "charts/errors_missed.png")
gallery(fp_names, "charts/errors_false_alarms.png")

# numbers: how big are the people in missed images compared with all test images?
def sizes(ns):
    b = [bx for n in ns for bx in boxes(n)]
    w = np.array([x2 - x1 for x1, _, x2, _ in b]); h = np.array([y2 - y1 for _, y1, _, y2 in b])
    return w, h

w_all, h_all = sizes([n for n in names if os.path.getsize(f"SARD1/test/labels/{os.path.splitext(n)[0]}.txt") > 0])
w_fn, h_fn = sizes(fn_names)
print(f"{len(fn_names)} missed images, {len(fp_names)} false-alarm images")
print("Person size (longer side, px): all test median %.0f | missed images median %.0f" %
      (np.median(np.maximum(w_all, h_all)), np.median(np.maximum(w_fn, h_fn))))
print("Share of people with longer side < 48 px: all %.0f%% | missed images %.0f%%" %
      (100 * (np.maximum(w_all, h_all) < 48).mean(), 100 * (np.maximum(w_fn, h_fn) < 48).mean()))
print("Share lying down (wider than tall): all %.0f%% | missed images %.0f%%" %
      (100 * (w_all > h_all).mean(), 100 * (w_fn > h_fn).mean()))

print("\nMissed images (name, #people, longest side px, max score):")
for n in fn_names:
    b = boxes(n)
    big = max(max(x2 - x1, y2 - y1) for x1, y1, x2, y2 in b)
    print(f"  {n[:28]:28s} people {len(b)}  size {big:3d}  score {scores[names.index(n)].max():.2f}")
print("\nFalse alarms: max score median %.2f" % np.median([scores[names.index(n)].max() for n in fp_names]))
print("saved errors_missed.png and errors_false_alarms.png")
