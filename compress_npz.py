# scripts/compress_npz.py  (run from the main ML folder)
import numpy as np
from pathlib import Path

LIMIT = 100 * 1e6   # 100 MB in bytes

folder = Path("patches, features and scores (numpy arrays)")
out = folder / "compressed"
out.mkdir(exist_ok=True)

for f in sorted(folder.glob("*.npz")):
    size = f.stat().st_size
    if size <= LIMIT:
        print(f"skip  {f.name} ({size / 1e6:.0f} MB)")
        continue

    data = np.load(f, allow_pickle=False)
    arrays = {}
    for k in data.files:
        a = data[k]
        if a.dtype == np.float64:          # halve float size
            a = a.astype(np.float32)
        arrays[k] = a
    np.savez_compressed(out / f.name, **arrays)

    after = (out / f.name).stat().st_size
    print(f"done  {f.name}: {size / 1e6:.0f} MB -> {after / 1e6:.0f} MB")