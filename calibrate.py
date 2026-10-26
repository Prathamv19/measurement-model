"""Learn a simple correction from your own data (also used by app.py).
Command line:  python calibrate.py data/training.csv
Needs columns raw_<name> and tape_<name>; writes calibration.json, which measure.py uses automatically."""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

NAMES = ["shoulder", "sleeve", "inseam", "outseam", "chest", "waist", "hip"]
CALIB_PATH = Path(__file__).with_name("calibration.json")
MIN_ROWS = 30


def train(csv_path):
    """Fit y = a*x + b per measurement. Returns report rows. Error is checked on 20% of
    customers that were left out of the fit, so it is an honest estimate."""
    rows = list(csv.DictReader(open(csv_path))) if Path(csv_path).exists() else []
    cal, report = {}, []
    for n in NAMES:
        pairs = [(float(r[f"raw_{n}"]), float(r[f"tape_{n}"]))
                 for r in rows if r.get(f"raw_{n}") and r.get(f"tape_{n}")]
        if len(pairs) < MIN_ROWS:
            report.append({"measurement": n, "customers": len(pairs), "status": f"need {MIN_ROWS - len(pairs)} more"})
            continue
        random.Random(0).shuffle(pairs)
        cut = int(len(pairs) * 0.8)
        fit, test = np.array(pairs[:cut]).T, np.array(pairs[cut:]).T
        a, b = np.polyfit(fit[0], fit[1], 1)
        before = float(np.mean(abs(test[0] - test[1])))
        after = float(np.mean(abs(a * test[0] + b - test[1])))
        if after < before:  # only keep a correction that really helps on unseen customers
            a, b = np.polyfit(np.concatenate([fit[0], test[0]]), np.concatenate([fit[1], test[1]]), 1)
            cal[n] = [float(a), float(b)]
        report.append({"measurement": n, "customers": len(pairs), "error before (cm)": round(before, 2),
                       "error after (cm)": round(after, 2), "status": "correction saved" if n in cal else "no gain, not used"})
    CALIB_PATH.write_text(json.dumps(cal, indent=2))
    return report


if __name__ == "__main__":
    for r in train(sys.argv[1] if len(sys.argv) > 1 else "data/training.csv"):
        print(r)
