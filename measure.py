"""Photo-based body measurements from one front photo and (optionally) one side photo.

STARTER CODE for your ML developer. The geometry has been checked on synthetic
silhouettes only; it has NOT been tested on real photos yet.

Usage:  python measure.py front.jpg 175 [side.jpg]     (175 = customer height in cm)
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_PATH = Path(__file__).with_name("pose_landmarker_heavy.task")
CALIB_PATH = Path(__file__).with_name("calibration.json")

# MediaPipe pose landmark indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_ANK, R_ANK = 23, 24, 27, 28
KEY_POINTS = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_ANK, R_ANK]

# Where each measurement is taken. These are starting guesses: tune them with real data.
CHEST_AT = 0.40             # fraction of the way down from the shoulder line to the hip-joint line
WAIST_RANGE = (0.55, 0.85)  # the narrowest torso row is searched in this range
HIP_DROP = 0.04             # fullest hip row: this fraction of body height below the hip-joint line
DEPTH_RATIO = 0.70          # body depth / width, used only when there is no side photo
TORSO_CLIP = 0.58           # torso half-width limit as a fraction of shoulder-joint spacing (ignores arms)


# ---------- step 1: find the person (needs mediapipe) ----------
_landmarker = None


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        if not MODEL_PATH.exists():
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)  # one-time download
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def analyze(src):
    """src: file path or file-like object. Returns landmarks (px), visibility, body mask."""
    import mediapipe as mp
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")  # fixes phone rotation
    img.thumbnail((1280, 1280))
    arr = np.ascontiguousarray(np.asarray(img))
    result = get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not result.pose_landmarks:
        raise ValueError("No person found. Retake the photo with the whole body in view.")
    h, w = arr.shape[:2]
    pts = result.pose_landmarks[0]
    lm = np.array([[p.x * w, p.y * h] for p in pts])
    vis = np.array([p.visibility for p in pts])
    mask = np.squeeze(result.segmentation_masks[0].numpy_view()) > 0.5
    return lm, vis, mask, img


def draw_overlay(img, lm, mask, marks):
    """Debug picture: body tinted pink, measurement rows in yellow, joints in blue."""
    from PIL import ImageDraw
    base = np.asarray(img).copy()
    base[mask] = (0.6 * base[mask] + 0.4 * np.array([214, 51, 108])).astype(np.uint8)
    out = Image.fromarray(base)
    g = ImageDraw.Draw(out)
    for name, y in marks.items():
        if y is not None:
            g.line([(0, y), (out.width, y)], fill=(255, 200, 0), width=2)
            g.text((6, y - 12), name, fill=(255, 200, 0))
    for i in KEY_POINTS:
        x, y = lm[i]
        g.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 200, 255))
    return out


# ---------- step 2: geometry on the body mask (numpy only) ----------
def runs(row):
    """(start, end) of every run of True values in a 1-D boolean array."""
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def body_rows(mask):
    ys = np.where(mask.any(axis=1))[0]
    return int(ys[0]), int(ys[-1])


def central_width(mask, y, cx, half):
    """Width (px) of the body run nearest column cx on row y, clipped to cx +/- half.
    Clipping stops arms hanging next to the torso from inflating the width."""
    row = mask[int(round(y))]
    found = runs(row)
    if not found:
        return 0
    a, b = min(found, key=lambda r: 0 if r[0] <= cx <= r[1] else min(abs(r[0] - cx), abs(r[1] - cx)))
    return int(min(b, cx + half) - max(a, cx - half) + 1)


def ellipse(width, depth):
    a, b = width / 2, depth / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))  # Ramanujan


def front_raw(lm, vis, mask, height_cm, warnings):
    img_h = mask.shape[0]
    top, bottom = body_rows(mask)
    H = bottom - top
    k = height_cm / H  # cm per pixel

    if top < 0.01 * img_h or bottom > 0.99 * img_h:
        warnings.append("Head or feet are cut off. Step back so the whole body fits.")
    if H < 0.55 * img_h:
        warnings.append("The person looks small in the frame. Move closer.")
    if vis[KEY_POINTS].min() < 0.5:
        warnings.append("Some joints are hidden. Wear fitted clothes and hold arms slightly away from the body.")

    sh_px = abs(lm[L_SH][0] - lm[R_SH][0])
    if abs(lm[L_SH][1] - lm[R_SH][1]) > 0.15 * sh_px:
        warnings.append("Shoulders look tilted. Stand straight and hold the phone level.")

    sh_y = lm[[L_SH, R_SH], 1].mean()
    hip_y = lm[[L_HIP, R_HIP], 1].mean()
    span = hip_y - sh_y
    cx = int(round(lm[[L_HIP, R_HIP], 0].mean()))
    half = TORSO_CLIP * sh_px

    chest_y = sh_y + CHEST_AT * span
    ys = np.arange(int(sh_y + WAIST_RANGE[0] * span), int(sh_y + WAIST_RANGE[1] * span) + 1)
    ws = np.array([central_width(mask, y, cx, half) for y in ys], float)
    ws[ws == 0] = np.inf
    waist_y = ys[int(np.argmin(ws))]
    hip_m_y = hip_y + HIP_DROP * H

    widths = {n: central_width(mask, y, cx, half) * k
              for n, y in (("chest", chest_y), ("waist", waist_y), ("hip", hip_m_y))}
    fracs = {"chest": CHEST_AT, "waist": (waist_y - sh_y) / span, "hip": (hip_m_y - sh_y) / span}

    marks = {"head top": top, "chest": chest_y, "waist": waist_y, "hip": hip_m_y, "crotch": None, "feet": bottom}
    arm = lambda s, e, w: math.dist(lm[s], lm[e]) + math.dist(lm[e], lm[w])
    raw = {
        "shoulder": sh_px * k,
        "sleeve": (arm(L_SH, L_EL, L_WR) + arm(R_SH, R_EL, R_WR)) / 2 * k,
        "outseam": (bottom - waist_y) * k,
        "inseam": None,
    }
    # crotch = first row below the hips where the body splits into two legs
    for y in range(int(hip_m_y), bottom):
        legs = [r for r in runs(mask[y]) if abs((r[0] + r[1]) / 2 - cx) < half]
        if len(legs) >= 2:
            raw["inseam"] = (bottom - y) * k
            marks["crotch"] = y
            break
    if raw["inseam"] is None:
        warnings.append("Could not see the gap between the legs. Stand with feet shoulder-width apart.")
    return raw, widths, fracs, marks


def side_depths(lm, mask, height_cm, fracs):
    top, bottom = body_rows(mask)
    ks = height_cm / (bottom - top)
    sh_y = lm[[L_SH, R_SH], 1].mean()
    hip_y = lm[[L_HIP, R_HIP], 1].mean()
    cx = int(round(lm[[L_HIP, R_HIP], 0].mean()))
    rows = {n: sh_y + f * (hip_y - sh_y) for n, f in fracs.items()}
    depths = {n: central_width(mask, y, cx, bottom - top) * ks for n, y in rows.items()}
    return depths, {"head top": top, "feet": bottom, **rows}


# ---------- step 3: correct systematic error with your own tape data ----------
def apply_calibration(raw):
    cal = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else {}
    return {n: (None if v is None else (cal[n][0] * v + cal[n][1] if n in cal else v))
            for n, v in raw.items()}


def measure(front, height_cm, side=None, debug=False):
    warnings = []
    lm, vis, mask, img = analyze(front)
    raw, widths, fracs, marks = front_raw(lm, vis, mask, height_cm, warnings)
    overlays = {"front": draw_overlay(img, lm, mask, marks) if debug else None, "side": None}
    depths = None
    if side is not None:
        slm, _, smask, simg = analyze(side)
        depths, smarks = side_depths(slm, smask, height_cm, fracs)
        if debug:
            overlays["side"] = draw_overlay(simg, slm, smask, smarks)
        if min(depths.values()) <= 0:
            depths = None
            warnings.append("Side photo could not be used. Using an estimated body depth instead.")
    if depths is None:
        depths = {n: w * DEPTH_RATIO for n, w in widths.items()}
        if side is None:
            warnings.append("No side photo: circumferences are less accurate.")
    for n in ("chest", "waist", "hip"):
        raw[n] = ellipse(widths[n], depths[n])
    final = apply_calibration(raw)
    r1 = lambda d: {n: None if v is None else round(v, 1) for n, v in d.items()}
    out = {"measurements_cm": r1(final), "raw_cm": r1(raw),
           "used_side_photo": side is not None and "Side photo could not be used" not in " ".join(warnings),
           "warnings": warnings}
    if debug:
        out["overlays"] = overlays  # pictures, not JSON: only used by app.py
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python measure.py front.jpg HEIGHT_CM [side.jpg]")
    out = measure(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
    print(json.dumps(out, indent=2))
