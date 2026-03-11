from pathlib import Path
import cv2
import numpy as np
import time
import torch
import pandas as pd
import torch.nn as nn
import json
import re
import plotly.express as px

from helpers import (
    angular_spectrum,
    get_device, load_model, read_kernel,
    detect_centers_from_img, crop_patch,
    color_to_bgr, draw_box_label
)

# -------------------------
# CONFIG (EDIT THESE)
# -------------------------
VIDEO_PATH = Path("./dataset/environmental_data/output.mp4")      # <-- change
OUT_MP4    = Path("./dataset/environmental_data/COC.mp4")

EXP_DIR     = Path("./dataset/experiments/cnn_20260224_213841")  # <-- change
KERNEL_PATH = Path("./kernel.png")

# Detection + patches + prediction
POL_SHOW   = "M"                 # which channel to detect on
POL_ORDER  = ("H","M","V","P")   # MUST match training order
HALF       = 100                 # patch half-size => 200x200 at channel resolution
THR        = 0.1
NMS_K      = 15
CONF_THR   = 0.6

MAX_FRAMES   = None              # e.g. 300 for quick test
FRAME_STRIDE = 1                 # e.g. 2 to process every 2nd frame

device = get_device()
print("Device:", device)

model, idx2label, ckpt = load_model(EXP_DIR, device=device)
model.eval()

kernel = read_kernel(KERNEL_PATH)

classes = [idx2label[i] for i in sorted(idx2label.keys())]
print("Classes:", classes)

palette = [
    "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)", "rgb(214,39,40)",
    "rgb(148,103,189)", "rgb(140,86,75)", "rgb(227,119,194)", "rgb(127,127,127)",
    "rgb(188,189,34)", "rgb(23,190,207)"
]
color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}

# Warm-up (important for timing on MPS)
with torch.no_grad():
    dummy = torch.zeros((1, int(ckpt["in_channels"]), 2*HALF, 2*HALF), device=device)
    _ = model(dummy)

print("Warm-up done.")

def split_polarsens_mosaic(frame_gray):
    """
    POLARSENS: 2x2 micro-polarizer mosaic -> 4 channels (each H/2 x W/2)

    Mapping used here:
      [0°, 45°]
      [90°,135°]

    We map names as:
      H = 0°
      M = 45°
      V = 90°
      P = 135°
    """
    I0   = frame_gray[0::2, 0::2]
    I45  = frame_gray[0::2, 1::2]
    I135  = frame_gray[1::2, 0::2]
    I90 = frame_gray[1::2, 1::2]
    return {"H": I0, "M": I45, "V": I90, "P": I135}

cap = cv2.VideoCapture(str(VIDEO_PATH))
assert cap.isOpened()
cap.set(cv2.CAP_PROP_POS_FRAMES, 6)
ok, frame0 = cap.read()
cap.release()
gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
frame0_ =  split_polarsens_mosaic(gray0)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Input:", (W, H), "fps:", fps, "frames:", N)

# Output: we write the original frame size (full-res)
out_fps = fps / FRAME_STRIDE if FRAME_STRIDE > 1 else fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_MP4), fourcc, out_fps, (W, H))
if not writer.isOpened():
    raise RuntimeError("Could not open VideoWriter. Try another output path.")

# timing breakdown
t0 = time.perf_counter()
frames_written = 0
frames_seen = 0

t_read = t_split = t_detect = t_crop = t_model = t_draw = 0.0
frame_class_counts = []
while True:
    # stride (skip frames but keep stream position)
    if FRAME_STRIDE > 1 and (frames_seen % FRAME_STRIDE != 0):
        ok = cap.grab()
        if not ok:
            break
        frames_seen += 1
        continue

    tr0 = time.perf_counter()
    ok, frame = cap.read()
    t_read += time.perf_counter() - tr0
    if not ok:
        break

    frames_seen += 1
    if MAX_FRAMES is not None and frames_written >= MAX_FRAMES:
        break

    # grayscale
    if frame.ndim == 3:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_full = frame


    ts0 = time.perf_counter()
    ch = split_polarsens_mosaic(gray_full)    # each channel: (H/2, W/2)
    t_split += time.perf_counter() - ts0

    # choose channel to detect on
    disp = ch[POL_SHOW]
    # disp_u8 = disp if disp.dtype == np.uint8 else cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # detect centers (in HALF-RES coordinates!)
    td0 = time.perf_counter()
    # c = frame0['M']-ch['M']
    # c = c - c.min()
    ch['H'] = frame0_['H'] - ch['H']
    ch['V'] = frame0_['V'] - ch['V']
    ch['M'] = frame0_['M'] - ch['M']
    ch['P'] = frame0_['P'] - ch['P']
    ch['H'] = ch['H'] - ch['H'].min()
    ch['V'] = ch['V'] - ch['V'].min()
    ch['P'] = ch['P'] - ch['P'].min()
    ch['M'] = ch['M'] - ch['M'].min()
    prop = np.abs(angular_spectrum(12e-2, ch['M'], 532e-9, 3.45e-6 * 2)) ** 1
    prop = 1 - prop / prop.max()
    centers = detect_centers_from_img(prop, kernel, thr=THR, nms_k=NMS_K)
    t_detect += time.perf_counter() - td0

    # ---- PATCH EXTRACTION (4 channels) ----
    tc0 = time.perf_counter()
    coords = []
    X = []
    for (cx, cy, score) in centers:
        patch = np.stack([crop_patch(ch[p], cx, cy, HALF) for p in POL_ORDER], axis=0)  # (4,2h,2h)
        X.append(patch)
        coords.append((cx, cy))
    t_crop += time.perf_counter() - tc0

    # ---- Batch predict ----
    tm0 = time.perf_counter()
    pred_labels = []
    pred_confs = []

    if len(X) > 0:
        X = np.stack(X, axis=0).astype(np.float32)  # (N,4,2h,2h)
        if X.max() > 1.5:
            X /= 255.0

        xb = torch.from_numpy(X).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()

        pred_idx = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)

        pred_labels = [idx2label[int(i)] for i in pred_idx]
        pred_confs = pred_conf.tolist()
        # Count only predictions above confidence threshold

        valid_labels = [lab for lab, conf in zip(pred_labels, pred_confs) if conf >= CONF_THR]

        counts_dict = {cls: 0 for cls in classes}
        for lab in valid_labels:
            counts_dict[lab] += 1

        frame_class_counts.append({
            "frame_idx": frames_written,
            **counts_dict
        })

    t_model += time.perf_counter() - tm0

    # ---- Draw overlay on the ORIGINAL FULL-RES frame ----
    # out = frame.copy()
    # out[]

    out = frame0.astype(np.float32) - frame.astype(np.float32)
    out = out - out.min()
    out = out/out.max() * 255
    out = out.astype(np.uint8)

    tdr0 = time.perf_counter()
    for (cx, cy), lab, conf in zip(coords, pred_labels, pred_confs):
        if conf < CONF_THR:
            continue

        # scale HALF-RES coords back to full-res
        cx_full = int(cx * 2)
        cy_full = int(cy * 2)
        half_full = int(HALF * 2)

        bgr = color_to_bgr(color_map[lab])
        draw_box_label(out, cx_full, cy_full, half_full, f"{lab} {conf:.2f}", bgr, thickness=2)

    t_draw += time.perf_counter() - tdr0

    writer.write(out)
    frames_written += 1

    if frames_written % 25 == 0:
        print(f"Processed {frames_written} frames | detections last frame: {len(centers)}")

cap.release()
writer.release()

total = time.perf_counter() - t0
timing = {
    "frames_written": frames_written,
    "total_s": total,
    "fps_effective": frames_written / total if total > 0 else 0,
    "t_read_s": t_read,
    "t_split_s": t_split,
    "t_detect_s": t_detect,
    "t_crop_s": t_crop,
    "t_model_s": t_model,
    "t_draw_s": t_draw,
    "avg_ms_total": 1000 * total / max(1, frames_written),
    "avg_ms_detect": 1000 * t_detect / max(1, frames_written),
    "avg_ms_model": 1000 * t_model / max(1, frames_written),
}

print("\nSaved:", OUT_MP4.resolve())
timing
counts_df = pd.DataFrame(frame_class_counts)
# counts_df.head()