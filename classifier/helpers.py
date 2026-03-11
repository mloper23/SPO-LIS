# helpers.py
# ------------------------------------------------------------
# Utilities for POAM patch dataset: loading metadata/patches,
# CNN model loading/inference, detection, cropping, and overlays.
# ------------------------------------------------------------

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# =========================
# Device
# =========================
def get_device(prefer_mps: bool = True) -> torch.device:
    """Return best available device for Mac (MPS) / CUDA / CPU."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# Model
# =========================
class SimpleCNN(nn.Module):
    """Same architecture used during training. Keep in sync."""
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


def load_label_mapping(label_map_path: Path) -> Dict[int, str]:
    """Load idx2label mapping saved as JSON."""
    with open(label_map_path, "r") as f:
        tmp = json.load(f)
    # keys are usually strings after json; coerce to int
    return {int(k): v for k, v in tmp.items()}


def load_model(exp_dir: Path, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[int, str], dict]:
    """
    Load model_best.pt + label_mapping.json from exp_dir.
    Returns: (model, idx2label, ckpt_dict)
    """
    exp_dir = Path(exp_dir)
    ckpt_path = exp_dir / "model_best.pt"
    label_map_path = exp_dir / "label_mapping.json"

    if device is None:
        device = get_device()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    idx2label = load_label_mapping(label_map_path)

    model = SimpleCNN(in_ch=int(ckpt["in_channels"]), n_classes=int(ckpt["num_classes"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(device)

    return model, idx2label, ckpt


# =========================
# Metadata + Patch loading
# =========================
def load_metadata(meta_csv: Path, safe: bool = True) -> pd.DataFrame:
    """
    Load metadata.csv. If safe=True, uses python engine and skips bad lines.
    """
    meta_csv = Path(meta_csv)
    if safe:
        return pd.read_csv(meta_csv, engine="python", on_bad_lines="skip")
    return pd.read_csv(meta_csv)


class NPZPatchCache:
    """
    Cached loader for patches saved as .npz with key "patches" (K,C,H,W).
    Many metadata rows point to the same npz; caching avoids re-loading.
    """
    def __init__(self, patches_dir: Path):
        self.patches_dir = Path(patches_dir)
        self._cache: Dict[Path, np.ndarray] = {}

    def resolve(self, patch_file: str | Path) -> Path:
        p = Path(patch_file)
        if p.exists():
            return p
        return self.patches_dir / p.name

    def get_patches(self, patch_file: str | Path) -> np.ndarray:
        p = self.resolve(patch_file)
        if p not in self._cache:
            self._cache[p] = np.load(p, allow_pickle=True)["patches"]
        return self._cache[p]

    def get_patch(self, patch_file: str | Path, patch_index: int) -> np.ndarray:
        patches = self.get_patches(patch_file)
        return patches[int(patch_index)]  # (C,H,W)


def group_df_for_frame(df: pd.DataFrame, row_or_idx) -> pd.DataFrame:
    """
    Return all rows corresponding to the same (root_folder, root_name, file_id, frame_idx)
    as the given row (Series) or index (int).
    """
    if isinstance(row_or_idx, int):
        row = df.iloc[row_or_idx]
    else:
        row = row_or_idx

    return df[
        (df["root_folder"] == row["root_folder"]) &
        (df["root_name"] == row["root_name"]) &
        (df["file_id"] == row["file_id"]) &
        (df["frame_idx"] == row["frame_idx"])
    ].copy().reset_index(drop=True)


# =========================
# Video IO
# =========================
def read_frame(video_file: Path | str, frame_idx: int = 0, grayscale: bool = True) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {video_file}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from: {video_file}")
    if grayscale and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def open_pol_caps(root_folder: Path, root_name: str, file_id: str, pol_states: Iterable[str] = ("H", "M", "V", "P")) -> Dict[str, cv2.VideoCapture]:
    """
    Open synchronized cv2.VideoCapture objects for polarization states.
    """
    caps = {}
    for pol in pol_states:
        vid = Path(root_folder) / f"rec_{root_name}_{file_id}_{pol}.avi"
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open: {vid}")
        caps[pol] = cap
    return caps


def close_caps(caps: Dict[str, cv2.VideoCapture]) -> None:
    for cap in caps.values():
        try:
            cap.release()
        except Exception:
            pass


def get_video_info(cap: cv2.VideoCapture) -> Tuple[int, int, int, float]:
    """
    Returns: (W, H, N_frames, fps)
    """
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, n, fps


# =========================
# Cropping
# =========================
def crop_patch(img: np.ndarray, cx: int, cy: int, half: int, pad_value=0) -> np.ndarray:
    """
    Crop a single patch centered at (cx,cy), size (2*half,2*half), with padding.
    img: (H,W)
    """
    H, W = img.shape
    patch = np.full((2 * half, 2 * half), pad_value, dtype=img.dtype)

    cx = int(cx); cy = int(cy)

    x0 = max(cx - half, 0)
    x1 = min(cx + half, W)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, H)

    x0p = x0 - (cx - half)
    y0p = y0 - (cy - half)
    x1p = x0p + (x1 - x0)
    y1p = y0p + (y1 - y0)

    patch[y0p:y1p, x0p:x1p] = img[y0:y1, x0:x1]
    return patch


# =========================
# Detection
# =========================
def read_kernel(kernel_path: Path | str) -> np.ndarray:
    k = cv2.imread(str(kernel_path), cv2.IMREAD_GRAYSCALE)
    if k is None:
        raise FileNotFoundError(f"Kernel not found at {kernel_path}")
    return k.astype(np.float32)


def detect_centers_from_img(img_for_corr: np.ndarray, kernel: np.ndarray, thr: float = 0.25, nms_k: int = 15) -> List[Tuple[int, int, float]]:
    """
    Template matching + local maxima NMS.
    Returns list of (cx, cy, score) in image coordinates.
    """
    img = img_for_corr - cv2.GaussianBlur(img_for_corr, (0, 0), sigmaX=10)
    tpl = kernel - cv2.GaussianBlur(kernel, (0, 0), sigmaX=10)

    corr = cv2.matchTemplate(img.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)

    dil = cv2.dilate(corr, np.ones((nms_k, nms_k), np.uint8))
    peaks = (corr == dil) & (corr >= thr)

    ys, xs = np.where(peaks)
    scores = corr[ys, xs].astype(np.float32)

    h, w = tpl.shape
    centers = [(int(x + w / 2), int(y + h / 2), float(s)) for x, y, s in zip(xs, ys, scores)]
    return centers


def angular_spectrum(z: float, field: np.ndarray, wavelength: float, pixel_pitch_in: float) -> np.ndarray:
    """
    Angular spectrum propagation (your pyDHM-style function).
    field: complex or real (M,N)
    """
    M, N = np.shape(field)
    x = np.arange(0, N, 1)
    y = np.arange(0, M, 1)
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing="xy")
    dfx = 1 / (pixel_pitch_in * M)
    dfy = 1 / (pixel_pitch_in * N)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    phase = np.exp2(
        1j * z * np.pi * np.sqrt((1 / wavelength) ** 2 - ((X * dfx) ** 2 + (Y * dfy) ** 2))
    )

    tmp = field_spec * phase
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)
    return out


# =========================
# Color + Drawing
# =========================
def color_to_bgr(color: str) -> Tuple[int, int, int]:
    """
    Accepts plotly-like colors: '#RRGGBB', 'rgb(r,g,b)', 'rgba(r,g,b,a)'.
    Returns OpenCV BGR tuple.
    """
    if isinstance(color, str) and color.startswith("#"):
        hx = color.lstrip("#")
        r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
        return (b, g, r)

    nums = list(map(int, re.findall(r"\d+", str(color))))
    if len(nums) < 3:
        raise ValueError(f"Unknown color format: {color}")
    r, g, b = nums[:3]
    return (b, g, r)


def draw_box_label(
    out_bgr: np.ndarray,
    cx: int,
    cy: int,
    half: int,
    text: str,
    color_bgr: Tuple[int, int, int],
    thickness: int = 2
) -> None:
    H, W = out_bgr.shape[:2]
    x0, y0 = int(cx - half), int(cy - half)
    x1, y1 = int(cx + half), int(cy + half)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W - 1, x1), min(H - 1, y1)

    cv2.rectangle(out_bgr, (x0c, y0c), (x1c, y1c), color_bgr, thickness)
    cv2.circle(out_bgr, (int(cx), int(cy)), 3, color_bgr, -1)
    cv2.putText(
        out_bgr,
        text,
        (x0c, max(15, y0c - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_bgr,
        1,
        cv2.LINE_AA,
    )


# =========================
# Inference + Overlay (single frame group from metadata)
# =========================
@dataclass
class OverlayResult:
    dfg: pd.DataFrame
    out_bgr: np.ndarray
    html_path: Optional[Path] = None


def predict_group_and_overlay(
    dfg: pd.DataFrame,
    model: nn.Module,
    idx2label: Dict[int, str],
    patches_dir: Path,
    pol_show: str = "M",
    conf_thr: float = 0.0,
    out_html: Optional[Path] = None,
    palette: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> OverlayResult:
    """
    For a metadata group (same frame/video), load the display frame from the original video,
    run classifier on the stored patches (npz), and draw predicted labels on the frame.

    Requires dfg columns:
      root_folder, root_name, file_id, frame_idx, center_x, center_y, half, patch_file, patch_index
    """
    if device is None:
        device = get_device()

    # Build class list in index order
    class_ids = sorted(idx2label.keys())
    classes = [idx2label[i] for i in class_ids]

    # Palette: if not given, use a simple deterministic list
    if palette is None:
        palette = [
            "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)", "rgb(214,39,40)",
            "rgb(148,103,189)", "rgb(140,86,75)", "rgb(227,119,194)", "rgb(127,127,127)",
            "rgb(188,189,34)", "rgb(23,190,207)"
        ]
    color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}

    r0 = dfg.iloc[0]
    root_folder = Path(r0["root_folder"])
    root_name = r0["root_name"]
    file_id = r0["file_id"]
    frame_idx = int(r0["frame_idx"])

    # Load display frame
    vid = root_folder / f"rec_{root_name}_{file_id}_{pol_show}.avi"
    img = read_frame(vid, frame_idx=frame_idx, grayscale=True)
    img_u8 = img if img.dtype == np.uint8 else cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    cache = NPZPatchCache(patches_dir)

    pred_labels: List[str] = []
    pred_confs: List[float] = []

    for _, row in dfg.iterrows():
        patch = cache.get_patch(row["patch_file"], int(row["patch_index"])).astype(np.float32)  # (C,H,W)
        if patch.max() > 1.5:
            patch /= 255.0

        xb = torch.from_numpy(patch[None]).to(device)

        with torch.no_grad():
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        lab = idx2label[pred_idx]
        conf = float(probs[pred_idx])

        pred_labels.append(lab)
        pred_confs.append(conf)

        if conf < conf_thr:
            continue

        cx = int(row["center_x"])
        cy = int(row["center_y"])
        half = int(row["half"])

        bgr = color_to_bgr(color_map[lab])
        draw_box_label(out, cx, cy, half, f"{lab} {conf:.2f}", bgr)

    dfg = dfg.copy()
    dfg["pred_label"] = pred_labels
    dfg["pred_conf"] = pred_confs

    html_path = None
    if out_html is not None:
        try:
            import plotly.express as px
            fig = px.imshow(out[..., ::-1], title=f"{pol_show} | {root_name} {file_id} frame {frame_idx}")
            fig.write_html(str(out_html), include_plotlyjs="cdn")
            html_path = Path(out_html)
        except Exception as e:
            # Plotly not available or failing—just skip HTML.
            print(f"[helpers.py] Plotly export failed: {e}")

    return OverlayResult(dfg=dfg, out_bgr=out, html_path=html_path)


# =========================
# Timing helper
# =========================
class Timer:
    def __init__(self):
        self.t0 = None
        self.dt = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
        return False

def annotate_raw_video(
    root_folder: Path,
    root_name: str,
    file_id: str,
    model: nn.Module,
    idx2label: Dict[int, str],
    kernel_path: Path,
    out_path: Path,
    pol_states=("H","M","V","P"),
    pol_show="M",
    half=100,
    thr=0.25,
    nms_k=15,
    conf_thr=0.5,
    device=None,
):
    """
    Annotate raw polarization videos frame-by-frame.
    No metadata. No dataset folder.
    """

    if device is None:
        device = get_device()

    kernel = read_kernel(kernel_path)

    # Open video streams
    caps = open_pol_caps(root_folder, root_name, file_id, pol_states)
    W, H, N, fps = get_video_info(caps[pol_show])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    classes = [idx2label[i] for i in sorted(idx2label.keys())]
    palette = [
        "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)",
        "rgb(214,39,40)", "rgb(148,103,189)", "rgb(140,86,75)"
    ]
    color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}

    print("Processing frames:", N)

    t0 = time.perf_counter()
    frame_count = 0

    while True:
        frames_pol = {}
        ok_all = True

        for pol, cap in caps.items():
            ok, fr = cap.read()
            if not ok:
                ok_all = False
                break
            if fr.ndim == 3:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            frames_pol[pol] = fr

        if not ok_all:
            break

        stack = np.stack([frames_pol[p] for p in pol_states], axis=0)
        display = frames_pol[pol_show]
        display_u8 = display if display.dtype == np.uint8 else cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out = cv2.cvtColor(display_u8, cv2.COLOR_GRAY2BGR)

        # --- Detection on raw channel ---
        centers = detect_centers_from_img(display_u8.astype(np.float32), kernel, thr=thr, nms_k=nms_k)

        # --- Crop + batch predict ---
        X = []
        coords = []

        for (cx, cy, score) in centers:
            chans = [crop_patch(stack[c], cx, cy, half) for c in range(stack.shape[0])]
            patch = np.stack(chans, axis=0)
            X.append(patch)
            coords.append((cx, cy))

        if len(X) > 0:
            X = np.stack(X).astype(np.float32)
            if X.max() > 1.5:
                X /= 255.0

            xb = torch.from_numpy(X).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(xb), dim=1).cpu().numpy()

            pred_idx = probs.argmax(axis=1)
            pred_conf = probs.max(axis=1)

            for (cx, cy), pi, conf in zip(coords, pred_idx, pred_conf):
                if conf < conf_thr:
                    continue

                lab = idx2label[int(pi)]
                bgr = color_to_bgr(color_map[lab])
                draw_box_label(out, cx, cy, half, f"{lab} {conf:.2f}", bgr)

        writer.write(out)
        frame_count += 1

        if frame_count % 25 == 0:
            print("Processed", frame_count)

    close_caps(caps)
    writer.release()

    total_time = time.perf_counter() - t0
    print(f"\nDone. {frame_count} frames.")
    print(f"Total time: {total_time:.2f}s")
    print(f"FPS effective: {frame_count/total_time:.2f}")
