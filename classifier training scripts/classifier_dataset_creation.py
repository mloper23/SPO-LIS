import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------

# names = ['COC', 'PMMA', 'PC', 'PS']
pol_states = ["H", "M", "V", "P"]

# Global dataset output (same for all plastic types)
dataset_root = Path("/Users/mjloperaa/Library/CloudStorage/OneDrive-SharedLibraries-VrijeUniversiteitBrussel/Maria Lopera - Documents/2026-I/Polarization_Off_Axis_Microplastics/POAM/dataset")  # CHANGE if you want
patches_dir = dataset_root / "patches_npz"
patches_dir.mkdir(parents=True, exist_ok=True)
metadata_csv = dataset_root / "metadata.csv"


half = 100
z_prop = 18.5e-2
wavelength = 532e-9
pixel_pitch = 3.45e-6 * 2

num_frames_to_get = 10  # how many frames per video to sample


# Template ROI for kernel on rec_0 (y0:y1, x0:x1)
# kernel_roi = (760, 850, 900, 980)  # (y0,y1,x0,x1)

kernel_path = "./kernel.png"  # or wherever you saved it
kernel = cv2.imread(str(kernel_path), cv2.IMREAD_GRAYSCALE)

if kernel is None:
    raise FileNotFoundError(f"Kernel not found at {kernel_path}")

kernel = kernel.astype(np.float32)


thr = 0.25
nms_k = 15  # dilation window for local maxima

# ---------------------------
# HELPERS
# ---------------------------

def crop_patches(img, cxs, cys, half, pad_value=0):
    """
    img: (H, W)
    cxs, cys: iterable of x (col), y (row) centers (ints)
    half: half patch size (patch size = 2*half)
    returns: patches (K, 2*half, 2*half)
    """
    H, W = img.shape
    K = len(cxs)
    patches = np.full((K, 2*half, 2*half), pad_value, dtype=img.dtype)

    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        cx = int(cx); cy = int(cy)

        # Image coordinates (clipped)
        x0 = max(cx - half, 0)
        x1 = min(cx + half, W)
        y0 = max(cy - half, 0)
        y1 = min(cy + half, H)

        # Patch coordinates (where to place the clipped region)
        x0p = x0 - (cx - half)   # shift right if left side was clipped
        y0p = y0 - (cy - half)   # shift down if top was clipped
        x1p = x0p + (x1 - x0)
        y1p = y0p + (y1 - y0)

        patches[i, y0p:y1p, x0p:x1p] = img[y0:y1, x0:x1]

    return patches

def angularSpectrum(z, field, wavelength, pixel_pitch_in):
    '''
    # Function from pyDHM (https://github.com/catrujilla/pyDHM)
    # Function to diffract a complex field using the angular spectrum approach
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''
    M, N = np.shape(field)
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')
    dfx = 1 / (pixel_pitch_in * M)
    dfy = 1 / (pixel_pitch_in * N)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    phase = np.exp2(1j * z * np.pi * np.sqrt(np.power(1/wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))

    tmp = field_spec*phase

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out

def read_frame(video_file, frame_idx=0, grayscale=True):
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

def parse_rec_filename(fname):
    parts = Path(fname).stem.split("_")
    if len(parts) < 4:
        return None
    return {"rootname": parts[1], "fileid": parts[2], "pol": parts[3]}


def get_frame_count(video_file):
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {video_file}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def detect_centers_from_rec0(rec_0, kernel, thr=0.25, nms_k=15):
    # y0, y1, x0, x1 = kernel_roi
    # kernel = rec_0[y0:y1, x0:x1]

    img = rec_0 - cv2.GaussianBlur(rec_0, (0, 0), sigmaX=10)
    tpl = kernel - cv2.GaussianBlur(kernel, (0, 0), sigmaX=10)

    corr = cv2.matchTemplate(img.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)

    dil = cv2.dilate(corr, np.ones((nms_k, nms_k), np.uint8))
    peaks = (corr == dil) & (corr >= thr)

    ys, xs = np.where(peaks)
    scores = corr[ys, xs].astype(np.float32)

    h, w = tpl.shape
    centers = [(int(x + w/2), int(y + h/2), float(s)) for x, y, s in zip(xs, ys, scores)]
    return centers

# ---------------------------
# DISCOVER FILE IDS
# ---------------------------
for name_ in names:
    # root = Path(f'/User/mjloperaa/Library/CloudStorage/OneDrive-SharedLibraries-VrijeUniversiteitBrussel/Maria Lopera - '
    #         f'Documents/2024-II/Polarization/data/POAM_data/210126/{name_}')
    root = Path(f'/Users/mjloperaa/Library/CloudStorage/OneDrive-SharedLibraries-VrijeUniversiteitBrussel/Maria Lopera - '
                f'Documents/2024-II/Polarization/data/POAM_data/220126/{name_}')
    label = root.name  # e.g., "COC" (folder name). You can override manually.
    avi_files = sorted([p.name for p in root.iterdir() if p.suffix.lower() == ".avi"])
    rec_files = [f for f in avi_files if f.lower().startswith("rec_")]
    parsed = [parse_rec_filename(f) for f in rec_files]
    parsed = [d for d in parsed if d is not None]

    if len(parsed) == 0:
        raise RuntimeError("No parsable rec_*.avi files found.")

    root_name = parsed[0]["rootname"]
    file_ids = sorted(set(d["fileid"] for d in parsed))

    print("root_name:", root_name)
    print("Total file_ids:", len(file_ids))

    # ---------------------------
    # LOOP ALL FILE IDS
    # ---------------------------
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []  # accumulate metadata then append once at end (faster/safer)

    for file_id in file_ids:
        base_stem = root / f"rec_{root_name}_{file_id}"

        # Pick one existing pol video to read N
        probe_video = None
        for pol in pol_states:
            fprobe = Path(str(base_stem) + f"_{pol}.avi")
            if fprobe.exists():
                probe_video = fprobe
                break
        if probe_video is None:
            print(f"[{file_id}] SKIP: no videos found.")
            continue

        N = get_frame_count(probe_video)
        if N <= 0:
            print(f"[{file_id}] SKIP: invalid frame count.")
            continue

        # Evenly spaced frame indices
        frame_idxs = np.linspace(0, N - 1, num_frames_to_get).astype(int)
        frame_idxs = np.unique(frame_idxs)  # avoid duplicates if N is small

        print(f"[{file_id}] frames in video: {N} | sampling: {frame_idxs.tolist()}")

        for frame_idx in frame_idxs:
            # Load that frame for all pol states
            frames_pol = {}
            for pol in pol_states:
                f = Path(str(base_stem) + f"_{pol}.avi")
                if not f.exists():
                    print(f"[{file_id}] WARNING missing {pol}: {f.name}")
                    continue
                frames_pol[pol] = read_frame(f, frame_idx=frame_idx, grayscale=True)

            pol_loaded = [p for p in pol_states if p in frames_pol]
            if len(pol_loaded) == 0:
                print(f"[{file_id}] frame {frame_idx}: SKIP (no pol frames).")
                continue

            stack = np.stack([frames_pol[p] for p in pol_loaded], axis=0)  # (C,H,W)

            # Reconstruct rec_0 from first loaded channel
            init = stack[0]
            rec_0 = cv2.blur(
                np.abs(angularSpectrum(z_prop, np.sqrt(init), wavelength, pixel_pitch))**2,
                (3, 3)
            )

            # Detect
            centers = detect_centers_from_rec0(rec_0, kernel, thr=thr, nms_k=nms_k)
            if len(centers) == 0:
                print(f"[{file_id}] frame {frame_idx}: No detections.")
                continue

            cxs = [c[0] for c in centers]
            cys = [c[1] for c in centers]
            det_scores = np.array([c[2] for c in centers], dtype=np.float32)
            K = len(centers)

            # Crop patches
            patches_pol = np.stack(
                [crop_patches(stack[c], cxs, cys, half) for c in range(stack.shape[0])],
                axis=1
            )  # (K, C, 2*half, 2*half)

            # Save one npz per file_id per frame (manageable file sizes)
            run_id = f"{label}_{root_name}_{file_id}_frame{frame_idx}_{run_stamp}"
            patch_file = patches_dir / f"{run_id}.npz"

            np.savez_compressed(
                patch_file,
                patches=patches_pol.astype(np.uint8),
                centers=np.stack([cxs, cys], axis=1).astype(np.int32),
                scores=det_scores,
                pol_order=np.array(pol_loaded),
                half=np.array(half, dtype=np.int32),
                root_name=np.array(root_name),
                file_id=np.array(file_id),
                frame_idx=np.array(frame_idx, dtype=np.int32),
                label=np.array(label),
            )

            # Metadata rows
            for i in range(K):
                all_rows.append({
                    "run_id": run_id,
                    "label": label,
                    "root_folder": str(root),
                    "root_name": root_name,
                    "file_id": file_id,
                    "frame_idx": int(frame_idx),
                    "patch_file": str(patch_file),
                    "patch_index": i,
                    "center_x": int(cxs[i]),
                    "center_y": int(cys[i]),
                    "score": float(det_scores[i]),
                    "half": half,
                    "patch_h": 2 * half,
                    "patch_w": 2 * half,
                    "pol_order": ",".join(pol_loaded),
                    "thr": thr,
                    "nms_k": nms_k,
                    "N_frames": int(N),
                    "sampled_frames": num_frames_to_get,
                })

            print(f"[{file_id}] frame {frame_idx}: saved {K} patches -> {patch_file.name}")


    # ---------------------------
    # APPEND METADATA ONCE
    # ---------------------------
    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows)
        if metadata_csv.exists():
            df.to_csv(metadata_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(metadata_csv, index=False)

        print(f"\nAppended {len(df)} rows to: {metadata_csv}")
    else:
        print("\nNo rows produced (no detections / missing videos).")