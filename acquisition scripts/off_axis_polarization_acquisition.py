import cv2
import os
import datetime
import tisgrabber as IC
import numpy as np


def Y16toPol(image):
    # Get image dimensions
    h, w, _ = image.shape

    # Create coordinates for each sub-section
    xp = np.linspace(0, w - 1, int(w / 2)).astype(int)
    yp = np.linspace(0, h - 1, int(h / 2)).astype(int)
    Xp, Yp = np.meshgrid(xp, yp)

    xip = np.linspace(0, w - 2, int(w / 2)).astype(int) + 1
    yip = np.linspace(0, h - 2, int(h / 2)).astype(int) + 1
    Xip, Yip = np.meshgrid(xip, yip)

    # Initialize a new image array for the four quadrants
    image_ = np.zeros([int(h / 2), int(w / 2), 4], dtype=np.uint8)

    # Fill the four quadrants with the corresponding pixel values
    image_[:, :, 0] = image[Yp,  Xp,  0]  # Quadrant 1
    image_[:, :, 1] = image[Yip, Xp,  0]  # Quadrant 2
    image_[:, :, 2] = image[Yp,  Xip, 0]  # Quadrant 3
    image_[:, :, 3] = image[Yip, Xip, 0]  # Quadrant 4
    return image_


def to_uint8(img):
    """
    Make sure frames are uint8 for VideoWriter.
    If already uint8 -> ok.
    Else scale per-frame to 0..255 for visualization.
    """
    if img.dtype == np.uint8:
        return img
    maxv = float(np.max(img))
    if maxv <= 0:
        return np.zeros_like(img, dtype=np.uint8)
    return cv2.convertScaleAbs(img, alpha=(255.0 / maxv))


# path = './260325/P2_v2/'
# path = r'C:/Users/mlopera/Vrije Universiteit Brussel/Maria Lopera - Documents/2025-I/HPS2/data/160625/Polarization_OffAxisDLHM/Polarization_camera'
path = r'./data/200226/Resolution/'
formats = ['Y16', 'RGB32']
format = formats[0]

# Make sure output directory exists (FIXED)
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
    print(f'Directory "{path}" created.')
else:
    print(f'Directory "{path}" already exists.')


# Initialize camera
Camera = IC.TIS_CAM()
Camera.open('DZK 33UX250 25220110')

if format == 'Y16':
    Camera.SetVideoFormat("Y16 (2448x2048)")
if format == 'RGB32':
    Camera.SetFormat(IC.SinkFormats.RGB32)

Camera.StartLive(0)


# ===================== VIDEO RECORDING STATE =====================
recording = False
writers = None
record_fps = 30  # set to your acquisition fps (approx)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # AVI MJPG is usually robust on Windows
record_base = None


def start_recording(frame4, out_dir):
    """
    frame4: [H, W, 4] (V,P,M,H channels)
    """
    h, w = frame4[:, :, 0].shape[:2]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"rec_{ts}")

    wV = cv2.VideoWriter(f"{base}_V.avi", fourcc, record_fps, (w, h), isColor=False)
    wP = cv2.VideoWriter(f"{base}_P.avi", fourcc, record_fps, (w, h), isColor=False)
    wM = cv2.VideoWriter(f"{base}_M.avi", fourcc, record_fps, (w, h), isColor=False)
    wH = cv2.VideoWriter(f"{base}_H.avi", fourcc, record_fps, (w, h), isColor=False)

    writers_dict = {"V": wV, "P": wP, "M": wM, "H": wH}

    if not all(wr.isOpened() for wr in writers_dict.values()):
        for wr in writers_dict.values():
            wr.release()
        raise RuntimeError("Could not open one or more VideoWriter outputs (codec/path issue).")

    return writers_dict, base


def stop_recording(writers_dict):
    if writers_dict is None:
        return
    for wr in writers_dict.values():
        wr.release()


print("\nControls:")
print("  r : start recording (4 videos: V/P/M/H)")
print("  t : stop recording")
print("  s : save snapshot (4 PNGs)")
print("  q : quit\n")


try:
    while True:
        # Acquire frame
        Camera.SnapImage()
        image = Camera.GetImage()

        if format == 'Y16':
            image = Y16toPol(image)  # -> [H/2, W/2, 4]
        else:
            # If RGB32, convert to 4 grayscale channels for compatibility
            if image.ndim == 3 and image.shape[2] >= 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image[:, :, 0] if image.ndim == 3 else image
            image = np.stack([gray, gray, gray, gray], axis=-1).astype(np.uint8)

        # ---- If recording, write ORIGINAL SIZE channels (not resized) ----
        if recording and writers is not None:
            writers["V"].write(to_uint8(image[:, :, 0]))
            writers["P"].write(to_uint8(image[:, :, 1]))
            writers["M"].write(to_uint8(image[:, :, 2]))
            writers["H"].write(to_uint8(image[:, :, 3]))

        # ---- Build display (resized preview) ----
        V = image[:, :, 0]

        # FFT preview: log-magnitude for display (FIXED: avoid complex)
        ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(V)))
        ft_mag = np.log1p(np.abs(ft))  # log(1 + |FT|)

        ft_mag = cv2.resize(ft_mag, (306, 256))
        img_small = cv2.resize(image, (306, 256))

        # Normalize FT to uint8 for display
        ft_disp = cv2.normalize(ft_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        final_image = np.concatenate((img_small[:, :, 0], ft_disp), axis=1)

        cv2.imshow('frame', final_image)
        cv2.resizeWindow('frame', 800, 600)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Snapshot
        elif key == ord('s'):
            Camera.SnapImage()
            snap = Camera.GetImage()
            if format == 'Y16':
                snap = Y16toPol(snap)
            else:
                if snap.ndim == 3 and snap.shape[2] >= 3:
                    gray = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
                else:
                    gray = snap[:, :, 0] if snap.ndim == 3 else snap
                snap = np.stack([gray, gray, gray, gray], axis=-1).astype(np.uint8)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename0 = f"{path}/V_{timestamp}.png"
            filename1 = f"{path}/P_{timestamp}.png"
            filename2 = f"{path}/M_{timestamp}.png"
            filename3 = f"{path}/H_{timestamp}.png"

            cv2.imwrite(filename0, snap[:, :, 0])
            cv2.imwrite(filename1, snap[:, :, 1])
            cv2.imwrite(filename2, snap[:, :, 2])
            cv2.imwrite(filename3, snap[:, :, 3])
            print("Frame saved!")

        # Start recording
        elif key == ord('r') and not recording:
            try:
                writers, record_base = start_recording(image, path)
                recording = True
                print(f"Recording started: {record_base}_*.avi")
            except Exception as e:
                writers = None
                recording = False
                print(f"ERROR starting recording: {e}")

        # Stop recording
        elif key == ord('t') and recording:
            stop_recording(writers)
            writers = None
            recording = False
            print("Recording stopped")

finally:
    # Cleanup
    if writers is not None:
        stop_recording(writers)
    cv2.destroyAllWindows()
    try:
        Camera.StopLive()
    except Exception:
        pass
