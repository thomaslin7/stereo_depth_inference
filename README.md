# Stereo Vision — Streaming, Calibration Image Capture, and Depth Inference

## Overview
This repository provides three workflows for working with a stereo camera pair (two USB webcams):
- Camera streaming test: preview both cameras and optionally save quick snapshots
- Capture calibration images: collect synchronized chessboard pairs for calibration
- Stereo depth inference: run calibration from images and visualize filtered disparity maps

Tested on macOS. Should also work on Linux/Windows with minor adjustments to camera indices.

## Requirements
- Python 3.9+ recommended
- Two USB cameras connected and accessible as indices `0` and `1`
- Permissions to access cameras (on macOS, grant Terminal/IDE camera access in System Settings)

### Python dependencies
Install in a virtual environment if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python opencv-contrib-python imutils openpyxl scikit-learn
```

Notes:
- `opencv-contrib-python` is required for `cv2.ximgproc` (WLS disparity filter) used in stereo depth.
- If you already have `opencv-python` installed, ensure versions are compatible; prefer only `opencv-contrib-python`.

## Repository structure
- `logitech_camera.py`: lightweight two-camera streaming and snapshot tool (save with `c`)
- `Take_images_for_calibration.py`: acquire synchronized chessboard pairs and save to `calibration_images/`
- `stereo_vision.py`: performs stereo calibration from saved images and visualizes a filtered disparity map
- `calibration_images/`: default folder for saved chessboard images (already included)
- `9x6chessboard.png`: printable 9×6 inner-corner chessboard for calibration

## Camera index assumptions
All scripts assume two cameras at indices `src=0` (Right) and `src=1` (Left). If your cameras appear swapped or not found:
- Swap USB ports, or
- Edit the `VideoStream(src=...)` indices inside the scripts accordingly.

---

## 1) Camera streaming test
Script: `logitech_camera.py`

What it does:
- Opens both cameras and displays two windows: `frame1` (src=0) and `frame2` (src=1)
- Keyboard controls:
  - `q`: quit
  - `c`: save a snapshot from each camera to the project root as `calibrate01_<timestamp>.png` and `calibrate02_<timestamp>.png`

Run:
```bash
python logitech_camera.py
```

Troubleshooting:
- If a window is black, verify camera indices and that no other app is using the camera.
- On macOS, ensure Terminal/IDE has Camera permission.

---

## 2) Capture calibration images
Script: `Take_images_for_calibration.py`

What it does:
- Continuously reads both cameras and attempts to detect a 9×6 inner-corner chessboard in each stream
- Shows raw frames in `imgR`/`imgL` and grayscale detections in `VideoR`/`VideoL` when corners are found
- Keyboard controls when corners are detected:
  - `s`: save the current stereo pair to `calibration_images/` as `chessboard-R<ID>.png` and `chessboard-L<ID>.png`
  - Any other key: skip and continue
- Global quit:
  - `q`: quit the program

Before you start:
- Print or display the provided `9x6chessboard.png` with flat mounting
- Ensure good, uniform lighting and minimal glare
- Move the chessboard around the scene and vary orientation/depth to capture robust calibration sets

Run:
```bash
python Take_images_for_calibration.py
```

Tips:
- Aim for 20–60 good pairs covering diverse poses.
- Keep the chessboard fully visible in both cameras and in focus.
- The inner-corner grid is 9×6; do not confuse with square count.

Saved files example:
- `calibration_images/chessboard-R0.png`
- `calibration_images/chessboard-L0.png`

---

## 3) Stereo depth inference (calibrate + disparity)
Script: `stereo_vision.py`

What it does:
- Loads saved chessboard image pairs from `calibration_images/`
- Detects corners and runs mono calibration for each camera
- Runs stereo calibration and rectification
- Streams from both cameras, rectifies frames, computes disparity using StereoSGBM, filters with WLS, and displays a colorized filtered disparity map window: `Filtered Color Depth`
- Double-clicking the disparity window prints an approximate distance estimate at the clicked pixel (empirical model)

Important configuration inside `stereo_vision.py`:
- Image count range for calibration:
  - Edit this line to match the number of image pairs you actually saved:
    ```python
    for i in range(0, 52):
    ```
    If you saved N pairs, change to `range(0, N)`.
- Chessboard size is fixed at 9×6 inner corners.
- Camera indices: `VideoStream(src=0)` and `VideoStream(src=1)`; adjust if needed.
- Exposure/white balance: the script disables auto exposure/WB and sets manual exposure to `-6`. Tweak values or comment lines if your cameras behave differently.

Run:
```bash
python stereo_vision.py
```

Controls:
- Disparity window `Filtered Color Depth`:
  - Double-click: print estimated distance at clicked pixel
  - `q`: quit

Performance tips:
- Reduce window size or camera resolution for higher FPS (requires editing capture settings or resizing frames).
- Ensure you installed `opencv-contrib-python` for `cv2.ximgproc` support.
