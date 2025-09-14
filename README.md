# Stereo Depth Inference System

## Overview
This repository provides a complete stereo vision system for depth estimation using two USB webcams. The system includes camera streaming, calibration image capture, stereo calibration, and real-time depth inference.

## Features
- **Dual Camera Streaming**: Test and preview both cameras simultaneously
- **Calibration Image Capture**: Collect synchronized chessboard pairs for stereo calibration
- **Stereo Calibration**: Automatic camera intrinsic and extrinsic parameter estimation
- **Real-time Depth Estimation**: Live disparity mapping with WLS filtering

## Requirements
- Python 3.9+ recommended
- Two USB cameras connected and accessible as indices `0` and `1`
- Permissions to access cameras (on macOS, grant Terminal/IDE camera access in System Settings)

### Python Dependencies
Install in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
- `opencv-contrib-python` - Required for `cv2.ximgproc` (WLS disparity filter)
- `numpy>=1.19.0` - Numerical computations
- `open3d>=0.13.0` - 3D point cloud processing
- `imutils>=0.5.0` - Video stream utilities
- `openpyxl>=3.0.0` - Excel data export
- `scikit-learn>=0.24.0` - Data preprocessing

## Repository Structure
```
├── video_stream.py              # Basic dual camera streaming test
├── calibration_image_capture.py # Capture chessboard calibration images
├── stereo_vision.py            # Main stereo depth inference system
├── calibration_images/         # Directory for saved chessboard images
├── image_pairs/               # Directory for general stereo image pairs
├── 9x6chessboard.png          # Printable 9×6 inner-corner chessboard
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Camera Configuration
All scripts assume two cameras at indices:
- `src=0` (Right camera)
- `src=1` (Left camera)

If your cameras appear swapped or not found:
- Swap USB ports, or
- Edit the `VideoStream(src=...)` indices in the scripts

---

## Usage Guide

### 1. Camera Streaming Test
**Script:** `video_stream.py`

Test basic camera functionality and verify both cameras are working.

```bash
python video_stream.py
```

**Features:**
- Displays live feed from both cameras in separate windows
- Press `q` to quit

### 2. Calibration Image Capture
**Script:** `calibration_image_capture.py`

Capture synchronized chessboard image pairs for stereo calibration.

```bash
python calibration_image_capture.py
```

**Controls:**
- `s`: Save current stereo pair when chessboard is detected
- `q`: Quit the program
- Any other key: Skip current frame and continue

**Before Starting:**
- Print or display the provided `9x6chessboard.png` with flat mounting
- Ensure good, uniform lighting and minimal glare
- Move the chessboard around the scene and vary orientation/depth

**Tips:**
- Aim for 20-60 good pairs covering diverse poses
- Keep the chessboard fully visible in both cameras and in focus
- The inner-corner grid is 9×6 (not square count)

**Saved Files:**
- `calibration_images/chessboard-R0.png`, `calibration_images/chessboard-L0.png`
- `calibration_images/chessboard-R1.png`, `calibration_images/chessboard-L1.png`
- etc.

### 3. Stereo Depth Inference
**Script:** `stereo_vision.py`

Main stereo vision system that performs calibration and real-time depth estimation.

```bash
python stereo_vision.py
```

**What it does:**
1. **Calibration Phase:**
   - Loads saved chessboard image pairs from `calibration_images/`
   - Performs individual camera calibration (intrinsic parameters)
   - Performs stereo calibration (extrinsic parameters)
   - **Prints camera matrices:**
     - Intrinsic Matrix
     - Extrinsic Matrix - Rotation Matrix (R)
     - Extrinsic Matrix - Translation Vector (T)

2. **Real-time Depth Estimation:**
   - Streams from both cameras
   - Rectifies frames using calibration parameters
   - Computes disparity using StereoSGBM algorithm
   - Applies WLS filtering for improved quality
   - Displays colorized disparity map

**Configuration:**
- **Image count range:** Edit the number to match your saved image pairs:
  ```python
  for i in range(0, 50):  # Change 50 to your actual number of pairs
  ```
- **Chessboard size:** Fixed at 9×6 inner corners
- **Camera exposure:** Manual exposure set to -6 (adjust if needed)

## Camera Parameters

The system automatically calculates and displays:

**Intrinsic Matrix:**
```
[[1.42945992e+03 0.00000000e+00 6.83046151e+02]
 [0.00000000e+00 1.42434397e+03 4.81538797e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
```

**Extrinsic Matrix - Rotation Matrix (R):**
```
[[ 0.99795576 -0.0120479  -0.06276256]
 [ 0.01112565  0.99982525 -0.01502309]
 [ 0.06293259  0.01429411  0.99791541]]
```

**Extrinsic Matrix - Translation Vector (T):**
```
[[-3.2806174 ]
 [ 0.03283476]
 [-0.15686016]]
```
