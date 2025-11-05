# AFS Analysis

Video analysis application for Atomic Force Spectroscopy HDF5 data with XY bead tracking.

**🚀 GPU-Accelerated**: Supports OpenCL for AMD/NVIDIA/Intel GPUs (3-5x faster processing). See [GPU_ACCELERATION.md](GPU_ACCELERATION.md) for details.

## Features

### Video Player
- **HDF5 Support**: Read video from AFS_acquisition files (`/raw_data/main_video` or `/data/main_video` for backward compatibility)
- **Playback Controls**: Play, pause, stop with frame-by-frame navigation
- **Metadata Display**: Show recording info (frames, FPS, duration)
- **GPU Acceleration**: Hardware-accelerated frame processing and display

### XY Bead Tracking
- **Auto-Detection**: Automatically find beads using adjustable parameters (GPU-accelerated)
- **Manual Validation**: Add/remove beads with left/right clicks
- **Template Matching**: Track bead positions across all frames
- **Auto-Save**: Saves every 100 frames during tracking
- **Resume Capability**: Continue from last tracked frame
- **HDF5 Storage**: Data saved to `/analysed_data/xy_tracking` in source file
- **CSV Export**: Export tracking data for external analysis

## Performance

- **GPU Acceleration**: 3-5x faster frame processing with OpenCL
- **Supported GPUs**: AMD (Radeon Pro), NVIDIA, Intel integrated
- **Auto Fallback**: Seamlessly uses CPU if GPU unavailable
- **Check Status**: View GPU info in the Info tab

## Installation

```bash
pip install PyQt5 opencv-python h5py numpy
```

## Usage

```bash
python src/main.py
```

### Tracking Workflow
1. Open HDF5 file (Ctrl+O)
2. Click **Load** if previous tracking exists, or
3. Click **Auto** to detect beads (adjust threshold/size if needed)
4. Remove unwanted beads (right-click) or add missed ones (left-click)
5. Click **Track** to process all frames
6. Click **Save** to save to HDF5 or **CSV** to export

## HDF5 Structure

**Input** (from AFS_acquisition):
- `/raw_data/main_video` - Video dataset (frames, height, width, channels)
  - Also supports `/data/main_video` for backward compatibility

**Output** (created by AFS_analysis):
- `/analysed_data/xy_tracking` - Tracking positions (frames, beads, xy)
- Templates and metadata stored as attributes/datasets

## Architecture

```
src/
├── main.py                 # Entry point
├── ui/
│   ├── main_window.py      # Main app window
│   ├── video_widget.py     # Video player with overlay
│   ├── analysis_widget.py  # Tab container
│   └── tabs/
│       ├── xy_traces_tab.py  # XY tracking UI
│       └── [z_traces_tab.py]  # Placeholder for Z tracking
└── utils/
    ├── video_loader.py     # HDF5 video source
    ├── video_controller.py # Playback control
    ├── analysis.py         # BeadTracker, auto-detection
    └── tracking_io.py      # HDF5 save/load
```

## Dependencies

- PyQt5 >= 5.15.0
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- h5py >= 3.0.0