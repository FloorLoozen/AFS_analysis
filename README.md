# AFS Analysis

Video analysis application for Atomic Force Spectroscopy HDF5 data.

## Architecture

**Modular Design**:
- `src/utils/`: Video loading, analysis algorithms, data export utilities
- `src/ui/`: User interface (PyQt5 widgets)

## Features

### Video Player
- **HDF5 Support**: Read video data from AFS_acquisition HDF5 files (`data/main_video` dataset)
core ayback Controls**: Play, Pause, Stop with accurate FPS playback
- **Timeline Navigation**: Slider for frame-by-frame navigation
- **Frame Information**: Display current frame, time, and FPS

### Analysis Tools
- **XY Tracking**: Track particle position over time
- **Z Tracking**: Measure vertical displacement using focus metrics
- **Data Export**: Export results to CSV, JSON, or NumPy formats
- **Metadata Display**: Display HDF5 metadata (frames, FPS, duration, sample info)
- **Modular Tab System**: Easy to extend with new analysis tools

### Interface
- **Two-Column Layout**: Video player (left) + Analysis tabs (right)
- **Maximized by Default**: Starts in maximized window mode
- **Keyboard Shortcuts**:
  - `Ctrl+O`: Open HDF5 file
  - `F11`: Toggle maximize window
  - `Ctrl+Q`: Exit application

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application
```bash
python src/main.py
```

### Using Utils in Scripts
```python
from src.utils.video_loader import VideoLoader

# Load HDF5 video
video = VideoLoader.load("data.hdf5")

# Get metadata
metadata = video.get_metadata()
print(f"Total frames: {video.total_frames}")
print(f"FPS: {video.fps}")

# Get frame
frame = video.get_frame(0)
print(f"Frame shape: {frame.shape}")
```

## Project Structure

```
AFS_analysis/
├── README.md
└── src/
    ├── main.py
    ├── utils/                         # Utilities
    │   ├── video_loader.py            # HDF5 video loading
    │   ├── video_controller.py        # Playback control
    │   └── frame_processor.py         # Frame processing
    │
    └── ui/                            # User Interface
        ├── main_window.py             # Main window
        ├── video_widget.py            # Video player
        └── measurement_info_widget.py # Info display
```

## Design

**Utils**: Video loading and playback (numpy, cv2, h5py)

**UI**: Simple PyQt5 video player interface

**Benefits**: Minimal, focused, easy to use

## HDF5 File Structure

Compatible with AFS_acquisition output files:
- `data/main_video`: Video dataset (frames, height, width, channels)
- Metadata attributes: `actual_fps`, `fps`, sample info, recording settings

## Dependencies

- PyQt5 >= 5.15.0
- NumPy >= 1.21.0
- OpenCV-Python >= 4.5.0
- h5py >= 3.0.0