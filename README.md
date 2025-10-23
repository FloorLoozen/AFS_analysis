# AFS Analysis

Video analysis application for Atomic Force Spectroscopy data with PyQt5 interface.

## Features

### Video Player
- **HDF5 Support**: Read video data directly from AFS_acquisition HDF5 files (`data/main_video` dataset)
- **Video Formats**: Support for MP4, AVI, MOV, MKV files
- **Playback Controls**: Play, Pause, Stop
- **Timeline Navigation**: Slider for frame-by-frame navigation
- **Frame Information**: Display current frame, time, and FPS

### Analysis Tools
- **Metadata Display**: Automatic display of HDF5 metadata including:
  - Recording information (frames, FPS, duration)
  - Sample details (name, operator, system)
  - Resolution and compression settings
  - Timestamps and file size
- **Modular Design**: Easy to extend with new analysis tools

### Interface
- **Two-Column Layout**: Video player (left) + Analysis controls (right)
- **Maximized by Default**: Starts in maximized window mode
- **Keyboard Shortcuts**:
  - `Ctrl+O`: Open video/HDF5 file
  - `F11`: Toggle maximize window
  - `Ctrl+Q`: Exit application

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## HDF5 File Structure

Compatible with AFS_acquisition output files containing:
- `data/main_video`: Main video dataset (frames, height, width, channels)
- Metadata attributes: fps, sample info, recording settings, etc.

## Project Structure

```
src/
├── main.py                 # Application entry point
└── ui/
    ├── main_window.py      # Main window with 2-column layout
    ├── video_widget.py     # Video player (HDF5 + regular video)
    └── analysis_widget.py  # Analysis controls and metadata display
```

## Dependencies

- PyQt5 >= 5.15.0
- NumPy >= 1.21.0
- OpenCV-Python >= 4.5.0
- h5py >= 3.0.0