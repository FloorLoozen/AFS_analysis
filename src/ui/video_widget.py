"""Video player widget for AFS Analysis."""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor
import cv2
import numpy as np
import h5py
from pathlib import Path


class VideoWidget(QGroupBox):
    """Widget for video playback with play, pause, stop controls."""
    
    video_loaded = pyqtSignal()  # Signal emitted when video is loaded

    def __init__(self):
        """Initialize video widget."""
        super().__init__("Video Player")
        
        self.video_capture = None
        self.hdf5_file = None
        self.hdf5_video_data = None
        self.current_frame = None
        self.is_playing = False
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 30
        self.video_source_type = None  # 'hdf5' or 'video'
        self.video_metadata = {}
        
        # Timer for playback
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._play_next_frame)
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 15, 5, 5)  # Further reduced margins
        
        # Create video frame with consistent styling
        video_frame = self._create_video_frame()
        main_layout.addWidget(video_frame)

    def _create_video_frame(self):
        """Create the main video frame with consistent styling."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(3, 3, 3, 3)  # Minimal margins for maximum video size
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1024, 768)  # Much larger minimum size
        self.video_label.setStyleSheet("")  # No styling - transparent background
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, 1)
        
        # Frame info label
        self.frame_info_label = QLabel("No video loaded")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setStyleSheet("padding: 2px;")  # Minimal padding
        layout.addWidget(self.frame_info_label)
        
        # Timeline slider
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.timeline_slider)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(0, 2, 0, 0)  # Minimal top margin
        
        self.open_button = QPushButton("Open Video")
        self.open_button.setFixedWidth(100)
        self.open_button.clicked.connect(self.open_video_dialog)
        controls_layout.addWidget(self.open_button)
        
        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(60)
        self.play_button.clicked.connect(self._toggle_play_pause)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setFixedWidth(60)
        self.stop_button.clicked.connect(self._stop_video)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        layout.addStretch(1)
        
        return frame

    def open_video_dialog(self):
        """Open file dialog to select video or HDF5 file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video or HDF5 File",
            "",
            "All Supported (*.hdf5 *.h5 *.mp4 *.avi *.mov *.mkv);;HDF5 Files (*.hdf5 *.h5);;Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        """Load video file (regular video or HDF5)."""
        # Clean up previous video
        self.cleanup()
        
        file_path = Path(file_path)
        
        # Check if HDF5 file
        if file_path.suffix.lower() in ['.hdf5', '.h5']:
            success = self._load_hdf5_video(str(file_path))
            if not success:
                self.frame_info_label.setText("Error: Could not open HDF5 file")
                return
        else:
            success = self._load_regular_video(str(file_path))
            if not success:
                self.frame_info_label.setText("Error: Could not open video file")
                return
        
        # Update UI
        self.timeline_slider.setMaximum(self.total_frames - 1)
        self.timeline_slider.setValue(0)
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
        # Load first frame
        self._load_frame(0)
        self._update_frame_info()
        
        # Emit signal that video is loaded
        self.video_loaded.emit()
    
    def _load_hdf5_video(self, file_path):
        """Load video from HDF5 file."""
        try:
            self.hdf5_file = h5py.File(file_path, 'r')
            
            # Try to find video data
            if 'data/main_video' in self.hdf5_file:
                self.hdf5_video_data = self.hdf5_file['data/main_video']
            elif 'main_video' in self.hdf5_file:
                self.hdf5_video_data = self.hdf5_file['main_video']
            elif 'video' in self.hdf5_file:
                self.hdf5_video_data = self.hdf5_file['video']
            else:
                # Try to find any dataset with 4 dimensions (frames, height, width, channels)
                for key in self.hdf5_file.keys():
                    if isinstance(self.hdf5_file[key], h5py.Dataset):
                        if len(self.hdf5_file[key].shape) == 4:
                            self.hdf5_video_data = self.hdf5_file[key]
                            break
            
            if self.hdf5_video_data is None:
                self.hdf5_file.close()
                return False
            
            # Get video properties
            self.total_frames = self.hdf5_video_data.shape[0]
            
            # Try to get FPS from attributes - prefer actual_fps over target fps
            if 'actual_fps' in self.hdf5_video_data.attrs:
                self.fps = float(self.hdf5_video_data.attrs['actual_fps'])
            elif 'fps' in self.hdf5_video_data.attrs:
                self.fps = float(self.hdf5_video_data.attrs['fps'])
            else:
                self.fps = 30.0
            
            # Store metadata
            self.video_metadata = dict(self.hdf5_video_data.attrs)
            
            self.video_source_type = 'hdf5'
            self.current_frame_index = 0
            
            return True
            
        except Exception as e:
            print(f"Error loading HDF5: {e}")
            if self.hdf5_file:
                self.hdf5_file.close()
            return False
    
    def _load_regular_video(self, file_path):
        """Load regular video file (mp4, avi, etc)."""
        self.video_capture = cv2.VideoCapture(file_path)
        
        if not self.video_capture.isOpened():
            return False
        
        # Get video properties
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_index = 0
        self.video_source_type = 'video'
        
        return True

    def _load_frame(self, frame_index):
        """Load and display specific frame."""
        frame = None
        
        if self.video_source_type == 'hdf5':
            if self.hdf5_video_data is not None and 0 <= frame_index < self.total_frames:
                frame = self.hdf5_video_data[frame_index]
                # Convert BGR to RGB if needed (HDF5 might store as BGR)
                if 'color_format' in self.video_metadata:
                    if self.video_metadata['color_format'] == 'BGR':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        elif self.video_source_type == 'video':
            if self.video_capture:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.video_capture.read()
                if not ret:
                    return
        
        if frame is not None:
            self.current_frame = frame
            self.current_frame_index = frame_index
            self._display_frame(frame)

    def _display_frame(self, frame):
        """Display frame in video label."""
        # Convert BGR to RGB (only if needed - HDF5 might already be in correct format)
        if self.video_source_type == 'video':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Get label size
        label_size = self.video_label.size()
        
        # Get frame dimensions
        h, w = frame_rgb.shape[:2]
        label_w, label_h = label_size.width(), label_size.height()
        
        # Calculate scaling to fit label while maintaining aspect ratio
        # Scale to fit the available space as much as possible
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create QPixmap
        from PyQt5.QtGui import QImage
        height, width, channel = frame_resized.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.video_label.setPixmap(pixmap)

    def _toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.is_playing:
            self._pause_video()
        else:
            self._play_video()

    def _play_video(self):
        """Start video playback."""
        self.is_playing = True
        self.play_button.setText("Pause")
        
        # Calculate timer interval from actual FPS (use recorded FPS if available)
        interval = int(1000 / self.fps)
        self.playback_timer.start(interval)

    def _pause_video(self):
        """Pause video playback."""
        self.is_playing = False
        self.play_button.setText("Play")
        self.playback_timer.stop()

    def _stop_video(self):
        """Stop video playback and return to beginning."""
        self._pause_video()
        self._load_frame(0)
        self.timeline_slider.setValue(0)
        self._update_frame_info()

    def _play_next_frame(self):
        """Play next frame in sequence."""
        # Check if we have a video source
        if not self.video_source_type:
            return
        
        next_frame = self.current_frame_index + 1
        
        if next_frame >= self.total_frames:
            # Reached end of video
            self._pause_video()
            return
        
        self._load_frame(next_frame)
        self.timeline_slider.setValue(next_frame)
        self._update_frame_info()

    def _on_slider_moved(self, position):
        """Handle slider movement."""
        self._load_frame(position)
        self._update_frame_info()

    def _update_frame_info(self):
        """Update frame information label."""
        if self.total_frames > 0:
            time_seconds = self.current_frame_index / self.fps
            total_seconds = self.total_frames / self.fps
            self.frame_info_label.setText(
                f"Frame {self.current_frame_index + 1}/{self.total_frames} | "
                f"Time: {time_seconds:.2f}s / {total_seconds:.2f}s"
            )
        else:
            self.frame_info_label.setText("No video loaded")

    def cleanup(self):
        """Clean up resources."""
        self.playback_timer.stop()
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        if self.hdf5_file:
            self.hdf5_file.close()
            self.hdf5_file = None
        
        self.hdf5_video_data = None
        self.video_source_type = None
        self.video_metadata = {}
    
    def get_metadata(self):
        """Get video metadata."""
        return self.video_metadata.copy()
