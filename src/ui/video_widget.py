"""Video player widget for AFS Analysis."""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

from src.utils.video_controller import VideoController
from src.utils.video_loader import VideoLoader
from src.utils.frame_processor import FrameProcessor


class VideoWidget(QGroupBox):
    """Widget for video playback with play, pause, stop controls.
    
    This is a thin UI layer that uses core.video_controller for all business logic.
    """
    
    video_loaded = pyqtSignal()  # Signal emitted when video is loaded

    def __init__(self):
        """Initialize video widget."""
        super().__init__("Video Player")
        
        # Controller handles all business logic
        self.controller = VideoController()
        
        # Connect controller signals to UI update methods
        self.controller.frame_changed.connect(self._on_frame_changed)
        self.controller.playback_state_changed.connect(self._on_playback_state_changed)
        self.controller.video_loaded.connect(self._on_video_loaded)
        self.controller.playback_finished.connect(self._on_playback_finished)
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 15, 5, 5)
        
        # Create video frame with consistent styling
        video_frame = self._create_video_frame()
        main_layout.addWidget(video_frame)

    def _create_video_frame(self):
        """Create the main video frame with consistent styling."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(3, 3, 3, 3)
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1024, 768)
        self.video_label.setStyleSheet("")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, 1)
        
        # Frame info label
        self.frame_info_label = QLabel("No video loaded")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setStyleSheet("padding: 2px;")
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
        controls_layout.setContentsMargins(0, 2, 0, 0)
        
        self.open_button = QPushButton("Open Video")
        self.open_button.setFixedWidth(100)
        self.open_button.clicked.connect(self.open_video_dialog)
        controls_layout.addWidget(self.open_button)
        
        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(60)
        self.play_button.clicked.connect(self._on_play_button_clicked)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setFixedWidth(60)
        self.stop_button.clicked.connect(self._on_stop_button_clicked)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        layout.addStretch(1)
        
        return frame

    def open_video_dialog(self):
        """Open file dialog to select HDF5 file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 File",
            "",
            "HDF5 Files (*.hdf5 *.h5);;All Files (*.*)"
        )
        
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        """Load HDF5 video file."""
        try:
            # Use core module to load video
            video_source = VideoLoader.load(file_path)
            
            # Pass to controller
            self.controller.load_video(video_source)
            
            # Update UI controls
            self.timeline_slider.setMaximum(self.controller.get_total_frames() - 1)
            self.timeline_slider.setValue(0)
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            
        except Exception as e:
            self.frame_info_label.setText(f"Error: {str(e)}")
            print(f"Error loading video: {e}")

    def _on_video_loaded(self):
        """Handle video loaded signal from controller."""
        # Emit our own signal for other widgets
        self.video_loaded.emit()
    
    def _on_frame_changed(self, frame_index: int, frame_data: np.ndarray):
        """Handle frame changed signal from controller."""
        # Display the frame
        self._display_frame(frame_data)
        
        # Update slider
        self.timeline_slider.setValue(frame_index)
        
        # Update info
        self._update_frame_info(frame_index)
    
    def _on_playback_state_changed(self, is_playing: bool):
        """Handle playback state changed signal from controller."""
        self.play_button.setText("Pause" if is_playing else "Play")
    
    def _on_playback_finished(self):
        """Handle playback finished signal from controller."""
        # Playback automatically paused, just update UI if needed
        pass

    def _display_frame(self, frame: np.ndarray):
        """Display frame in video label."""
        # Get label size
        label_size = self.video_label.size()
        label_w, label_h = label_size.width(), label_size.height()
        
        # Use FrameProcessor to resize frame
        frame_resized = FrameProcessor.resize_to_fit(frame, label_w, label_h)
        
        # Convert to QPixmap
        height, width, channel = frame_resized.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.video_label.setPixmap(pixmap)

    def _on_play_button_clicked(self):
        """Handle play button click."""
        self.controller.toggle_play_pause()

    def _on_stop_button_clicked(self):
        """Handle stop button click."""
        self.controller.stop()

    def _on_slider_moved(self, position):
        """Handle slider movement."""
        self.controller.seek_to_frame(position)

    def _update_frame_info(self, frame_index: int):
        """Update frame information label."""
        total_frames = self.controller.get_total_frames()
        fps = self.controller.get_fps()
        
        if total_frames > 0:
            time_seconds = frame_index / fps
            total_seconds = total_frames / fps
            self.frame_info_label.setText(
                f"Frame {frame_index + 1}/{total_frames} | "
                f"Time: {time_seconds:.2f}s / {total_seconds:.2f}s"
            )
        else:
            self.frame_info_label.setText("No video loaded")

    def cleanup(self):
        """Clean up resources."""
        self.controller.cleanup()
    
    def get_metadata(self):
        """Get video metadata."""
        return self.controller.get_metadata()
    
    def get_current_frame(self) -> np.ndarray:
        """Get current frame data."""
        return self.controller.get_current_frame()
