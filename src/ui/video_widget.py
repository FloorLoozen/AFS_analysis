"""Video player widget for AFS Analysis."""

from typing import Optional
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent
import numpy as np
import cv2

from src.utils.video_controller import VideoController
from src.utils.video_loader import VideoLoader
from src.utils.frame_processor import FrameProcessor


class ClickableLabel(QLabel):
    """Label that emits click signals with coordinates."""
    
    clicked = pyqtSignal(int, int)  # x, y coordinates (left click)
    right_clicked = pyqtSignal(int, int)  # x, y coordinates (right click)
    
    def mousePressEvent(self, ev: QMouseEvent):
        """Handle mouse press events."""
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(ev.x(), ev.y())
        elif ev.button() == Qt.RightButton:
            self.right_clicked.emit(ev.x(), ev.y())
        super().mousePressEvent(ev)


class VideoWidget(QGroupBox):
    """Widget for video playback with play, pause, stop controls.
    
    This is a thin UI layer that uses core.video_controller for all business logic.
    """
    
    video_loaded = pyqtSignal()  # Signal emitted when video is loaded
    bead_clicked = pyqtSignal(int, int)  # Signal when bead is clicked (x, y in original frame coords)
    bead_right_clicked = pyqtSignal(int, int)  # Signal when bead is right-clicked

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
        
        # Tracking state
        self.tracking_enabled = False
        self.bead_positions = {}  # {bead_id: (x, y)} for current frame
        self.bead_traces = {}  # {bead_id: [(x, y), ...]} for trace history
        self.show_traces = True  # Flag to show/hide traces
        self.click_to_select_mode = False
        self.last_displayed_frame = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        
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
        layout.setSpacing(3)
        
        # Video display area (takes most space)
        self.video_label = ClickableLabel()
        self.video_label.clicked.connect(self._on_video_clicked)
        self.video_label.right_clicked.connect(self._on_video_right_clicked)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(False)  # Don't auto-scale, we handle it manually
        layout.addWidget(self.video_label, 1)
        
        # Timeline slider below video
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.timeline_slider)
        
        # Frame info label (compact, below slider)
        self.frame_info_label = QLabel("")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setStyleSheet("padding: 2px; color: #666;")
        layout.addWidget(self.frame_info_label)
        
        # Control buttons at the bottom
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(0, 2, 0, 0)
        
        self.open_button = QPushButton("Open Video")
        self.open_button.setFixedWidth(100)
        self.open_button.clicked.connect(self.open_video_dialog)
        controls_layout.addWidget(self.open_button)
        
        self.play_button = QPushButton("▶️ Play")
        self.play_button.setFixedWidth(80)
        self.play_button.clicked.connect(self._on_play_button_clicked)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)
        
        self.pause_button = QPushButton("⏸️ Pause")
        self.pause_button.setFixedWidth(80)
        self.pause_button.clicked.connect(self._on_pause_button_clicked)
        self.pause_button.setEnabled(False)
        controls_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.setFixedWidth(80)
        self.stop_button.clicked.connect(self._on_stop_button_clicked)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
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
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            
        except Exception as e:
            self.frame_info_label.setText(f"Error: {str(e)}")
            print(f"Error loading video: {e}")

    def _on_video_loaded(self):
        """Handle video loaded signal from controller."""
        # Emit our own signal for other widgets
        self.video_loaded.emit()
    
    def _on_frame_changed(self, frame_index: int, frame_data: np.ndarray):
        """Handle frame changed signal from controller."""
        # Store original frame
        self.last_displayed_frame = frame_data.copy()
        
        # Draw tracking overlays if enabled using FrameProcessor
        if self.tracking_enabled and len(self.bead_positions) > 0:
            # Pass traces only if show_traces is True
            traces_to_show = self.bead_traces if self.show_traces else None
            frame_data = FrameProcessor.draw_bead_overlays(
                frame_data, self.bead_positions, traces_to_show, box_size=36, box_thickness=2
            )
        
        # Display the frame
        self._display_frame(frame_data)
        
        # Update slider
        self.timeline_slider.setValue(frame_index)
        
        # Update info
        self._update_frame_info(frame_index)
    
    def _on_video_clicked(self, display_x: int, display_y: int):
        """Handle clicks on the video display."""
        if self.click_to_select_mode and self.last_displayed_frame is not None:
            # Convert display coordinates to original frame coordinates
            original_x, original_y = self._display_to_frame_coords(display_x, display_y)
            
            if original_x is not None and original_y is not None:
                # Emit signal with original frame coordinates
                self.bead_clicked.emit(original_x, original_y)
    
    def _on_video_right_clicked(self, display_x: int, display_y: int):
        """Handle right-clicks on the video display."""
        if self.click_to_select_mode and self.last_displayed_frame is not None:
            # Convert display coordinates to original frame coordinates
            original_x, original_y = self._display_to_frame_coords(display_x, display_y)
            
            if original_x is not None and original_y is not None:
                # Emit signal with original frame coordinates
                self.bead_right_clicked.emit(original_x, original_y)
    
    def _display_to_frame_coords(self, display_x: int, display_y: int):
        """Convert display coordinates to original frame coordinates."""
        if self.last_displayed_frame is None:
            return None, None
        
        # Get original frame dimensions
        frame_h, frame_w = self.last_displayed_frame.shape[:2]
        
        # Get display label size
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        
        # Calculate scale that was used (same as in _display_frame)
        scale = min(label_w / frame_w, label_h / frame_h)
        
        # Calculate displayed dimensions
        display_w = int(frame_w * scale)
        display_h = int(frame_h * scale)
        
        # Calculate offset (centering)
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2
        
        # Remove offset
        relative_x = display_x - offset_x
        relative_y = display_y - offset_y
        
        # Check if click is within the actual image
        if relative_x < 0 or relative_y < 0 or relative_x >= display_w or relative_y >= display_h:
            return None, None
        
        # Scale back to original coordinates
        original_x = int(relative_x / scale)
        original_y = int(relative_y / scale)
        
        return original_x, original_y
    
    def set_tracking_enabled(self, enabled: bool):
        """Enable or disable tracking visualization."""
        self.tracking_enabled = enabled
        # Redraw current frame
        if self.last_displayed_frame is not None:
            frame_index = self.controller.current_frame_index
            self._on_frame_changed(frame_index, self.last_displayed_frame.copy())
    
    def set_click_to_select_mode(self, enabled: bool):
        """Enable or disable click-to-select mode."""
        self.click_to_select_mode = enabled
        if enabled:
            self.video_label.setCursor(Qt.CrossCursor)
        else:
            self.video_label.setCursor(Qt.ArrowCursor)
    
    def update_bead_positions(self, bead_positions: dict):
        """Update bead positions for current frame."""
        self.bead_positions = bead_positions
        
        # Update trace history
        for bead_id, (x, y) in bead_positions.items():
            if bead_id not in self.bead_traces:
                self.bead_traces[bead_id] = []
            self.bead_traces[bead_id].append((x, y))
            
            # Keep last 100 points for performance
            if len(self.bead_traces[bead_id]) > 100:
                self.bead_traces[bead_id] = self.bead_traces[bead_id][-100:]
        
        # Trigger redraw if tracking is enabled
        if self.tracking_enabled and self.last_displayed_frame is not None:
            frame_index = self.controller.current_frame_index
            self._on_frame_changed(frame_index, self.last_displayed_frame.copy())
    
    def _on_playback_state_changed(self, is_playing: bool):
        """Handle playback state changed signal from controller."""
        if is_playing:
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        else:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            # Stop button enabled if not at beginning
            if self.controller.current_frame_index > 0:
                self.stop_button.setEnabled(True)
            else:
                self.stop_button.setEnabled(False)
    
    def _on_playback_finished(self):
        """Handle playback finished signal from controller."""
        # Playback automatically paused, just update UI if needed
        pass

    def _display_frame(self, frame: np.ndarray):
        """Display frame in video label."""
        # Get label size
        label_size = self.video_label.size()
        label_w, label_h = label_size.width(), label_size.height()
        
        if label_w <= 0 or label_h <= 0:
            return
        
        # Use FrameProcessor to resize frame (returns tuple now)
        frame_resized, _ = FrameProcessor.resize_to_fit(frame, label_w, label_h)
        
        # Convert to QPixmap
        height, width = frame_resized.shape[:2]
        if len(frame_resized.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = width
            q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def _on_play_button_clicked(self):
        """Handle play button click."""
        self.controller.play()

    def _on_pause_button_clicked(self):
        """Handle pause button click."""
        self.controller.pause()

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
                f"Frame {frame_index + 1}/{total_frames}  |  "
                f"{time_seconds:.2f}s / {total_seconds:.2f}s"
            )
        else:
            self.frame_info_label.setText("")

    def cleanup(self):
        """Clean up resources."""
        self.controller.cleanup()
    
    def get_metadata(self):
        """Get video metadata."""
        return self.controller.get_metadata()
    
    def get_hdf5_file(self):
        """Get the HDF5 file handle from the video source."""
        if self.controller and self.controller.video_source:
            return self.controller.video_source.hdf5_file
        return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame data."""
        return self.controller.get_current_frame()
