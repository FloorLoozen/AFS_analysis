"""Measurement information widget for AFS Analysis."""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt


class MeasurementInfoWidget(QGroupBox):
    """Widget for displaying measurement information and metadata."""

    def __init__(self):
        """Initialize measurement info widget."""
        super().__init__("Measurement Information")
        self.video_widget = None  # Will be set by main window
        self._init_ui()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create info frame with consistent styling
        frame = self._create_info_frame()
        main_layout.addWidget(frame)

    def _create_info_frame(self):
        """Create the main info frame with consistent styling."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Info display area
        self.info_label = QLabel("No video loaded")
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 5px;")
        layout.addWidget(self.info_label, 1)
        
        return frame
    
    def update_info(self):
        """Update measurement information display."""
        if not self.video_widget:
            self.info_label.setText("No video loaded")
            return
        
        metadata = self.video_widget.get_metadata()
        
        if not metadata:
            self.info_label.setText("No video loaded")
            return
        
        # Format metadata in a clean, minimal way
        info_text = ""
        
        # Sample information first
        if 'user_sample_name' in metadata:
            info_text += f"<b>Sample:</b> {metadata['user_sample_name']}<br>"
        if 'user_operator' in metadata:
            info_text += f"<b>Operator:</b> {metadata['user_operator']}<br>"
        
        if info_text:
            info_text += "<br>"
        
        # Recording stats
        if 'total_frames' in metadata:
            info_text += f"<b>Frames:</b> {metadata['total_frames']}<br>"
        if 'actual_fps' in metadata:
            info_text += f"<b>FPS:</b> {metadata['actual_fps']:.2f}<br>"
        if 'recording_duration_s' in metadata:
            duration = metadata['recording_duration_s']
            info_text += f"<b>Duration:</b> {duration:.1f}s<br>"
        
        if info_text and ('frame_shape' in metadata or 'total_data_mb' in metadata):
            info_text += "<br>"
        
        # Technical details
        if 'frame_shape' in metadata:
            shape = metadata['frame_shape']
            info_text += f"<b>Resolution:</b> {shape}<br>"
        if 'downscale_factor' in metadata:
            info_text += f"<b>Downscale:</b> {metadata['downscale_factor']}x<br>"
        if 'total_data_mb' in metadata:
            size_mb = metadata['total_data_mb']
            info_text += f"<b>Size:</b> {size_mb:.1f} MB<br>"
        
        self.info_label.setText(info_text)
