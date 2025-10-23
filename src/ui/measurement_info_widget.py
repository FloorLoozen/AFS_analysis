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
        
        # Info display area - compact single line with smaller text
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(False)
        self.info_label.setStyleSheet("padding: 3px; color: #666;")
        layout.addWidget(self.info_label)
        
        return frame
    
    def update_info(self):
        """Update measurement information display."""
        if not self.video_widget:
            self.info_label.setText("")
            return
        
        metadata = self.video_widget.get_metadata()
        
        if not metadata:
            self.info_label.setText("")
            return
        
        # Format metadata - simple and essential only
        info_parts = []
        
        # Only show key recording information
        if 'total_frames' in metadata:
            info_parts.append(f"{metadata['total_frames']} frames")
        
        if 'actual_fps' in metadata:
            fps = metadata['actual_fps']
            info_parts.append(f"{fps:.1f} fps")
            
            # Calculate and show duration
            if 'total_frames' in metadata:
                duration = metadata['total_frames'] / fps
                info_parts.append(f"{duration:.1f}s")
        
        if 'frame_shape' in metadata:
            shape = metadata['frame_shape']
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                info_parts.append(f"{shape[1]}×{shape[0]}")
        
        # Join with separators
        info_text = "  |  ".join(info_parts) if info_parts else ""
        self.info_label.setText(info_text)
