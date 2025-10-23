"""Analysis controls widget for AFS Analysis."""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QFrame
)
from PyQt5.QtCore import Qt


class AnalysisWidget(QGroupBox):
    """Widget for analysis controls and settings."""

    def __init__(self):
        """Initialize analysis widget."""
        super().__init__("Analysis Tools")
        self.video_widget = None  # Will be set by main window
        self._init_ui()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create empty frame with consistent styling
        frame = self._create_frame()
        main_layout.addWidget(frame)

    def _create_frame(self):
        """Create the main frame with consistent styling."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Empty placeholder for future analysis tools
        placeholder = QLabel("Analysis tools will be added here")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(placeholder)
        
        layout.addStretch(1)
        
        return frame
    
    def update_video_info(self):
        """Update video information display."""
        # Placeholder for future implementation
        pass
