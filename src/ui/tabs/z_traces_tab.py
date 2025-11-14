"""Z Traces tab for tracking vertical displacement."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class ZTracesTab(QWidget):
    """Tab for Z-axis tracking and visualization."""

    def __init__(self):
        """Initialize Z traces tab."""
        super().__init__()
        self.video_widget = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 8)
        
        # Placeholder content
        label = QLabel("Z Traces\n\n(To be implemented)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(label)
        
        layout.addStretch()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        # Add your code here to process video data
        pass
