"""XY Traces tab - empty placeholder."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class XYTracesTab(QWidget):
    """Empty placeholder tab for XY tracking."""

    def __init__(self):
        """Initialize XY traces tab."""
        super().__init__()
        self.video_widget = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Placeholder content
        label = QLabel("XY Traces\n\n(To be implemented)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-style: italic; font-size: 14px;")
        layout.addWidget(label)
        
        layout.addStretch()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        pass
