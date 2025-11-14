"""Traces content tab with XY and Z traces (no video - video is shared)."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel
)
from PyQt5.QtCore import Qt
from src.ui.tabs.xy_traces_tab import XYTracesTab


class ZTracesPlaceholder(QWidget):
    """Placeholder for Z traces."""
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        label = QLabel("Z Traces\n\n(To be implemented)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(label)


class TracesContentTab(QWidget):
    """Traces content tab with XY traces (upper) and Z traces (lower)."""
    
    def __init__(self):
        super().__init__()
        self.video_widget = None
        self.xy_traces_tab = None
        self.z_traces_placeholder = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # Upper row: XY traces (from existing XY Traces tab)
        self.xy_traces_tab = XYTracesTab()
        layout.addWidget(self.xy_traces_tab, 1)
        
        # Lower row: Z traces placeholder
        self.z_traces_placeholder = ZTracesPlaceholder()
        layout.addWidget(self.z_traces_placeholder, 1)
    
    def set_video_widget(self, video_widget):
        """Set reference to shared video widget."""
        self.video_widget = video_widget
        if self.xy_traces_tab:
            self.xy_traces_tab.set_video_widget(video_widget)
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        if self.xy_traces_tab:
            self.xy_traces_tab.on_video_loaded()
