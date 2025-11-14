"""Traces tab with video and XY/Z traces side by side."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QLabel, QSizePolicy
)
from PyQt5.QtCore import Qt
from src.ui.video_widget import VideoWidget
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


class TracesTab(QWidget):
    """Main Traces tab with video (left) and XY/Z traces (right) in 1:2 ratio."""
    
    def __init__(self):
        super().__init__()
        self.video_widget = None
        self.xy_traces_tab = None
        self.z_traces_placeholder = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # Create splitter for 1:2 ratio
        splitter = QSplitter(Qt.Horizontal)
        
        # Left column: Video widget (1/3 width)
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.video_widget)
        
        # Right column: XY and Z traces stacked vertically (2/3 width)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        # Upper row: XY traces (from existing XY Traces tab)
        self.xy_traces_tab = XYTracesTab()
        self.xy_traces_tab.set_video_widget(self.video_widget)
        right_layout.addWidget(self.xy_traces_tab, 1)
        
        # Lower row: Z traces placeholder
        self.z_traces_placeholder = ZTracesPlaceholder()
        right_layout.addWidget(self.z_traces_placeholder, 1)
        
        splitter.addWidget(right_widget)
        
        # Set 1:2 ratio (video gets 1 part, traces get 2 parts)
        splitter.setSizes([300, 600])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        if self.xy_traces_tab:
            self.xy_traces_tab.on_video_loaded()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget (not needed, we create our own)."""
        pass
