"""Preview main tab with video and preview graphs side by side."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt
from src.ui.video_widget import VideoWidget
from src.ui.tabs.preview_tab import PreviewTab


class PreviewMainTab(QWidget):
    """Main Preview tab with video (left) and graphs (right) in 1:2 ratio."""
    
    def __init__(self):
        super().__init__()
        self.video_widget = None
        self.preview_tab = None
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
        
        # Right column: Preview graphs (2/3 width)
        self.preview_tab = PreviewTab()
        self.preview_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.preview_tab)
        
        # Set 1:2 ratio (video gets 1 part, preview gets 2 parts)
        splitter.setSizes([300, 600])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget from traces tab."""
        # We need to use the video widget from Traces tab for synchronization
        # For now, keep our own video widget - synchronization can be added later
        if self.preview_tab:
            self.preview_tab.set_video_widget(self.video_widget)
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        if self.preview_tab:
            self.preview_tab.on_video_loaded()
