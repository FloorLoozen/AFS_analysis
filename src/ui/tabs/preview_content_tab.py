"""Preview content tab with graphs (no video - video is shared)."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from src.ui.tabs.preview_tab import PreviewTab


class PreviewContentTab(QWidget):
    """Preview content tab with graphs and bead list."""
    
    def __init__(self):
        super().__init__()
        self.video_widget = None
        self.preview_tab = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Preview graphs
        self.preview_tab = PreviewTab()
        self.preview_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.preview_tab)
    
    def set_video_widget(self, video_widget):
        """Set reference to shared video widget."""
        self.video_widget = video_widget
        if self.preview_tab:
            self.preview_tab.set_video_widget(video_widget)
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        if self.preview_tab:
            self.preview_tab.on_video_loaded()
