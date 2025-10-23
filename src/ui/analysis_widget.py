"""Analysis controls widget for AFS Analysis."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget
)
from PyQt5.QtCore import Qt


class AnalysisWidget(QWidget):
    """Widget for analysis controls with tabbed interface."""

    def __init__(self):
        """Initialize analysis widget."""
        super().__init__()
        self.video_widget = None
        self._init_ui()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget
        
        # Pass video widget reference to all tabs
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, 'set_video_widget'):
                tab.set_video_widget(video_widget)
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        
        # Import and add tabs
        self._add_tabs()
        
        main_layout.addWidget(self.tab_widget)

    def _add_tabs(self):
        """Add all analysis tabs."""
        # Import tab modules
        from src.ui.tabs.xy_traces_tab import XYTracesTab
        from src.ui.tabs.z_traces_tab import ZTracesTab
        from src.ui.tabs.analysis_tab import AnalysisTab
        
        # Create and add tabs
        self.tab_widget.addTab(XYTracesTab(), "XY Traces")
        self.tab_widget.addTab(ZTracesTab(), "Z Traces")
        self.tab_widget.addTab(AnalysisTab(), "Analysis")
    
    def update_video_info(self):
        """Update video information display."""
        # Notify all tabs that video info has changed
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, 'on_video_loaded'):
                tab.on_video_loaded()
