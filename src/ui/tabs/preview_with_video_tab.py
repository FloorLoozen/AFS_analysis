"""Preview tab with video on left and graphs on right (1:2 ratio)."""
# type: ignore

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QApplication, QGroupBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from src.ui.tabs.preview_tab import PreviewTab


class PreviewWithVideoTab(QWidget):
    """Preview tab with video (left 1/3) and graphs (right 2/3)."""
    
    def __init__(self):
        super().__init__()
        self.video_widget = None
        self.video_container_layout = None
        self.preview_tab = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Match application background
        app = QApplication.instance()
        if isinstance(app, QApplication):
            pal = self.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            self.setPalette(pal)
            self.setAutoFillBackground(True)
        
        layout = QHBoxLayout(self)
        # Uniform outer margin so group boxes have equal space from the widget edges
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Left: Video container (1/3 width) - divided into 2 rows
        video_container = QWidget()
        # Match background
        if isinstance(app, QApplication):
            pal = video_container.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            video_container.setPalette(pal)
            video_container.setAutoFillBackground(True)
        
        # Use vertical layout to stack two videos
        video_vertical_layout = QVBoxLayout(video_container)
        video_vertical_layout.setContentsMargins(0, 0, 0, 0)
        video_vertical_layout.setSpacing(6)
        
        # Top video container (main video)
        top_video_group = QGroupBox("Acquisition Movie")
        top_video_layout = QVBoxLayout(top_video_group)
        top_video_layout.setContentsMargins(4, 4, 4, 4)
        top_video_layout.setSpacing(0)
        
        # Video widget container
        video_widget_container = QWidget()
        self.video_container_layout = QHBoxLayout(video_widget_container)
        self.video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_widget_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        top_video_layout.addWidget(video_widget_container)
        
        video_vertical_layout.addWidget(top_video_group, 1)
        
        # Bottom video placeholder
        bottom_video_group = QGroupBox("LUT Movie")
        bottom_video_layout = QVBoxLayout(bottom_video_group)
        bottom_video_layout.setContentsMargins(4, 4, 4, 4)
        bottom_video_layout.setSpacing(0)
        
        # Placeholder content
        bottom_video_placeholder = QWidget()
        if isinstance(app, QApplication):
            pal = bottom_video_placeholder.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            bottom_video_placeholder.setPalette(pal)
            bottom_video_placeholder.setAutoFillBackground(True)
        placeholder_layout = QVBoxLayout(bottom_video_placeholder)
        placeholder_layout.setContentsMargins(8, 8, 8, 8)
        
        placeholder_label = QLabel("(To be implemented)")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("color: #888; font-style: italic;")
        placeholder_layout.addWidget(placeholder_label)
        
        bottom_video_layout.addWidget(bottom_video_placeholder)
        
        video_vertical_layout.addWidget(bottom_video_group, 1)
        
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(video_container, 1)  # 1:2 ratio - video gets 1 part
        
        # Right: Preview graphs (2/3 width)
        self.preview_tab = PreviewTab()
        self.preview_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.preview_tab, 2)  # 1:2 ratio - content gets 2 parts
    
    def set_video_widget(self, video_widget):
        """Set the shared video widget from Traces tab."""
        self.video_widget = video_widget
        
        # Set video widget for preview tab
        if self.preview_tab:
            self.preview_tab.set_video_widget(self.video_widget)
    
    def attach_video_widget(self):
        """Attach video widget to this tab's layout."""
        if self.video_widget and self.video_widget.parent() != self:
            self.video_container_layout.addWidget(self.video_widget)
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        if self.preview_tab:
            self.preview_tab.on_video_loaded()
    
    def load_tracking_data(self, tracking_data):
        """Forward tracking data to the inner preview tab."""
        if self.preview_tab:
            self.preview_tab.load_tracking_data(tracking_data)
