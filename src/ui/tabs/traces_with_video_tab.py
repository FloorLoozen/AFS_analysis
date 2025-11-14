"""Traces tab with video on left and XY/Z traces on right (1:2 ratio)."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QApplication, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from src.ui.video_widget import VideoWidget
from src.ui.tabs.xy_traces_tab import XYTracesTab


class ZTracesPlaceholder(QWidget):
    """Placeholder for Z traces."""
    
    def __init__(self):
        super().__init__()
        # Match application background
        app = QApplication.instance()
        if isinstance(app, QApplication):
            pal = self.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            self.setPalette(pal)
            self.setAutoFillBackground(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        label = QLabel("Z Traces\n\n(To be implemented)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(label)


class TracesWithVideoTab(QWidget):
    """Traces tab with video (left 1/3) and XY/Z traces (right 2/3)."""
    
    def __init__(self):
        super().__init__()
        self.video_widget = None
        self.video_container_layout = None
        self.xy_traces_tab = None
        self.z_traces_placeholder = None
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
        # Uniform outer margin for consistent spacing around group boxes
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
        
        # Create main video widget
        from src.ui.video_widget import VideoWidget
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_container_layout.addWidget(self.video_widget)
        
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
        
        # Right: XY and Z traces side-by-side in 2 columns (2/3 width)
        right_widget = QWidget()
        # Match background
        if isinstance(app, QApplication):
            pal = right_widget.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            right_widget.setPalette(pal)
            right_widget.setAutoFillBackground(True)
        right_layout = QHBoxLayout(right_widget)  # Changed to horizontal layout
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        # Left column: XY traces with group box
        xy_traces_group = QGroupBox("XY Traces")
        xy_traces_layout = QVBoxLayout(xy_traces_group)
        xy_traces_layout.setContentsMargins(4, 4, 4, 4)
        xy_traces_layout.setSpacing(0)
        
        self.xy_traces_tab = XYTracesTab()
        self.xy_traces_tab.set_video_widget(self.video_widget)
        xy_traces_layout.addWidget(self.xy_traces_tab)
        
        right_layout.addWidget(xy_traces_group, 1)
        
        # Right column: Z traces with group box
        z_traces_group = QGroupBox("Z Traces")
        z_traces_layout = QVBoxLayout(z_traces_group)
        z_traces_layout.setContentsMargins(4, 4, 4, 4)
        z_traces_layout.setSpacing(0)
        
        self.z_traces_placeholder = ZTracesPlaceholder()
        z_traces_layout.addWidget(self.z_traces_placeholder)
        
        right_layout.addWidget(z_traces_group, 1)
        
        layout.addWidget(right_widget, 2)  # 1:2 ratio - content gets 2 parts
    
    def attach_video_widget(self):
        """Attach video widget to this tab's layout."""
        if self.video_widget and self.video_widget.parent() != self:
            self.video_container_layout.addWidget(self.video_widget)
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        if self.xy_traces_tab:
            self.xy_traces_tab.on_video_loaded()
