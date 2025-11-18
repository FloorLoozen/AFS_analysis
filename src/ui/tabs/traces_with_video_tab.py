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

        # We'll create three top-level columns so each gets equal width (1:1:1).
        # Left: Video container (first column) - divided into 2 rows
        video_container = QWidget()
        # Match background
        if isinstance(app, QApplication):
            pal = video_container.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            video_container.setPalette(pal)
            video_container.setAutoFillBackground(True)

        # Use vertical layout for video (only one video now)
        video_vertical_layout = QVBoxLayout(video_container)
        video_vertical_layout.setContentsMargins(0, 0, 0, 0)
        video_vertical_layout.setSpacing(6)

        # Video container (main video only, no LUT)
        video_group = QGroupBox("Acquisition Movie")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(4, 4, 4, 4)
        video_layout.setSpacing(0)

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

        video_layout.addWidget(video_widget_container)
        video_vertical_layout.addWidget(video_group)

        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Force minimum width to ensure equal sizing
        video_container.setMinimumWidth(100)

        # Middle column: XY traces group
        xy_traces_group = QGroupBox("XY Traces")
        xy_traces_layout = QVBoxLayout(xy_traces_group)
        xy_traces_layout.setContentsMargins(4, 4, 4, 4)
        xy_traces_layout.setSpacing(0)

        self.xy_traces_tab = XYTracesTab()
        self.xy_traces_tab.set_video_widget(self.video_widget)
        xy_traces_layout.addWidget(self.xy_traces_tab)
        
        # Force size policy to respect layout ratios
        xy_traces_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        xy_traces_group.setMinimumWidth(100)

        # Right column: Z traces group
        z_traces_group = QGroupBox("Z Traces")
        z_traces_layout = QVBoxLayout(z_traces_group)
        z_traces_layout.setContentsMargins(4, 4, 4, 4)
        z_traces_layout.setSpacing(0)

        self.z_traces_placeholder = ZTracesPlaceholder()
        z_traces_layout.addWidget(self.z_traces_placeholder)
        
        # Force size policy to respect layout ratios
        z_traces_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        z_traces_group.setMinimumWidth(100)

        # Add three columns: video=1/2, XY traces=1/4 (1/2 of 1/2), Z traces=1/4 (1/2 of 1/2)
        # Ratio 4:2:2 ensures video is exactly 1/2, then remaining 1/2 splits equally between XY and Z
        layout.addWidget(video_container, 4)  # 4/8 = 1/2
        layout.addWidget(xy_traces_group, 2)  # 2/8 = 1/4
        layout.addWidget(z_traces_group, 2)  # 2/8 = 1/4
    
    def attach_video_widget(self):
        """Attach video widget to this tab's layout."""
        if self.video_widget and self.video_widget.parent() != self:
            self.video_container_layout.addWidget(self.video_widget)
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        if self.xy_traces_tab:
            self.xy_traces_tab.on_video_loaded()
