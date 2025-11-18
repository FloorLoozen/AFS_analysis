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

        # We'll arrange three equal columns: left (video split in two rows), middle (preview plots),
        # right (tracked beads / controls placeholder). Each column receives equal stretch.
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

        video_layout.addWidget(video_widget_container)
        video_vertical_layout.addWidget(video_group)

        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Force minimum width to ensure proper sizing
        video_container.setMinimumWidth(100)

        # Middle column: Bead list panel (swapped position with plots)
        # We'll create the bead list here and pass it to PreviewTab
        beads_group = QGroupBox("Tracked Beads")
        beads_layout = QVBoxLayout(beads_group)
        beads_layout.setContentsMargins(4, 4, 4, 4)
        beads_layout.setSpacing(4)
        
        # Create bead list widget
        from PyQt5.QtWidgets import QListWidget, QCheckBox
        self.bead_list = QListWidget()
        self.bead_list.setStyleSheet("QListWidget { background-color: #f0f0f0; border: none; }")
        beads_layout.addWidget(self.bead_list)
        
        # Controls row at bottom
        from PyQt5.QtWidgets import QHBoxLayout as HBox
        controls = HBox()
        controls.setContentsMargins(0, 4, 0, 0)
        self.select_all = QCheckBox("Select All")
        controls.addWidget(self.select_all)
        controls.addStretch()
        
        self.count_label = QLabel("0 beads loaded")
        controls.addWidget(self.count_label)
        beads_layout.addLayout(controls)
        
        # Force size policy for bead list group
        beads_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        beads_group.setMinimumWidth(100)

        # Right column: Preview plots (swapped position with bead list)
        self.preview_tab = PreviewTab()
        self.preview_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Prevent preview tab from expanding too much
        self.preview_tab.setMinimumWidth(100)
        
        # Connect bead list to preview tab
        self.preview_tab.bead_list = self.bead_list
        self.preview_tab.select_all = self.select_all
        self.preview_tab.count_label = self.count_label
        # Connect signals
        self.bead_list.currentItemChanged.connect(self.preview_tab._on_bead_selected)
        self.select_all.stateChanged.connect(self.preview_tab._toggle_select_all)

        # Add three columns: video=1/2, bead list=1/10 (1/5 of 1/2), preview plots=2/5 (4/5 of 1/2)
        # Ratio 5:1:4 ensures video is exactly 1/2, then remaining 1/2 splits as 1:4 (bead list : plots)
        layout.addWidget(video_container, 5)  # 5/10 = 1/2
        layout.addWidget(beads_group, 1)  # 1/10
        layout.addWidget(self.preview_tab, 4)  # 4/10 = 2/5
    
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
        from utils.logger import Logger
        Logger.info(f"PreviewWithVideoTab received {len(tracking_data)} beads", "PREVIEW_WITH_VIDEO")
        if self.preview_tab:
            Logger.info(f"Forwarding to inner preview_tab", "PREVIEW_WITH_VIDEO")
            self.preview_tab.load_tracking_data(tracking_data)
        else:
            Logger.warning("preview_tab is None!", "PREVIEW_WITH_VIDEO")
