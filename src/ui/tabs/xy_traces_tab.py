"""XY Traces tab for bead tracking with HDF5 integration."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLabel, QMessageBox, QSpinBox, QGroupBox, QFormLayout, QCheckBox, QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette
import numpy as np
from pathlib import Path

from src.analysis import BeadTracker, detect_beads_auto
from src.utils.tracking_io import TrackingDataIO
from src.utils.logger import Logger

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple
if TYPE_CHECKING:
    from src.ui.video_widget import VideoWidget


class XYTracesTab(QWidget):
    """Tab for XY bead tracking with auto-detection and HDF5 storage."""

    def __init__(self):
        """Initialize XY traces tab."""
        super().__init__()
        # Ensure this tab uses the application's Window palette color so it matches other tabs
        app = QApplication.instance()
        if app is not None:
            pal = self.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))  # type: ignore
            self.setPalette(pal)
            self.setAutoFillBackground(True)
        self.video_widget: Optional['VideoWidget'] = None
        self.tracker = BeadTracker(window_size=40)
        self.is_tracking = False
        self.is_paused = False
        self.is_selecting = False
        self.is_validating = False
        # template feature removed
        self.next_bead_id = 0
        self.detected_positions = []
        self.current_hdf5_path = None
        
        # Tracking state
        self.current_tracking_frame = 0
        self.total_tracking_frames = 0
        self.tracking_timer = QTimer()
        self.tracking_timer.timeout.connect(self._process_next_frame)
        
        # Performance optimization: Process multiple frames per timer tick
        self.frames_per_batch = 5  # Process 5 frames at a time for better performance
        
        self._init_ui()
        # Track maximum label width for status rows so we can align values consistently
        self._status_label_width = 0
    
    def _init_ui(self):
        """Initialize the user interface matching AFS_acquisition style."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 24, 8, 8)
        layout.setSpacing(10)
        
        # Settings group matching AFS style
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QFormLayout()
        settings_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)  # type: ignore
        settings_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # type: ignore
        # Left-align labels and fields so text and inputs sit flush to the left
        settings_layout.setLabelAlignment(Qt.AlignLeft)  # type: ignore
        settings_layout.setFormAlignment(Qt.AlignLeft)  # type: ignore
        
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(50, 255)
        self.threshold_spinbox.setValue(150)
        self.threshold_spinbox.setFixedWidth(80)
        settings_layout.addRow("Brightness:", self.threshold_spinbox)
        
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(100, 2000)
        self.min_size_spinbox.setValue(500)
        self.min_size_spinbox.setFixedWidth(80)
        settings_layout.addRow("Min Size:", self.min_size_spinbox)
        
        self.max_size_spinbox = QSpinBox()
        self.max_size_spinbox.setRange(100, 10000)
        self.max_size_spinbox.setValue(5000)
        self.max_size_spinbox.setFixedWidth(80)
        settings_layout.addRow("Max Size:", self.max_size_spinbox)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Control buttons group
        controls_group = QGroupBox("Tracking Controls")
        # Use modest internal margins so widgets sit clearly inside the group box
        controls_group.setContentsMargins(8, 20, 8, 8)
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(8)
        # Align controls to the left and use modest inner margins so buttons sit inside the frame
        controls_layout.setContentsMargins(6, 8, 6, 6)
        controls_layout.setAlignment(Qt.AlignLeft)  # type: ignore

        # Use a grid layout so columns line up vertically across the two rows
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setAlignment(Qt.AlignLeft)  # type: ignore

        # Row 1 order: Auto Detect, Add (Manual), Load Saved
        self.auto_detect_button = QPushButton("Auto Detect")
        self.auto_detect_button.setEnabled(False)
        self.auto_detect_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.auto_detect_button.setFixedHeight(30)
        self.auto_detect_button.clicked.connect(self._on_auto_detect_clicked)

        self.select_beads_button = QPushButton("Add")
        self.select_beads_button.setEnabled(False)
        self.select_beads_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.select_beads_button.setFixedHeight(30)
        self.select_beads_button.clicked.connect(self._on_select_beads_clicked)

        self.load_tracking_button = QPushButton("Load Saved")
        self.load_tracking_button.setEnabled(False)
        self.load_tracking_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.load_tracking_button.setFixedHeight(30)
        self.load_tracking_button.clicked.connect(self._on_load_tracking_clicked)

        # Row 2: Start Tracking, Pause, Clear
        self.start_tracking_button = QPushButton("Start Tracking")
        self.start_tracking_button.setEnabled(False)
        self.start_tracking_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.start_tracking_button.setFixedHeight(30)
        self.start_tracking_button.clicked.connect(self._on_start_tracking_clicked)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.pause_button.setFixedHeight(30)
        self.pause_button.clicked.connect(self._on_pause_tracking_clicked)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.clear_button.setFixedHeight(30)
        self.clear_button.clicked.connect(self._on_clear_clicked)

        # Keep checkbox compact so it doesn't stretch with the buttons
        self.show_traces_checkbox = QCheckBox("Show Traces")
        self.show_traces_checkbox.setEnabled(False)
        self.show_traces_checkbox.setChecked(True)
        self.show_traces_checkbox.stateChanged.connect(self._on_toggle_traces_clicked)
        self.show_traces_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.show_traces_checkbox.setMinimumHeight(30)
        
        # Stuck bead checkbox (placeholder for drift correction)
        self.stuck_bead_checkbox = QCheckBox("Stuck Bead")
        self.stuck_bead_checkbox.setEnabled(False)  # Disabled for now
        self.stuck_bead_checkbox.setToolTip("Mark as stuck bead for drift correction (coming soon)")
        self.stuck_bead_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stuck_bead_checkbox.setMinimumHeight(30)

        # Place widgets in grid so columns align
        grid.addWidget(self.auto_detect_button, 0, 0)
        grid.addWidget(self.select_beads_button, 0, 1)
        grid.addWidget(self.load_tracking_button, 0, 2)

        grid.addWidget(self.start_tracking_button, 1, 0)
        grid.addWidget(self.pause_button, 1, 1)
        grid.addWidget(self.clear_button, 1, 2)
        grid.addWidget(self.show_traces_checkbox, 1, 3)
        grid.addWidget(self.stuck_bead_checkbox, 1, 4)

        # Make columns aligned by using the widest button in each column
        col0_w = max(self.auto_detect_button.sizeHint().width(), self.start_tracking_button.sizeHint().width()) + 20
        col1_w = max(self.select_beads_button.sizeHint().width(), self.pause_button.sizeHint().width()) + 20
        col2_w = max(self.load_tracking_button.sizeHint().width(), self.clear_button.sizeHint().width()) + 20

        for btn in (self.auto_detect_button, self.start_tracking_button):
            btn.setFixedWidth(col0_w)
        for btn in (self.select_beads_button, self.pause_button):
            btn.setFixedWidth(col1_w)
        for btn in (self.load_tracking_button, self.clear_button):
            btn.setFixedWidth(col2_w)

        controls_layout.addLayout(grid)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Status display - matching Info tab style exactly
        status_group = QGroupBox("Status")
        self.status_layout = QFormLayout(status_group)
        self.status_layout.setContentsMargins(8, 8, 8, 8)
        self.status_layout.setSpacing(5)
        # Align labels to the left and we'll set a fixed label column width based on the longest label
        self.status_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(status_group)
        
        layout.addStretch()
    
    def _add_status_row(self, label_text, value_text=""):
        """Add a label-value row to the status layout (matching info tab style)."""
        # Label
        label = QLabel(label_text)
        label.setStyleSheet("color: #666;")
        # Ensure labels have a consistent width based on the longest label seen
        fm = label.fontMetrics()
        label_w = fm.boundingRect(label_text).width() + 8
        if label_w > self._status_label_width:
            self._status_label_width = label_w
        label.setFixedWidth(self._status_label_width)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # type: ignore
        
        # Value
        value = QLabel(value_text)
        value.setWordWrap(True)
        value.setStyleSheet("color: #222;")
        value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # type: ignore
        
        # Add to form layout
        self.status_layout.addRow(label, value)
        
        return value
    
    def _clear_status(self):
        """Clear all rows from status layout."""
        while self.status_layout.count():
            item = self.status_layout.takeAt(0)
            if item and item.widget():  # type: ignore
                item.widget().deleteLater()  # type: ignore
    
    def _update_status(self, info_text="", status_text=""):
        """Update status display with consistent formatting."""
        self._clear_status()
        
        if info_text:
            self._add_status_row("Info:", info_text)
        
        if status_text:
            self._add_status_row("Status:", status_text)
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        if self.video_widget:
            try:
                self.video_widget.controller.frame_changed.disconnect(self._on_video_frame_changed)  # type: ignore
            except TypeError:
                pass

        self.video_widget = video_widget
        if video_widget:
            video_widget.bead_clicked.connect(self._on_bead_clicked)
            video_widget.bead_right_clicked.connect(self._on_bead_right_clicked)
            video_widget.controller.frame_changed.connect(self._on_video_frame_changed)  # type: ignore
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        self.auto_detect_button.setEnabled(True)
        self.select_beads_button.setEnabled(True)
        
        # Get video file path
        if self.video_widget:
            metadata = self.video_widget.get_metadata()  # type: ignore
            self.current_hdf5_path = metadata.get('file_path', None)
            
            # Check if tracking data exists
            if self.current_hdf5_path and TrackingDataIO.has_tracking_data(self.current_hdf5_path):
                self.load_tracking_button.setEnabled(True)
                self._update_status("Previous tracking found - click Load", "Data in /analysed_data/xy_tracking")
            else:
                self.load_tracking_button.setEnabled(False)
                self._update_status("Click Auto or Manual to begin", "")
        
        self._reset_tracking()
    
    def _on_load_tracking_clicked(self):
        """Load saved tracking data from HDF5."""
        if not self.current_hdf5_path:
            return
        
        try:
            beads_data, metadata = TrackingDataIO.load_from_hdf5(self.current_hdf5_path)
            
            if not beads_data:
                QMessageBox.information(self, "No Data", "No tracking data found.")
                return
            
            # Load into tracker
            self.tracker.load_from_data(beads_data)
            self.next_bead_id = max(bead['id'] for bead in beads_data) + 1
            
            # Display beads on current frame
            if self.video_widget:
                frame_idx = self.video_widget.controller.current_frame_index  # type: ignore
                bead_positions = {}
                for bead in beads_data:
                    if frame_idx < len(bead['positions']):
                        bead_positions[bead['id']] = bead['positions'][frame_idx]

                trace_history = self._build_trace_history(frame_idx)

                self.video_widget.set_tracking_enabled(True)  # type: ignore
                self.video_widget.update_bead_positions(  # type: ignore
                    bead_positions,
                    record_trace=False,
                    traces_override=trace_history
                )
            
            # Update UI
            self.is_validating = True
            if self.video_widget:
                self.video_widget.set_click_to_select_mode(True)  # type: ignore
            self.auto_detect_button.setEnabled(False)
            self.select_beads_button.setText("Add")
            self.start_tracking_button.setEnabled(True)
            self.clear_button.setEnabled(True)
            
            # Check completeness
            num_tracked = len(beads_data[0]['positions']) if beads_data else 0
            if self.video_widget:
                total_frames = self.video_widget.controller.get_total_frames()  # type: ignore
            else:
                total_frames = 0
            
            if num_tracked >= total_frames:
                self._update_status(f"Loaded {len(beads_data)} beads - Complete", "")
            else:
                self._update_status(f"Loaded {len(beads_data)} beads", f"{num_tracked}/{total_frames} frames")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")
    
    def _on_auto_detect_clicked(self):
        """Auto-detect beads in current frame."""
        if not self.video_widget:
            return
        assert self.video_widget is not None  # Type hint for checker
        
        current_frame = self.video_widget.get_current_frame()
        if current_frame is None:
            QMessageBox.warning(self, "No Frame", "No frame available.")
            return
        
        # Clear any existing tracking data and traces
        self.tracker.clear()
        self.next_bead_id = 0
        if self.video_widget:
            self.video_widget.bead_traces = {}  # type: ignore # Clear trace history
        
        # Detect beads
        self.detected_positions = detect_beads_auto(
            current_frame,
            min_area=self.min_size_spinbox.value(),
            max_area=self.max_size_spinbox.value(),
            threshold_value=self.threshold_spinbox.value()
        )
        
        if not self.detected_positions:
            QMessageBox.information(self, "No Beads", "No beads detected. Adjust parameters.")
            return
        
        # Add to tracker
        self.is_validating = True
        if self.video_widget:
            self.video_widget.set_tracking_enabled(True)  # type: ignore
            self.video_widget.set_click_to_select_mode(True)  # type: ignore
        
        bead_positions = {}
        for x, y in self.detected_positions:
            bead_id = self.next_bead_id
            try:
                self.tracker.add_bead(current_frame, x, y, bead_id)
            except ValueError as exc:
                Logger.warning(f"Skipping bead at ({x}, {y}): {exc}", "XY_TAB")
                continue
            bead_positions[bead_id] = (x, y)
            self.next_bead_id += 1
        
        # Update display with correct bead IDs (use tracker trace history)
        if self.video_widget:
            frame_idx = self.video_widget.controller.current_frame_index
            traces = self._build_trace_history(frame_idx)
            self.video_widget.update_bead_positions(bead_positions, record_trace=False, traces_override=traces)  # type: ignore
        
        # Update UI
        self.auto_detect_button.setEnabled(False)
        self.select_beads_button.setText("Add")
        self.start_tracking_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.show_traces_checkbox.setEnabled(True)

        self._update_status(f"{len(self.detected_positions)} beads detected", "R-click remove, L-click add")
    
    def _on_select_beads_clicked(self):
        """Manual bead selection (Add)."""
        if not self.is_selecting and not self.is_validating:
            # Start selection
            self.is_selecting = True
            self.select_beads_button.setText("Done")
            self.start_tracking_button.setEnabled(False)
            if self.video_widget:
                self.video_widget.set_click_to_select_mode(True)  # type: ignore
                self.video_widget.set_tracking_enabled(True)  # type: ignore
            self._update_status("Manual selection mode", "Click beads to add")
        else:
            # End selection
            self.is_selecting = False
            self.is_validating = False
            self.select_beads_button.setText("Add")
            if self.video_widget:
                self.video_widget.set_click_to_select_mode(False)  # type: ignore
            
            if len(self.tracker.beads) > 0:
                self.start_tracking_button.setEnabled(True)
                self.clear_button.setEnabled(True)
                self._update_status(f"{len(self.tracker.beads)} beads selected", "Ready to track")

    
    
    def _on_bead_clicked(self, x: int, y: int):
        """Add bead at clicked position."""

        if not self.is_selecting and not self.is_validating:
            return
        
        if not self.video_widget:
            return
        current_frame = self.video_widget.get_current_frame()  # type: ignore
        if current_frame is None:
            return
        
        # Add bead
        try:
            self.tracker.add_bead(current_frame, x, y, self.next_bead_id)
        except ValueError as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return
        self.next_bead_id += 1

        # Update display (use record_trace=False; traces will be built from tracker history)
        bead_positions = {bead['id']: bead['positions'][0] for bead in self.tracker.beads}
        traces = self._build_trace_history(0)
        if self.video_widget:
            self.video_widget.update_bead_positions(bead_positions, record_trace=False, traces_override=traces)  # type: ignore

        self._update_status(f"{len(self.tracker.beads)} beads", "Bead added")
    
    def _on_bead_right_clicked(self, x: int, y: int):
        """Remove bead at clicked position."""

        if not self.is_selecting and not self.is_validating:
            return
        
        # Find clicked bead
        bead_to_remove = None
        for bead in self.tracker.beads:
            bead_x, bead_y = bead['positions'][0]
            if np.sqrt((x - bead_x)**2 + (y - bead_y)**2) < 20:
                bead_to_remove = bead['id']
                break
        
        if bead_to_remove is not None:
            self.tracker.remove_bead(bead_to_remove)
            
            # Update display
            bead_positions = {bead['id']: bead['positions'][0] for bead in self.tracker.beads}
            if self.video_widget:
                self.video_widget.update_bead_positions(bead_positions)  # type: ignore
            
            self._update_status(f"{len(self.tracker.beads)} beads", "Bead removed")
            if len(self.tracker.beads) == 0:
                # nothing to disable (template UI removed)
                pass

    # Template-related functionality removed
    
    def _on_start_tracking_clicked(self):
        """Start or stop tracking."""
        if self.is_tracking:
            self._stop_tracking()
        else:
            self._start_tracking()
    
    def _on_pause_tracking_clicked(self):
        """Pause or resume tracking."""
        if self.is_paused:
            self._resume_tracking()
        else:
            self._pause_tracking()
    
    def _start_tracking(self):
        """Start tracking beads through video with auto-save."""
        if not self.video_widget or len(self.tracker.beads) == 0:
            return
        
        num_beads = len(self.tracker.beads)
        self.total_tracking_frames = self.video_widget.controller.get_total_frames()  # type: ignore
        
        # Resume from last tracked frame
        self.current_tracking_frame = len(self.tracker.beads[0]['positions']) if self.tracker.beads[0]['positions'] else 0
        if self.current_tracking_frame > 1:
            self.current_tracking_frame -= 1
        
        # Update UI
        self.is_tracking = True
        self.is_paused = False
        self.start_tracking_button.setText("Stop")
        self.start_tracking_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(True)
        self.select_beads_button.setEnabled(False)
        
        # Show initial status with bead count
        self._update_status(f"Starting: {num_beads} beads", f"Frame 0/{self.total_tracking_frames}")
        
        # Start timer (process one frame every 10ms for responsive UI)
        self.tracking_timer.start(10)
    
    def _pause_tracking(self):
        """Pause tracking."""
        self.is_paused = True
        self.tracking_timer.stop()
        self.pause_button.setText("Resume")
        num_beads = len(self.tracker.beads)
        self._update_status(f"Paused: {num_beads} beads", f"Frame {self.current_tracking_frame}/{self.total_tracking_frames}")
    
    def _resume_tracking(self):
        """Resume tracking."""
        self.is_paused = False
        self.pause_button.setText("Pause")
        self.tracking_timer.start(10)
        num_beads = len(self.tracker.beads)
        self._update_status(f"Resuming: {num_beads} beads", f"Frame {self.current_tracking_frame}/{self.total_tracking_frames}")
    
    def _process_next_frame(self):
        """Process multiple frames per timer tick for better performance."""
        if self.current_tracking_frame >= self.total_tracking_frames:
            # Tracking complete
            self._finish_tracking()
            return
        
        # Process multiple frames in one timer tick for better performance
        frames_to_process = min(self.frames_per_batch, self.total_tracking_frames - self.current_tracking_frame)
        
        for i in range(frames_to_process):
            if self.current_tracking_frame >= self.total_tracking_frames:
                break
                
            # Process frame
            if self.video_widget is not None:
                self.video_widget.controller.seek_to_frame(self.current_tracking_frame)
                frame = self.video_widget.get_current_frame()
            else:
                return
            
            if frame is not None:
                results = self.tracker.track_frame(frame)
                
                # Only update display on last frame of batch
                if i == frames_to_process - 1 and self.video_widget is not None:
                    bead_positions = {bid: (x, y) for bid, x, y in results}
                    # Use tracker-built trace history to avoid mismatch / double-appending
                    traces = self._build_trace_history(self.current_tracking_frame)
                    self.video_widget.update_bead_positions(bead_positions, record_trace=False, traces_override=traces)
            
            # Update status and save
            num_beads = len(self.tracker.beads)
            if self.current_tracking_frame % 100 == 0 and self.current_hdf5_path:
                self._save_tracking_to_hdf5()
                self._update_status(f"Tracking: {num_beads} beads", f"Frame {self.current_tracking_frame}/{self.total_tracking_frames} (saved)")
            elif self.current_tracking_frame % 10 == 0:
                self._update_status(f"Tracking: {num_beads} beads", f"Frame {self.current_tracking_frame}/{self.total_tracking_frames}")
            
            self.current_tracking_frame += 1
    
    def _finish_tracking(self):
        """Complete tracking and save final data."""
        self.tracking_timer.stop()
        
        # Final save
        if self.current_hdf5_path:
            self._save_tracking_to_hdf5()
        
        num_beads = len(self.tracker.beads)
        self._update_status(f"Complete: {num_beads} beads", f"All {self.total_tracking_frames} frames tracked")
        
        # Notify Preview tab with tracking data
        self._update_preview_tab()
        
    # Export/Save buttons removed (auto-save occurs during tracking)
        self._stop_tracking()
    
    def _update_preview_tab(self):
        """Update Preview tab with current tracking data."""
        if not self.video_widget:
            return
        
        # Get parent widget (AnalysisWidget) to access other tabs
        parent = self.parent()
        while parent and not hasattr(parent, 'tab_widget'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'tab_widget'):
            # Find Preview tab
            for i in range(parent.tab_widget.count()):  # type: ignore
                tab = parent.tab_widget.widget(i)  # type: ignore
                if hasattr(tab, 'load_tracking_data'):
                    # Convert tracker beads to format Preview tab expects
                    tracking_data = {}
                    for bead in self.tracker.beads:
                        bead_id = bead['id']
                        tracking_data[bead_id] = {
                            'positions': bead['positions'],
                            'initial_pos': bead.get('initial_pos'),
                        }
                    tab.load_tracking_data(tracking_data)  # type: ignore
                    Logger.info(f"Updated Preview tab with {len(tracking_data)} beads", "XY_TAB")
                    break
    
    def _stop_tracking(self):
        """Stop tracking."""
        self.tracking_timer.stop()
        self.is_tracking = False
        self.is_paused = False
        self.start_tracking_button.setText("Start Tracking")
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(False)
        self.select_beads_button.setEnabled(True)
        # no template button to re-enable
    
    def _save_tracking_to_hdf5(self):
        """Save tracking data to HDF5."""
        if not self.current_hdf5_path:
            Logger.debug("Cannot save - no HDF5 path", "XY_TAB")
            return
            
        if len(self.tracker.beads) == 0:
            Logger.debug("Cannot save - no beads in tracker", "XY_TAB")
            return
        
        Logger.info(f"Saving {len(self.tracker.beads)} beads", "XY_TAB")
        try:
            metadata = {
                'num_beads': len(self.tracker.beads),
                'num_frames': len(self.tracker.beads[0]['positions']) if self.tracker.beads else 0
            }
            
            # Temporarily close the video file to allow writing
            if self.video_widget and self.video_widget.controller and self.video_widget.controller.video_source:
                self.video_widget.controller.video_source.temporary_close_for_writing()
            
            try:
                TrackingDataIO.save_to_hdf5(
                    self.current_hdf5_path, 
                    self.tracker.beads, 
                    metadata
                )
            finally:
                # Reopen the video file
                if self.video_widget and self.video_widget.controller and self.video_widget.controller.video_source:
                    self.video_widget.controller.video_source.reopen_after_writing()
                    
        except Exception as e:
            Logger.error(f"Save error: {e}", "XY_TAB")
            import traceback
            traceback.print_exc()
    
    # Manual Save/Export UI removed - saving happens automatically during tracking
    
    def _on_clear_clicked(self):
        """Clear all tracking data."""
        reply = QMessageBox.question(
            self, "Clear",
            "Clear all tracked beads?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._reset_tracking()
    
    def _on_toggle_traces_clicked(self):
        """Toggle trace visibility on/off."""
        if not self.video_widget:
            return
        
        # Update the show_traces flag based on checkbox state
        self.video_widget.show_traces = self.show_traces_checkbox.isChecked()
        
        # Force redraw
        if self.video_widget.last_displayed_frame is not None:
            frame_index = self.video_widget.controller.current_frame_index
            self.video_widget._on_frame_changed(frame_index, self.video_widget.last_displayed_frame.copy())

    def _build_trace_history(self, frame_index: int) -> Dict[int, List[Tuple[int, int]]]:
        """Return bead traces up to the provided frame index."""
        traces: Dict[int, List[Tuple[int, int]]] = {}
        for bead in self.tracker.beads:
            positions = bead.get('positions', [])
            if not positions:
                continue

            cutoff = min(frame_index + 1, len(positions))
            if cutoff > 0:
                traces[bead['id']] = positions[:cutoff]

        return traces

    def _on_video_frame_changed(self, frame_index: int, _frame_data):
        """Refresh overlays when the video frame changes via scrubbing or playback."""
        if self.is_tracking:
            return

        if not self.video_widget or not self.video_widget.tracking_enabled:
            return

        if not self.tracker.beads:
            return

        bead_positions = {}
        for bead in self.tracker.beads:
            positions = bead.get('positions', [])
            if frame_index < len(positions):
                bead_positions[bead['id']] = positions[frame_index]

        trace_history = self._build_trace_history(frame_index)

        if not bead_positions and trace_history:
            bead_positions = {bead_id: trace[-1] for bead_id, trace in trace_history.items() if trace}

        # Update display without extending trace history
        if bead_positions or trace_history:
            self.video_widget.update_bead_positions(  # type: ignore
                bead_positions,
                record_trace=False,
                traces_override=trace_history
            )
        elif self.video_widget.bead_positions:
            self.video_widget.update_bead_positions({}, record_trace=False, traces_override={})  # type: ignore
    
    def _reset_tracking(self):
        """Reset tracking state."""
        self.tracker.clear()
        self.next_bead_id = 0
        self.is_tracking = False
        self.is_selecting = False
        self.is_validating = False
        self.detected_positions = []
        
        if self.video_widget:
            self.video_widget.bead_traces = {}  # Clear trace history
            self.video_widget.update_bead_positions({})
            self.video_widget.set_tracking_enabled(False)
            self.video_widget.set_click_to_select_mode(False)
        
        self.auto_detect_button.setEnabled(True)
        self.select_beads_button.setText("Add")
        self.start_tracking_button.setText("Start Tracking")
        self.start_tracking_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.show_traces_checkbox.setEnabled(False)
        # Save/Export/UI removed - traces are handled automatically
        self._clear_status()