"""XY Traces tab for bead tracking with HDF5 integration."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QMessageBox, QFileDialog, QSpinBox, QGroupBox, QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer
import numpy as np
from pathlib import Path

from src.analysis import BeadTracker, detect_beads_auto
from src.utils.tracking_io import TrackingDataIO
from src.utils.logger import Logger

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from src.ui.video_widget import VideoWidget


class XYTracesTab(QWidget):
    """Tab for XY bead tracking with auto-detection and HDF5 storage."""

    def __init__(self):
        """Initialize XY traces tab."""
        super().__init__()
        self.video_widget: Optional['VideoWidget'] = None
        self.tracker = BeadTracker(window_size=40)
        self.is_tracking = False
        self.is_paused = False
        self.is_selecting = False
        self.is_validating = False
        self.is_adding_template = False
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
        settings_layout.setLabelAlignment(Qt.AlignRight)  # type: ignore
        
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
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(8)
        
        # Row 1: Detection buttons
        button_row1 = QHBoxLayout()
        button_row1.setSpacing(8)
        
        self.load_tracking_button = QPushButton("Load Saved")
        self.load_tracking_button.setEnabled(False)
        self.load_tracking_button.setFixedWidth(80)
        self.load_tracking_button.clicked.connect(self._on_load_tracking_clicked)
        button_row1.addWidget(self.load_tracking_button)
        
        self.auto_detect_button = QPushButton("Auto Detect")
        self.auto_detect_button.setEnabled(False)
        self.auto_detect_button.setFixedWidth(90)
        self.auto_detect_button.clicked.connect(self._on_auto_detect_clicked)
        button_row1.addWidget(self.auto_detect_button)
        
        self.select_beads_button = QPushButton("Manual")
        self.select_beads_button.setEnabled(False)
        self.select_beads_button.setFixedWidth(70)
        self.select_beads_button.clicked.connect(self._on_select_beads_clicked)
        button_row1.addWidget(self.select_beads_button)

        self.add_template_button = QPushButton("Add Template")
        self.add_template_button.setEnabled(False)
        self.add_template_button.setCheckable(True)
        self.add_template_button.setFixedWidth(110)
        self.add_template_button.toggled.connect(self._on_add_template_mode_toggled)
        button_row1.addWidget(self.add_template_button)
        
        button_row1.addStretch()
        controls_layout.addLayout(button_row1)
        
        # Row 2: Action buttons
        button_row2 = QHBoxLayout()
        button_row2.setSpacing(8)
        
        self.start_tracking_button = QPushButton("Start Tracking")
        self.start_tracking_button.setEnabled(False)
        self.start_tracking_button.setFixedWidth(100)
        self.start_tracking_button.clicked.connect(self._on_start_tracking_clicked)
        button_row2.addWidget(self.start_tracking_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.setFixedWidth(60)
        self.pause_button.clicked.connect(self._on_pause_tracking_clicked)
        button_row2.addWidget(self.pause_button)
        
        self.save_button = QPushButton("Save")
        self.save_button.setEnabled(False)
        self.save_button.setFixedWidth(60)
        self.save_button.clicked.connect(self._on_save_to_hdf5_clicked)
        button_row2.addWidget(self.save_button)
        
        self.export_button = QPushButton("Export CSV")
        self.export_button.setEnabled(False)
        self.export_button.setFixedWidth(80)
        self.export_button.clicked.connect(self._on_export_clicked)
        button_row2.addWidget(self.export_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.setFixedWidth(60)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_row2.addWidget(self.clear_button)
        
        self.show_traces_checkbox = QCheckBox("Show Traces")
        self.show_traces_checkbox.setEnabled(False)
        self.show_traces_checkbox.setChecked(True)  # Traces shown by default
        self.show_traces_checkbox.stateChanged.connect(self._on_toggle_traces_clicked)
        button_row2.addWidget(self.show_traces_checkbox)
        
        button_row2.addStretch()
        controls_layout.addLayout(button_row2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Status display - matching Info tab style exactly
        status_group = QGroupBox("Status")
        self.status_layout = QFormLayout(status_group)
        self.status_layout.setContentsMargins(8, 8, 8, 8)
        self.status_layout.setSpacing(5)
        self.status_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(status_group)
        
        layout.addStretch()
    
    def _add_status_row(self, label_text, value_text=""):
        """Add a label-value row to the status layout (matching info tab style)."""
        # Label
        label = QLabel(label_text)
        label.setStyleSheet("color: #666;")
        
        # Value
        value = QLabel(value_text)
        value.setWordWrap(True)
        value.setStyleSheet("color: #222;")
        
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

                self.video_widget.set_tracking_enabled(True)  # type: ignore
                self.video_widget.update_bead_positions(bead_positions)  # type: ignore
            
            # Update UI
            self.is_validating = True
            if self.video_widget:
                self.video_widget.set_click_to_select_mode(True)  # type: ignore
            self.auto_detect_button.setEnabled(False)
            self.select_beads_button.setText("Add")
            self.start_tracking_button.setEnabled(True)
            self.clear_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.add_template_button.setEnabled(True)
            
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
        
        # Update display with correct bead IDs
        if self.video_widget:
            self.video_widget.update_bead_positions(bead_positions)  # type: ignore
        
        # Update UI
        self.auto_detect_button.setEnabled(False)
        self.select_beads_button.setText("Add")
        self.start_tracking_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.show_traces_checkbox.setEnabled(True)
        self.add_template_button.setEnabled(True)
        
        self._update_status(f"{len(self.detected_positions)} beads detected", "R-click remove, L-click add")
    
    def _on_select_beads_clicked(self):
        """Manual bead selection."""
        if self.add_template_button.isChecked():
            self.add_template_button.setChecked(False)

        if not self.is_selecting and not self.is_validating:
            # Start selection
            self.is_selecting = True
            self.select_beads_button.setText("Done")
            self.start_tracking_button.setEnabled(False)
            if self.video_widget:
                self.video_widget.set_click_to_select_mode(True)  # type: ignore
                self.video_widget.set_tracking_enabled(True)  # type: ignore
            self.add_template_button.setEnabled(False)
            self._update_status("Manual selection mode", "Click beads to add")
        else:
            # End selection
            self.is_selecting = False
            self.is_validating = False
            self.select_beads_button.setText("Manual")
            if self.video_widget:
                self.video_widget.set_click_to_select_mode(False)  # type: ignore
            
            if len(self.tracker.beads) > 0:
                self.start_tracking_button.setEnabled(True)
                self.clear_button.setEnabled(True)
                self.add_template_button.setEnabled(True)
                self._update_status(f"{len(self.tracker.beads)} beads selected", "Ready to track")

    def _on_add_template_mode_toggled(self, checked: bool):
        """Toggle focus template capture mode."""
        if checked:
            if self.is_selecting:
                QMessageBox.information(self, "Finish Selection", "Finish manual selection before adding templates.")
                self.add_template_button.setChecked(False)
                return

            if not self.tracker.beads:
                QMessageBox.warning(self, "No Beads", "Add beads before capturing templates.")
                self.add_template_button.setChecked(False)
                return

            self.is_adding_template = True
            if self.video_widget:
                self.video_widget.set_click_to_select_mode(True)  # type: ignore
            bead_count = len(self.tracker.beads)
            self._update_status(f"Template mode: {bead_count} beads", "Click bead focus appearance to store template")
        else:
            self.is_adding_template = False
            if not self.is_selecting and not self.is_validating and self.video_widget:
                self.video_widget.set_click_to_select_mode(False)  # type: ignore
    
    def _on_bead_clicked(self, x: int, y: int):
        """Add bead at clicked position."""
        if self.is_adding_template:
            self._add_template_from_click(x, y)
            return

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
            QMessageBox.warning(self, "Template Error", str(exc))
            return
        self.next_bead_id += 1
        self.add_template_button.setEnabled(True)
        
        # Update display
        bead_positions = {bead['id']: bead['positions'][0] for bead in self.tracker.beads}
        self.video_widget.update_bead_positions(bead_positions)  # type: ignore
        
        self._update_status(f"{len(self.tracker.beads)} beads", "Bead added")
    
    def _on_bead_right_clicked(self, x: int, y: int):
        """Remove bead at clicked position."""
        if self.is_adding_template:
            return

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
                self.add_template_button.setChecked(False)
                self.add_template_button.setEnabled(False)

    def _add_template_from_click(self, x: int, y: int) -> None:
        """Capture an additional template for the nearest bead."""
        if not self.video_widget:
            return

        current_frame = self.video_widget.get_current_frame()
        if current_frame is None:
            return

        bead = self._find_nearest_bead(x, y)
        if bead is None:
            self._update_status("Template mode", "Click closer to an existing bead")
            return

        bead_id = bead['id']
        try:
            self.tracker.add_focus_template(current_frame, bead_id, x, y)
            template_count = len(bead.get('templates', []))
            self._update_status(f"Template saved for bead {bead_id}", f"Stored templates: {template_count}")
        except ValueError as exc:
            QMessageBox.warning(self, "Template Error", str(exc))

    def _find_nearest_bead(self, x: int, y: int, max_distance: float = 25.0):
        """Return the bead closest to the provided coordinates within a threshold."""
        nearest_bead = None
        min_distance = float('inf')
        for bead in self.tracker.beads:
            bead_x, bead_y = bead['positions'][-1] if bead['positions'] else (None, None)
            if bead_x is None or bead_y is None:
                continue
            distance = float(np.hypot(x - bead_x, y - bead_y))
            if distance < min_distance:
                min_distance = distance
                nearest_bead = bead

        if nearest_bead is not None and min_distance <= max_distance:
            return nearest_bead
        return None
    
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
        self.add_template_button.setChecked(False)
        self.add_template_button.setEnabled(False)
        
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
                    self.video_widget.update_bead_positions(bead_positions)
            
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
        
        self.export_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self._stop_tracking()
    
    def _stop_tracking(self):
        """Stop tracking."""
        self.tracking_timer.stop()
        self.is_tracking = False
        self.is_paused = False
        self.start_tracking_button.setText("Start Tracking")
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(False)
        self.select_beads_button.setEnabled(True)
        if len(self.tracker.beads) > 0:
            self.add_template_button.setEnabled(True)
    
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
    
    def _on_save_to_hdf5_clicked(self):
        """Manual save to HDF5."""
        if not self.current_hdf5_path:
            QMessageBox.warning(self, "No File", "No HDF5 file loaded.")
            return
        
        if len(self.tracker.beads) == 0:
            # Check if data exists in HDF5 file already
            if TrackingDataIO.has_tracking_data(self.current_hdf5_path):
                QMessageBox.information(
                    self, "Already Saved",
                    f"Tracking data already saved to:\n/analysed_data/xy_tracking"
                )
            else:
                QMessageBox.warning(self, "No Data", "No tracking data to save.\n\nPlease:\n1. Click 'Auto' or 'Manual' to detect beads\n2. Click 'Start Tracking' to track them")
            return
        
        try:
            self._save_tracking_to_hdf5()
            num_frames = len(self.tracker.beads[0]['positions']) if self.tracker.beads else 0
            QMessageBox.information(
                self, "Saved",
                f"Saved to /analysed_data/xy_tracking\n\nBeads: {len(self.tracker.beads)}\nFrames: {num_frames}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")
    
    def _on_export_clicked(self):
        """Export tracking data to CSV."""
        if not self.current_hdf5_path:
            QMessageBox.warning(self, "No File", "No HDF5 file loaded.")
            return
        
        # Check if we have data in tracker or in HDF5 file
        has_data_in_memory = len(self.tracker.beads) > 0
        has_data_in_file = TrackingDataIO.has_tracking_data(self.current_hdf5_path)
        
        if not has_data_in_memory and not has_data_in_file:
            QMessageBox.warning(self, "No Data", "No tracking data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            str(Path(self.current_hdf5_path).with_suffix('.csv')),
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Save to HDF5 first if we have data in memory
                if has_data_in_memory:
                    self._save_tracking_to_hdf5()
                
                # Export from HDF5 to CSV
                TrackingDataIO.export_to_csv(self.current_hdf5_path, file_path)
                QMessageBox.information(self, "Exported", f"Exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
    
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

        # Update display without extending trace history
        if bead_positions:
            self.video_widget.update_bead_positions(bead_positions, record_trace=False)  # type: ignore
        elif self.video_widget.bead_positions:
            self.video_widget.update_bead_positions({}, record_trace=False)  # type: ignore
    
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
        self.select_beads_button.setText("Manual")
        self.start_tracking_button.setText("Start Tracking")
        self.start_tracking_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.show_traces_checkbox.setEnabled(False)
        self.export_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.add_template_button.setChecked(False)
        self.add_template_button.setEnabled(False)
        self._clear_status()