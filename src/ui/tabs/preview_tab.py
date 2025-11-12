"""Preview tab for visualizing and selecting tracked beads."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QGroupBox, QSplitter, QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtWidgets import QApplication
from typing import Optional, TYPE_CHECKING
import numpy as np
import cv2

if TYPE_CHECKING:
    from src.ui.video_widget import VideoWidget


class ResizablePixmapLabel(QLabel):
    """QLabel that emits a signal when resized."""
    resized = pyqtSignal()

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self.resized.emit()


class PreviewTab(QWidget):
    """Tab for previewing and selecting tracked beads for analysis."""

    def __init__(self):
        """Initialize preview tab."""
        super().__init__()
        # Ensure this tab uses the application's Window palette color so it matches other tabs
        app = QApplication.instance()
        if app is not None:
            pal = self.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))  # type: ignore
            self.setPalette(pal)
            self.setAutoFillBackground(True)

        self.video_widget: Optional['VideoWidget'] = None
        self.tracked_beads = {}  # Will store {bead_id: {data, selected}}

        self._init_ui()

    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)  # type: ignore

        # Left side: Bead list with checkboxes
        left_panel = self._create_bead_list_panel()
        splitter.addWidget(left_panel)

        # Right side: Plots
        right_panel = self._create_plots_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)

    def _create_bead_list_panel(self):
        """Create the left panel with bead list and checkboxes."""
        panel = QGroupBox("Tracked Beads")
        layout = QVBoxLayout(panel)

        # Bead list widget
        self.bead_list = QListWidget()
        self.bead_list.setAlternatingRowColors(True)
        self.bead_list.currentItemChanged.connect(self._on_bead_selected)
        layout.addWidget(self.bead_list)

        # Selection controls
        controls_layout = QHBoxLayout()

        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.stateChanged.connect(self._toggle_select_all)
        controls_layout.addWidget(self.select_all_checkbox)

        controls_layout.addStretch()

        # Count label
        self.count_label = QLabel("0 beads loaded")
        self.count_label.setStyleSheet("color: #666;")
        controls_layout.addWidget(self.count_label)

        layout.addLayout(controls_layout)

        return panel

    def _create_plots_panel(self):
        """Create the right panel with XY and Z plots."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top: XY trajectory plot
        xy_group = QGroupBox("XY Trajectory")
        xy_layout = QVBoxLayout(xy_group)
        xy_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore
        xy_layout.setContentsMargins(3, 3, 3, 3)  # Minimal group box padding
        xy_layout.setSpacing(0)  # No spacing between widgets

        # Use resizable label so we can redraw at new sizes
        self.xy_plot_label = ResizablePixmapLabel()
        # Align pixmap center inside the group box
        self.xy_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.xy_plot_label.setStyleSheet(
            "background-color: #f0f0f0; "
            "border: none;"
        )
        # Make the label expand to fill all available space
        self.xy_plot_label.setMinimumHeight(100)
        self.xy_plot_label.setScaledContents(False)
        self.xy_plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.xy_plot_label.resized.connect(self._on_plot_resize)
        xy_layout.addWidget(self.xy_plot_label, 1)  # stretch factor 1

        layout.addWidget(xy_group)

        # Bottom: Z/Voltage vs Time plot (placeholder)
        z_group = QGroupBox("Z Position / Voltage vs Time")
        z_layout = QVBoxLayout(z_group)

        self.z_plot_label = QLabel()
        self.z_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore
        self.z_plot_label.setStyleSheet(
            "background-color: #f0f0f0; "
            "border: none;"
        )
        self.z_plot_label.setMinimumHeight(200)
        self.z_plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        z_layout.addWidget(self.z_plot_label)

        layout.addWidget(z_group)

        # Set vertical stretch: xy 1, z 2 (fills the space together without resizing the outer box)
        layout.setStretch(0, 1)
        layout.setStretch(1, 2)

        # Ensure the XY group will try to be as large as possible within its space
        xy_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        return panel

    def _on_bead_selected(self, current, previous):
        """Handle bead selection from list."""
        if current is None:
            # Clear plot when no bead selected
            self.xy_plot_label.clear()
            return

        # Get bead data from the item
        bead_id = current.data(Qt.UserRole)  # type: ignore

        # Plot XY trajectory
        if bead_id in self.tracked_beads:
            self._plot_xy_trajectory(bead_id, self.tracked_beads[bead_id]['data'])

    def _on_plot_resize(self):
        """Called when the xy plot label is resized; redraw current bead if any"""
        current_item = self.bead_list.currentItem()
        if current_item is None:
            return
        bead_id = current_item.data(Qt.UserRole)  # type: ignore
        if bead_id in self.tracked_beads:
            self._plot_xy_trajectory(bead_id, self.tracked_beads[bead_id]['data'])

    def _plot_xy_trajectory(self, bead_id, bead_data):
        """Plot XY trajectory for a bead.

        Args:
            bead_id: Bead identifier
            bead_data: Dictionary with 'positions' key containing [(x, y), ...] list
        """
        positions = bead_data.get('positions', [])
        if not positions or len(positions) < 2:
            self.xy_plot_label.clear()
            return

        # Extract x and y coordinates
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        # Determine available drawing area from the label and create the largest square image that fits
        avail_w = max(50, self.xy_plot_label.width())
        avail_h = max(50, self.xy_plot_label.height())
        # Use available height as base (Y-axis fills completely), make square
        base_size = max(50, avail_h) if avail_h > 0 else 100
        # Use 3x resolution for better anti-aliasing and smoothness
        size = base_size * 3
        bg_color = (240, 240, 240)  # Match UI grey background
        plot_img = np.ones((size, size, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)

        # Calculate plot area (maximize the plotting area, ensure space for labels and ticks)
        margin_left = 180
        margin_right = 30
        margin_top = 30
        margin_bottom = 120
        plot_width = size - margin_left - margin_right
        plot_height = size - margin_top - margin_bottom

        # Find data range
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add padding to ranges
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Make the data view square by equalizing ranges (use the larger range for both axes)
        x_range_padded = x_max - x_min
        y_range_padded = y_max - y_min
        if x_range_padded > y_range_padded:
            # X is bigger, expand Y to match
            diff = x_range_padded - y_range_padded
            y_min -= diff / 2
            y_max += diff / 2
        elif y_range_padded > x_range_padded:
            # Y is bigger, expand X to match
            diff = y_range_padded - x_range_padded
            x_min -= diff / 2
            x_max += diff / 2

        # Compute a single scale to ensure 1:1 pixel scaling on X and Y
        data_w = x_max - x_min if (x_max - x_min) != 0 else 1
        data_h = y_max - y_min if (y_max - y_min) != 0 else 1
        scale_x = plot_width / data_w
        scale_y = plot_height / data_h
        scale = min(scale_x, scale_y)

        # Determine used plot pixel size to preserve equal scaling and center horizontally, top-aligned vertically
        used_w = int(round(data_w * scale))
        used_h = int(round(data_h * scale))
        offset_x = margin_left + (plot_width - used_w) // 2
        offset_y = margin_top  # top-aligned as requested

        # Convert data coordinates to pixel coordinates using same scale for x and y
        # Note: Y-axis is NOT inverted to match video coordinate system (origin top-left)
        def data_to_pixel(x, y):
            px = int(offset_x + (x - x_min) * scale)
            py = int(offset_y + (y - y_min) * scale)
            return px, py

        # Draw border around the used plotting area (thicker for visibility)
        cv2.rectangle(plot_img, (offset_x, offset_y), (offset_x + used_w, offset_y + used_h), (60, 60, 60), 3)

        # Draw trajectory line (blue) with anti-aliasing and thicker line
        trajectory_color = (200, 100, 50)  # Blue (BGR)
        for i in range(len(positions) - 1):
            pt1 = data_to_pixel(x_coords[i], y_coords[i])
            pt2 = data_to_pixel(x_coords[i + 1], y_coords[i + 1])
            cv2.line(plot_img, pt1, pt2, trajectory_color, 3, cv2.LINE_AA)

        # Mark start point (green)
        start_pt = data_to_pixel(x_coords[0], y_coords[0])
        # OpenCV uses BGR - green is (0, 255, 0)
        cv2.circle(plot_img, start_pt, 9, (0, 255, 0), -1, cv2.LINE_AA)
        
        # Mark end point (red)
        end_pt = data_to_pixel(x_coords[-1], y_coords[-1])
        # OpenCV uses BGR - red is (0, 0, 255)
        cv2.circle(plot_img, end_pt, 9, (0, 0, 255), -1, cv2.LINE_AA)
        
        # Draw tick marks on axes (larger ticks at min/max, smaller intermediate ticks)
        tick_length_major = 25
        tick_length_minor = 15
        tick_color = (0, 0, 0)
        tick_thickness_major = 5
        tick_thickness_minor = 3
        
        # X-axis major ticks at min and max (extending downward from bottom axis)
        cv2.line(plot_img, (offset_x, offset_y + used_h), (offset_x, offset_y + used_h + tick_length_major), tick_color, tick_thickness_major, cv2.LINE_AA)
        cv2.line(plot_img, (offset_x + used_w, offset_y + used_h), (offset_x + used_w, offset_y + used_h + tick_length_major), tick_color, tick_thickness_major, cv2.LINE_AA)
        
        # X-axis minor ticks (at 25%, 50%, 75%)
        for fraction in [0.25, 0.5, 0.75]:
            x_pos = int(offset_x + used_w * fraction)
            cv2.line(plot_img, (x_pos, offset_y + used_h), (x_pos, offset_y + used_h + tick_length_minor), tick_color, tick_thickness_minor, cv2.LINE_AA)
        
        # Y-axis major ticks at min and max (extending leftward from left axis)
        cv2.line(plot_img, (offset_x - tick_length_major, offset_y), (offset_x, offset_y), tick_color, tick_thickness_major, cv2.LINE_AA)
        cv2.line(plot_img, (offset_x - tick_length_major, offset_y + used_h), (offset_x, offset_y + used_h), tick_color, tick_thickness_major, cv2.LINE_AA)
        
        # Y-axis minor ticks (at 25%, 50%, 75%)
        for fraction in [0.25, 0.5, 0.75]:
            y_pos = int(offset_y + used_h * fraction)
            cv2.line(plot_img, (offset_x - tick_length_minor, y_pos), (offset_x, y_pos), tick_color, tick_thickness_minor, cv2.LINE_AA)

        # Convert BGR -> RGB for QImage
        rgb_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)

        # Convert to QPixmap
        height, width, channel = rgb_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Scale down to actual display size (square based on height) with smooth transformation for anti-aliasing
        pixmap = QPixmap.fromImage(q_img).scaled(base_size, base_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # type: ignore

        # Use QPainter to draw crisp Arial text (axis labels, min/max ticks)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setPen(QColor(0, 0, 0))

        # Scale factors for text positioning (since we're drawing on the downscaled pixmap)
        scale_factor = base_size / size
        offset_x_scaled = int(offset_x * scale_factor)
        offset_y_scaled = int(offset_y * scale_factor)
        used_w_scaled = int(used_w * scale_factor)
        used_h_scaled = int(used_h * scale_factor)

        # Axis labels (Arial, slightly smaller) - centered on the axis, close to the axis line
        label_font = QFont("Arial", 10)
        painter.setFont(label_font)
        fm_label = painter.fontMetrics()
        x_label = "X-position (pixels)"
        x_w = fm_label.horizontalAdvance(x_label)
        # Center X label on the x-axis (horizontal center of plot area), below tick marks
        x_center = offset_x_scaled + used_w_scaled // 2
        painter.drawText(x_center - x_w // 2, offset_y_scaled + used_h_scaled + 35, x_label)

        y_label = "Y-position (pixels)"
        # Draw Y label vertically, centered on the y-axis (vertical center of plot area), left of ticks
        painter.save()
        y_center = offset_y_scaled + used_h_scaled // 2
        painter.translate(30, y_center)
        painter.rotate(-90)
        painter.drawText(-fm_label.horizontalAdvance(y_label) // 2, 0, y_label)
        painter.restore()

        # Tick labels (min/max) using same label font - positioned clearly visible
        painter.setFont(label_font)
        # X axis min/max - below the tick marks with enough space
        x_min_text = f"{x_min:.0f}"
        x_max_text = f"{x_max:.0f}"
        painter.drawText(offset_x_scaled - fm_label.horizontalAdvance(x_min_text) // 2, offset_y_scaled + used_h_scaled + 25, x_min_text)
        painter.drawText(offset_x_scaled + used_w_scaled - fm_label.horizontalAdvance(x_max_text) // 2, offset_y_scaled + used_h_scaled + 25, x_max_text)
        # Y axis min/max - left of the tick marks with enough space
        y_min_text = f"{y_min:.0f}"
        y_max_text = f"{y_max:.0f}"
        painter.drawText(offset_x_scaled - fm_label.horizontalAdvance(y_min_text) - 15, offset_y_scaled + used_h_scaled + 5, y_min_text)
        painter.drawText(offset_x_scaled - fm_label.horizontalAdvance(y_max_text) - 15, offset_y_scaled + fm_label.ascent(), y_max_text)

        painter.end()

        # Set pixmap at the proper size so it fits the available slot; do not force the label size
        self.xy_plot_label.setPixmap(pixmap)

    def _toggle_select_all(self, state):
        """Toggle selection of all beads."""
        is_checked = state == Qt.CheckState.Checked  # type: ignore

        for i in range(self.bead_list.count()):
            item = self.bead_list.item(i)
            if item:
                if is_checked:
                    item.setCheckState(Qt.CheckState.Checked)  # type: ignore
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)  # type: ignore

    def load_tracking_data(self, tracking_data):
        """Load tracked bead data from HDF5 file.

        Args:
            tracking_data: Dictionary with bead tracking information
                          {bead_id: {'positions': [(x, y), ...], ...}}
        """
        self.bead_list.clear()
        self.tracked_beads = {}

        if not tracking_data:
            self.count_label.setText("0 beads loaded")
            return

        # Populate list with beads (no frame counts shown)
        for bead_id, bead_data in tracking_data.items():
            # Store bead data
            self.tracked_beads[bead_id] = {
                'data': bead_data,
                'selected': True  # Default to selected
            }

            # Create list item with checkbox (display bead_id + 1 to match tracking view)
            item = QListWidgetItem(f"Bead {bead_id + 1}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)  # type: ignore
            item.setCheckState(Qt.CheckState.Checked)  # type: ignore
            item.setData(Qt.UserRole, bead_id)  # type: ignore

            self.bead_list.addItem(item)

        # Update count
        count = len(tracking_data)
        self.count_label.setText(f"{count} bead{'s' if count != 1 else ''} loaded")

        # Update select all checkbox
        self.select_all_checkbox.setChecked(True)

    def get_selected_beads(self):
        """Get list of selected bead IDs.

        Returns:
            List of bead IDs that are checked
        """
        selected = []
        for i in range(self.bead_list.count()):
            item = self.bead_list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:  # type: ignore
                bead_id = item.data(Qt.UserRole)  # type: ignore
                selected.append(bead_id)
        return selected

    def on_video_loaded(self):
        """Called when a new video is loaded."""
        # Clear current data
        self.bead_list.clear()
        self.tracked_beads = {}
        self.count_label.setText("0 beads loaded")
        self.xy_plot_label.clear()
        self.z_plot_label.clear()
