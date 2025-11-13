"""Preview tab - optimized implementation.

Provides a bead list and three plots:
- XY trajectory (centered at origin)
- X/Y vs time (combined)
- Z/Voltage vs time (combined)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QGroupBox, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QPixmap, QImage
from PyQt5.QtWidgets import QApplication
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2


class ResizablePixmapLabel(QLabel):
    """QLabel that emits a signal when resized."""
    resized = pyqtSignal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


class PreviewTab(QWidget):
    """Preview tab with bead list and trajectory/time plots."""
    
    # Constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_LABEL = 0.45
    FONT_TICK = 0.35
    FONT_SMALL = 0.3
    LINE_THICKNESS = 1
    LINE_AA = cv2.LINE_AA
    
    def __init__(self):
        super().__init__()
        self._setup_palette()
        self.video_widget: Optional = None
        self.tracked_beads: Dict = {}
        # Shared timeline (seconds) and optional function generator values (global)
        self.shared_times: Optional[np.ndarray] = None
        self.function_generator_timeline: Optional[np.ndarray] = None
        self._init_ui()

    def _setup_palette(self):
        """Configure widget palette to match app theme."""
        app = QApplication.instance()
        if app:
            pal = self.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            self.setPalette(pal)
            self.setAutoFillBackground(True)

    def _init_ui(self):
        """Initialize UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_bead_list_panel())
        splitter.addWidget(self._create_plots_panel())
        splitter.setSizes([260, 640])
        layout.addWidget(splitter)

    def _create_bead_list_panel(self) -> QGroupBox:
        """Create bead list panel with checkboxes."""
        gb = QGroupBox("Tracked Beads")
        layout = QVBoxLayout(gb)
        
        self.bead_list = QListWidget()
        self.bead_list.currentItemChanged.connect(self._on_bead_selected)
        layout.addWidget(self.bead_list)
        
        # Controls row
        controls = QHBoxLayout()
        self.select_all = QCheckBox("Select All")
        self.select_all.stateChanged.connect(self._toggle_select_all)
        controls.addWidget(self.select_all)
        controls.addStretch()
        
        self.count_label = QLabel("0 beads loaded")
        controls.addWidget(self.count_label)
        layout.addLayout(controls)
        
        return gb

    def _create_plots_panel(self) -> QGroupBox:
        """Create panel with three equal-height plots."""
        gb = QGroupBox()
        layout = QVBoxLayout(gb)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Create three plot labels
        self.xy_label = self._create_plot_label(150)
        self.time_xy_label = self._create_plot_label(150)
        self.time_zv_label = self._create_plot_label(150)
        
        layout.addWidget(self.xy_label, 1)
        layout.addWidget(self.time_xy_label, 1)
        layout.addWidget(self.time_zv_label, 1)
        
        return gb
    
    def _create_plot_label(self, min_height: int) -> ResizablePixmapLabel:
        """Create a resizable plot label."""
        label = ResizablePixmapLabel()
        label.setMinimumHeight(min_height)
        label.setScaledContents(False)
        label.resized.connect(self._on_plot_resize)
        return label

    def set_video_widget(self, vw):
        """Set video widget reference for FPS access."""
        self.video_widget = vw

    def load_tracking_data(self, tracking_data: Dict):
        """Load bead tracking data.
        
        Args:
            tracking_data: Dict of bead_id -> {positions, z, voltage, stuck}
        """
        self.bead_list.clear()
        self.tracked_beads = {}
        
        for bead_id, data in tracking_data.items():
            self.tracked_beads[bead_id] = {
                'data': data,
                'stuck': bool(data.get('stuck', False))
            }
            
            text = f"Bead {bead_id + 1}"
            if self.tracked_beads[bead_id]['stuck']:
                text += " 🔒"
                
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, bead_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.bead_list.addItem(item)
        
        count = len(tracking_data)
        self.count_label.setText(f"{count} bead{'s' if count != 1 else ''} loaded")
        self.select_all.setChecked(True)

        # Try to read a global function generator timeline from the opened HDF5
        # (located at /raw_data/function_generator_timeline when present).
        self.shared_times = None
        self.function_generator_timeline = None
        try:
            if self.video_widget and hasattr(self.video_widget, 'get_hdf5_file'):
                h5 = self.video_widget.get_hdf5_file()
                if h5 is not None:
                    # Look under raw_data then data for backward compatibility
                    data_group = None
                    if 'raw_data' in h5:
                        data_group = h5['raw_data']
                    elif 'data' in h5:
                        data_group = h5['data']

                    if data_group is not None and 'function_generator_timeline' in data_group:
                        ds = data_group['function_generator_timeline'][:]
                        # Dataset can be either 1D (values per frame) or Nx2 (time, value)
                        if ds is not None and ds.size > 0:
                            arr = np.array(ds)
                            if arr.ndim == 1:
                                # treat as values per frame -> create time axis from video fps
                                self.function_generator_timeline = arr.astype(float)
                                fps = 1.0
                                try:
                                    if self.video_widget and hasattr(self.video_widget, 'controller'):
                                        fps = float(self.video_widget.controller.get_fps())
                                except Exception:
                                    pass
                                self.shared_times = np.arange(len(arr)) / max(1.0, fps)
                            elif arr.ndim == 2 and arr.shape[1] >= 2:
                                # assume columns are (time, value)
                                self.shared_times = arr[:, 0].astype(float)
                                self.function_generator_timeline = arr[:, 1].astype(float)
        except Exception:
            # Non-fatal: timeline optional
            self.shared_times = None
            self.function_generator_timeline = None

    def _on_bead_selected(self, current, previous=None):
        """Handle bead selection change."""
        if current:
            self._update_plots_for_bead(current.data(Qt.UserRole))

    def _toggle_select_all(self, state):
        """Toggle all bead checkboxes."""
        checked = (state == Qt.CheckState.Checked)
        for i in range(self.bead_list.count()):
            self.bead_list.item(i).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def _on_plot_resize(self):
        """Redraw plots on resize."""
        # Compute target height: keep top plot as reference and slightly reduce lower two
        try:
            panel = self.xy_label.parentWidget()  # the QGroupBox containing the plots
            panel_h = panel.height() if panel is not None else self.height()
            # divide into 3 rows; keep xy plot as reference and make lower two a bit smaller
            h_third = max(20, int(panel_h / 3))
            # leave ~6% extra space for axis labels by making lower plots slightly smaller
            lower_factor = 0.94
            small = max(18, int(h_third * lower_factor))

            # Apply fixed height to each plot label
            self.xy_label.setFixedHeight(h_third)
            self.xy_label.setMinimumHeight(h_third)
            self.time_xy_label.setFixedHeight(small)
            self.time_xy_label.setMinimumHeight(small)
            self.time_zv_label.setFixedHeight(small)
            self.time_zv_label.setMinimumHeight(small)
        except Exception:
            # If anything goes wrong, fall back to previous behaviour (no crash)
            pass

        current = self.bead_list.currentItem()
        if current:
            self._update_plots_for_bead(current.data(Qt.UserRole))

    def _update_plots_for_bead(self, bead_id):
        """Update all plots for selected bead."""
        bead = self.tracked_beads.get(bead_id)
        if not bead:
            return
            
        data = bead['data']
        self._draw_xy(data.get('positions', []))
        self._draw_time_xy(data.get('positions', []))
        self._draw_time_zv(data.get('z', []), data.get('voltage', []))

    @staticmethod
    def _create_transparent_buffer(w: int, h: int) -> np.ndarray:
        """Create transparent RGBA buffer."""
        buf = np.ones((h, w, 4), dtype=np.uint8) * 255
        buf[:, :, 3] = 0
        return buf
    
    @staticmethod
    def _buffer_to_pixmap(buf: np.ndarray) -> QPixmap:
        """Convert BGRA buffer to QPixmap."""
        img = cv2.cvtColor(buf, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(img.data, img.shape[1], img.shape[0], 
                     4 * img.shape[1], QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg)
    
    def _draw_rotated_text(self, buf: np.ndarray, text: str, x: int, y: int,
                          font_scale: float = FONT_LABEL,
                          color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                          rotate_angle: int = 90):
        """Draw rotated text on buffer with optional color and rotation.

        color is BGRA tuple. rotate_angle should be one of: 90 (CCW), -90 (CW), 180.
        Falls back to 90 deg CCW if unsupported value provided.
        """
        text_size = cv2.getTextSize(text, self.FONT, font_scale, self.LINE_THICKNESS)[0]

        # Create text buffer (w x h) with transparent alpha
        text_buf = np.ones((text_size[1] + 10, text_size[0] + 10, 4), dtype=np.uint8) * 255
        text_buf[:, :, 3] = 0
        cv2.putText(text_buf, text, (5, text_size[1] + 2), self.FONT,
                   font_scale, color, self.LINE_THICKNESS, self.LINE_AA)

        # Rotate according to requested angle
        if rotate_angle == 90:
            text_rotated = cv2.rotate(text_buf, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate_angle == -90:
            text_rotated = cv2.rotate(text_buf, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == 180 or rotate_angle == -180:
            text_rotated = cv2.rotate(text_buf, cv2.ROTATE_180)
        else:
            # unsupported angle - default to 90 CCW
            text_rotated = cv2.rotate(text_buf, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if y >= 0 and x >= 0:
            h_rot, w_rot = text_rotated.shape[:2]
            y_end = min(y + h_rot, buf.shape[0])
            x_end = min(x + w_rot, buf.shape[1])
            h_crop = y_end - y
            w_crop = x_end - x

            # Alpha blend
            alpha = text_rotated[:h_crop, :w_crop, 3:4] / 255.0
            buf[y:y_end, x:x_end] = (
                buf[y:y_end, x:x_end] * (1 - alpha) +
                text_rotated[:h_crop, :w_crop] * alpha
            ).astype(np.uint8)

    def _draw_xy(self, positions: List[Tuple[float, float]]):
        """Draw XY trajectory plot centered at origin."""
        if not positions:
            self.xy_label.clear()
            return

        # Convert to numpy and make relative to start
        xs = np.array([p[0] for p in positions], dtype=float) - positions[0][0]
        ys = np.array([p[1] for p in positions], dtype=float) - positions[0][1]

        # Flip Y to match microscope orientation
        ys = -ys

        # Setup buffer
        w = max(200, self.xy_label.width())
        h = max(150, self.xy_label.height())
        buf = self._create_transparent_buffer(w, h)

        # Calculate plot area (square, centered)
        ml, mt, mr, mb = 80, 40, 40, 60
        plot_size = min(w - ml - mr, h - mt - mb)
        ox = ml + (w - ml - mr - plot_size) // 2
        oy = mt + (h - mt - mb - plot_size) // 2
        cx, cy = ox + plot_size // 2, oy + plot_size // 2

        # Scale data to fit (centered around 0) and add padding so data
        # doesn't touch the frame
        padding_factor = 1.15
        max_extent = max(np.nanmax(np.abs(xs)), np.nanmax(np.abs(ys)), 1e-6)
        total_span = max_extent * 2 * padding_factor
        total_span = max(total_span, 1.0)
        scale = plot_size / total_span

        def to_px(x, y):
            return (int(cx + x * scale), int(cy - y * scale))

        # Draw frame and axes
        cv2.rectangle(buf, (ox, oy), (ox + plot_size, oy + plot_size), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ox, cy), (ox + plot_size, cy), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (cx, oy), (cx, oy + plot_size), (0, 0, 0, 255), 1, self.LINE_AA)

        # Axis labels
        cv2.putText(buf, 'X (px)', (ox + plot_size // 2 - 30, oy + plot_size + 45), self.FONT, self.FONT_LABEL,
                    (0, 0, 0, 255), 1, self.LINE_AA)
        # Move Y label: match the middle-plot left-label distance (use same offset)
        self._draw_rotated_text(buf, 'Y (px)', max(0, ox - 56), cy - 20,
                                font_scale=self.FONT_LABEL, color=(0, 0, 0, 255), rotate_angle=90)

        # Tick marks and numbers
        for i in range(5):
            frac = i / 4.0
            val = (frac - 0.5) * total_span

            # X ticks (bottom and top)
            tick_x = ox + int(frac * plot_size)
            cv2.line(buf, (tick_x, oy + plot_size), (tick_x, oy + plot_size + 5), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (tick_x, oy), (tick_x, oy - 5), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.putText(buf, f'{val:.0f}', (tick_x - 12, oy + plot_size + 20), self.FONT, self.FONT_TICK,
                        (0, 0, 0, 255), 1, self.LINE_AA)

            # Y ticks (left and right)
            tick_y = int(oy + (1 - frac) * plot_size)
            cv2.line(buf, (ox - 5, tick_y), (ox, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (ox + plot_size, tick_y), (ox + plot_size + 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            # Left-side numbers: moved slightly further left for clarity
            cv2.putText(buf, f'{val:.0f}', (ox - 36, tick_y + 5), self.FONT, self.FONT_TICK, (0, 0, 0, 255), 1,
                        self.LINE_AA)

        # Draw trajectory (thin blue line)
        for i in range(len(xs) - 1):
            if not (np.isnan(xs[i]) or np.isnan(xs[i + 1]) or np.isnan(ys[i]) or np.isnan(ys[i + 1])):
                cv2.line(buf, to_px(xs[i], ys[i]), to_px(xs[i + 1], ys[i + 1]), (200, 0, 0, 255), 1, self.LINE_AA)

        self.xy_label.setPixmap(self._buffer_to_pixmap(buf))


    def _draw_time_xy(self, positions: List[Tuple[float, float]]):
        """Draw X and Y vs time (combined plot)."""
        if not positions:
            self.time_xy_label.clear()
            return

        # Convert to numpy and make relative
        xs = np.array([p[0] for p in positions], dtype=float) - positions[0][0]
        ys = np.array([p[1] for p in positions], dtype=float) - positions[0][1]

        # Get shared time axis if available; otherwise compute per-bead times from FPS
        fps = 1.0
        if self.video_widget and hasattr(self.video_widget, 'controller'):
            try:
                fps = float(self.video_widget.controller.get_fps())
            except Exception:
                pass

        if self.shared_times is not None and len(self.shared_times) >= len(xs):
            times = self.shared_times[:len(xs)]
        else:
            times = np.arange(len(xs)) / max(1.0, fps)

        # Setup buffer
        w = max(300, self.time_xy_label.width())
        h = max(150, self.time_xy_label.height())
        buf = self._create_transparent_buffer(w, h)

        # Plot area (increase left/right/bottom margins so x-axis is shorter
        # and there's room for rotated vertical labels)
        ml, mt, mr, mb = 140, 20, 140, 90
        pw, ph = w - ml - mr, h - mt - mb

        # Data ranges: include zero and make symmetric around 0 so 0 is always shown
        # add small padding around data so it doesn't sit on the axes
        pad = 1.08
        tmin, tmax = 0.0, times.max() if times.size else 1.0
        vmin_raw = min(xs.min(), ys.min())
        vmax_raw = max(xs.max(), ys.max())
        max_abs = max(abs(vmin_raw), abs(vmax_raw), 1e-6) * pad
        vmin = -max_abs
        vmax = max_abs
        v_range = vmax - vmin

        def tx(t):
            # t is a time value in seconds
            return int(ml + (t - tmin) / max(tmax - tmin, 1e-6) * pw)

        def py(v):
            return int(mt + ph - (v - vmin) / v_range * ph) if not np.isnan(v) else mt + ph // 2

        # Draw axes (bottom/left/top/right)
        cv2.line(buf, (ml, mt + ph), (ml + pw, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml, mt), (ml, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml, mt), (ml + pw, mt), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml + pw, mt), (ml + pw, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)

        # Labels: Time centered; left label for X (blue, rotated), right vertical label for Y (red)
        blue = (255, 0, 0, 255)
        red = (0, 0, 255, 255)
        # move time label further down so it doesn't collide with tick numbers
        cv2.putText(buf, 'Time (s)', (ml + pw // 2 - 25, mt + ph + 40),
                   self.FONT, self.FONT_LABEL, (0, 0, 0, 255), 1, self.LINE_AA)
        # Draw left rotated X label (match upper graph orientation)
        try:
            # move left rotated label slightly closer to the plot
            self._draw_rotated_text(buf, 'X (px)', max(0, ml - 56), mt + ph // 2 - 20, color=blue, rotate_angle=90)
        except Exception:
            cv2.putText(buf, 'X (px)', (5, mt + ph // 2), self.FONT, self.FONT_LABEL, blue, 1, self.LINE_AA)

        # Tick marks (bottom/top and left/right) with numbers on both sides
        for i in range(5):
            frac = i / 4.0
            # Time ticks (bottom and top)
            tick_x = ml + int(frac * pw)
            cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph + 5), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (tick_x, mt), (tick_x, mt - 5), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.putText(buf, f'{tmin + frac * (tmax - tmin):.1f}', (tick_x - 12, mt + ph + 22),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)

            # Value ticks (left and right)
            tick_y = mt + int((1 - frac) * ph)
            cv2.line(buf, (ml - 5, tick_y), (ml, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (ml + pw, tick_y), (ml + pw + 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            # left-side numbers: move a couple pixels further left
            cv2.putText(buf, f'{vmin + frac * v_range:.0f}', (ml - 28, tick_y + 4),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
            # right-side numbers: move many pixels to the left (closer to plot)
            cv2.putText(buf, f'{vmin + frac * v_range:.0f}', (ml + pw + 8, tick_y + 4),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)

        # Draw traces: X in blue, Y in red (thin lines)
        for i in range(len(xs) - 1):
            if not (np.isnan(xs[i]) or np.isnan(xs[i + 1])):
                cv2.line(buf, (tx(times[i]), py(xs[i])), (tx(times[i + 1]), py(xs[i + 1])),
                         blue, 1, self.LINE_AA)
            if not (np.isnan(ys[i]) or np.isnan(ys[i + 1])):
                cv2.line(buf, (tx(times[i]), py(ys[i])), (tx(times[i + 1]), py(ys[i + 1])),
                         red, 1, self.LINE_AA)

        # Add a vertical Y label on the right side of the middle plot
        try:
            # place rotated Y at right margin and color it red to match Y trace
            # move it further right so numbers don't collide
            # move right rotated label slightly left so it's closer to the plot but not overlapping numbers
            # bring it 16 px closer compared to previous value
            self._draw_rotated_text(buf, 'Y (px)', ml + pw + 40, mt + ph // 2 - 20, color=red, rotate_angle=90)
        except Exception:
            pass

        self.time_xy_label.setPixmap(self._buffer_to_pixmap(buf))

    def _draw_time_zv(self, z: List[float], voltage: List[float]):
        """Draw Z and Voltage vs time (combined plot)."""
        z = np.array(z, dtype=float) if z else np.array([])
        voltage = np.array(voltage, dtype=float) if voltage else np.array([])
        
        # Setup buffer
        w = max(300, self.time_zv_label.width())
        h = max(150, self.time_zv_label.height())
        buf = self._create_transparent_buffer(w, h)

        # Use same margins as the middle plot for visual parity
        ml, mt, mr, mb = 140, 20, 140, 90
        pw, ph = w - ml - mr, h - mt - mb

        # Draw axes (bottom/left/top/right)
        cv2.line(buf, (ml, mt + ph), (ml + pw, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml, mt), (ml, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml, mt), (ml + pw, mt), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml + pw, mt), (ml + pw, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)

        # Labels: Time centered; left label for Z (pixels, blue, rotated), right vertical label for Amplitude (red)
        blue = (255, 0, 0, 255)
        red = (0, 0, 255, 255)
        cv2.putText(buf, 'Time (s)', (ml + pw // 2 - 25, mt + ph + 40),
                   self.FONT, self.FONT_LABEL, (0, 0, 0, 255), 1, self.LINE_AA)

        # Left rotated Z label (match middle graph spacing)
        try:
            self._draw_rotated_text(buf, 'Z (px)', max(0, ml - 56), mt + ph // 2 - 20, color=blue, rotate_angle=90)
        except Exception:
            cv2.putText(buf, 'Z (px)', (5, mt + ph // 2), self.FONT, self.FONT_LABEL, blue, 1, self.LINE_AA)
        # Combined value range (include both series, padded). Also consider global timeline if present.
        has_z = z.size > 0 and not np.all(np.isnan(z))
        has_v = voltage.size > 0 and not np.all(np.isnan(voltage))
        has_timeline = self.function_generator_timeline is not None and len(self.function_generator_timeline) > 0

        if has_z or has_v or has_timeline:
            # determine number of samples to display
            n = max(len(z), len(voltage), len(self.function_generator_timeline) if self.function_generator_timeline is not None else 0)

            # Build time axis: prefer shared_times if available, else index/fps
            fps = 1.0
            if self.video_widget and hasattr(self.video_widget, 'controller'):
                try:
                    fps = float(self.video_widget.controller.get_fps())
                except Exception:
                    pass

            if self.shared_times is not None and len(self.shared_times) >= n:
                t_arr = self.shared_times[:n]
            else:
                t_arr = np.arange(n) / max(1.0, fps)

            tmin, tmax = float(t_arr[0]) if t_arr.size else 0.0, float(t_arr[-1]) if t_arr.size else 1.0

            # Left axis (Z and voltage) range: symmetric padded
            all_vals = []
            if has_z:
                all_vals.extend(z[~np.isnan(z)])
            if has_v:
                all_vals.extend(voltage[~np.isnan(voltage)])

            if all_vals:
                pad = 1.08
                vmin = min(all_vals)
                vmax = max(all_vals)
                max_abs = max(abs(vmin), abs(vmax), 1e-6) * pad
                vmin_sym, vmax_sym = -max_abs, max_abs
            else:
                vmin_sym, vmax_sym = -1.0, 1.0

            v_range = max(vmax_sym - vmin_sym, 1e-6)

            def tx_time(t):
                return int(ml + (t - tmin) / max(tmax - tmin, 1e-6) * pw)

            def py_left(v):
                return int(mt + ph - (v - vmin_sym) / v_range * ph) if not np.isnan(v) else mt + ph // 2

            # Timeline right-axis mapping if present
            if has_timeline:
                tl = np.array(self.function_generator_timeline, dtype=float)
                # align length
                if len(tl) >= len(t_arr):
                    tl_used = tl[:len(t_arr)]
                else:
                    tl_used = np.full(len(t_arr), np.nan)
                    tl_used[:len(tl)] = tl

                t_vmin = float(np.nanmin(tl_used)) if not np.all(np.isnan(tl_used)) else 0.0
                t_vmax = float(np.nanmax(tl_used)) if not np.all(np.isnan(tl_used)) else 1.0
                # pad timeline range slightly
                pad_t = 1.08
                t_vmin_p = t_vmin * pad_t if t_vmin <= 0 else t_vmin / pad_t
                t_vmax_p = t_vmax * pad_t if t_vmax >= 0 else t_vmax / pad_t
                t_vmin_final = min(t_vmin_p, t_vmax_p)
                t_vmax_final = max(t_vmin_p, t_vmax_p)

                def py_right(v):
                    return int(mt + ph - (v - t_vmin_final) / max((t_vmax_final - t_vmin_final), 1e-6) * ph)

            # Tick marks and numbers (left and right), match middle plot offsets
            for i in range(5):
                frac = i / 4.0
                tick_x = ml + int(frac * pw)
                cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph + 5), (0, 0, 0, 255), 1, self.LINE_AA)
                cv2.line(buf, (tick_x, mt), (tick_x, mt - 5), (0, 0, 0, 255), 1, self.LINE_AA)
                cv2.putText(buf, f'{tmin + frac * (tmax - tmin):.1f}', (tick_x - 12, mt + ph + 22),
                           self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)

                tick_y = mt + int((1 - frac) * ph)
                cv2.line(buf, (ml - 5, tick_y), (ml, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
                cv2.line(buf, (ml + pw, tick_y), (ml + pw + 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
                # left-side numbers
                cv2.putText(buf, f'{vmin_sym + frac * v_range:.1f}', (ml - 28, tick_y + 4),
                           self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
                # right-side: timeline scale if present, otherwise mirror left values
                if has_timeline:
                    tick_val = t_vmin_final + frac * (t_vmax_final - t_vmin_final)
                    cv2.putText(buf, f'{tick_val:.2f}', (ml + pw + 8, tick_y + 4),
                               self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
                else:
                    cv2.putText(buf, f'{vmin_sym + frac * v_range:.1f}', (ml + pw + 8, tick_y + 4),
                               self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)

            # Draw traces: Z in blue, Voltage in red (left axis)
            if has_z:
                for i in range(min(len(z), len(t_arr)) - 1):
                    if not (np.isnan(z[i]) or np.isnan(z[i + 1])):
                        cv2.line(buf, (tx_time(t_arr[i]), py_left(z[i])), (tx_time(t_arr[i + 1]), py_left(z[i + 1])), blue, 1, self.LINE_AA)

            if has_v:
                for i in range(min(len(voltage), len(t_arr)) - 1):
                    if not (np.isnan(voltage[i]) or np.isnan(voltage[i + 1])):
                        cv2.line(buf, (tx_time(t_arr[i]), py_left(voltage[i])), (tx_time(t_arr[i + 1]), py_left(voltage[i + 1])), red, 1, self.LINE_AA)

            # Draw timeline on right axis if present (green)
            if has_timeline:
                for i in range(len(t_arr) - 1):
                    v1 = tl_used[i]
                    v2 = tl_used[i + 1]
                    if not (np.isnan(v1) or np.isnan(v2)):
                        cv2.line(buf, (tx_time(t_arr[i]), py_right(v1)), (tx_time(t_arr[i + 1]), py_right(v2)), (0, 150, 0, 255), 1, self.LINE_AA)

            # Right rotated label for Amplitude
            try:
                self._draw_rotated_text(buf, 'Amplitude', ml + pw + 40, mt + ph // 2 - 20, color=red, rotate_angle=90)
            except Exception:
                pass
        else:
            # Empty plot - just show ticks (match middle plot style)
            for i in range(5):
                frac = i / 4
                tick_x = ml + int(frac * pw)
                cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph + 5), (0, 0, 0, 255), 1, self.LINE_AA)
                tick_y = mt + int((1 - frac) * ph)
                cv2.line(buf, (ml - 5, tick_y), (ml, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)

        self.time_zv_label.setPixmap(self._buffer_to_pixmap(buf))
