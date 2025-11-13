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
                          font_scale: float = FONT_LABEL):
        """Draw vertical text (rotated 90° CCW) on buffer."""
        text_size = cv2.getTextSize(text, self.FONT, font_scale, self.LINE_THICKNESS)[0]
        
        # Create text buffer
        text_buf = np.ones((text_size[1] + 10, text_size[0] + 10, 4), dtype=np.uint8) * 255
        text_buf[:, :, 3] = 0
        cv2.putText(text_buf, text, (5, text_size[1] + 2), self.FONT, 
                   font_scale, (0, 0, 0, 255), self.LINE_THICKNESS, self.LINE_AA)
        
        # Rotate and place
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

        # Scale data to fit (centered around 0)
        max_range = max(abs(xs.min()), abs(xs.max()), 
                       abs(ys.min()), abs(ys.max())) * 2
        max_range = max(max_range, 1.0)
        scale = plot_size / max_range

        def to_px(x, y):
            return (int(cx + x * scale), int(cy - y * scale))

        # Draw frame and axes
        cv2.rectangle(buf, (ox, oy), (ox + plot_size, oy + plot_size), 
                     (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ox, cy), (ox + plot_size, cy), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (cx, oy), (cx, oy + plot_size), (0, 0, 0, 255), 1, self.LINE_AA)
        
        # Axis labels
        cv2.putText(buf, 'X (px)', (ox + plot_size // 2 - 20, oy + plot_size + 50), 
                   self.FONT, self.FONT_LABEL, (0, 0, 0, 255), 1, self.LINE_AA)
        self._draw_rotated_text(buf, 'Y (px)', ox - 72, cy - 20)
        
        # Tick marks
        for i in range(5):
            frac = i / 4
            val = (frac - 0.5) * max_range
            
            # X ticks
            tick_x = int(ox + frac * plot_size)
            cv2.line(buf, (tick_x, oy + plot_size), (tick_x, oy + plot_size + 5), 
                    (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (tick_x, oy - 5), (tick_x, oy), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.putText(buf, f'{val:.0f}', (tick_x - 12, oy + plot_size + 20),
                       self.FONT, self.FONT_TICK, (0, 0, 0, 255), 1, self.LINE_AA)
            
            # Y ticks
            tick_y = int(oy + (1 - frac) * plot_size)
            cv2.line(buf, (ox - 5, tick_y), (ox, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (ox + plot_size, tick_y), (ox + plot_size + 5, tick_y), 
                    (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.putText(buf, f'{val:.0f}', (ox - 45, tick_y + 5),
                       self.FONT, self.FONT_TICK, (0, 0, 0, 255), 1, self.LINE_AA)

        # Draw trajectory
        for i in range(len(xs) - 1):
            cv2.line(buf, to_px(xs[i], ys[i]), to_px(xs[i + 1], ys[i + 1]), 
                    (50, 100, 200, 255), 1, self.LINE_AA)

        self.xy_label.setPixmap(self._buffer_to_pixmap(buf))


    def _draw_time_xy(self, positions: List[Tuple[float, float]]):
        """Draw X and Y vs time (combined plot)."""
        if not positions:
            self.time_xy_label.clear()
            return
        
        # Convert to numpy and make relative
        xs = np.array([p[0] for p in positions], dtype=float) - positions[0][0]
        ys = np.array([p[1] for p in positions], dtype=float) - positions[0][1]
        
        # Get FPS
        fps = 1.0
        if self.video_widget and hasattr(self.video_widget, 'controller'):
            try:
                fps = float(self.video_widget.controller.get_fps())
            except Exception:
                pass
        
        times = np.arange(len(xs)) / max(1.0, fps)
        
        # Setup buffer
        w = max(300, self.time_xy_label.width())
        h = max(150, self.time_xy_label.height())
        buf = self._create_transparent_buffer(w, h)
        
        # Plot area
        ml, mt, mr, mb = 50, 20, 30, 40
        pw, ph = w - ml - mr, h - mt - mb
        
        # Data ranges
        tmin, tmax = 0.0, times.max() if times.size else 1.0
        vmin = min(xs.min(), ys.min())
        vmax = max(xs.max(), ys.max())
        v_range = max(vmax - vmin, 1e-6)
        
        def tx(t):
            return int(ml + (t - tmin) / max(tmax - tmin, 1e-6) * pw)
        
        def py(v):
            return int(mt + ph - (v - vmin) / v_range * ph) if not np.isnan(v) else mt + ph // 2
        
        # Draw axes
        cv2.line(buf, (ml, mt + ph), (ml + pw, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml, mt), (ml, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        
        # Labels
        cv2.putText(buf, 'Time (s)', (ml + pw // 2 - 25, mt + ph + 30), 
                   self.FONT, self.FONT_LABEL, (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.putText(buf, 'X,Y (px)', (5, mt + ph // 2), 
                   self.FONT, self.FONT_LABEL, (0, 0, 0, 255), 1, self.LINE_AA)
        
        # Tick marks
        for i in range(5):
            frac = i / 4
            # Time ticks
            tick_x = ml + int(frac * pw)
            cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph + 5), 
                    (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.putText(buf, f'{tmin + frac * (tmax - tmin):.1f}', 
                       (tick_x - 12, mt + ph + 18),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
            # Value ticks
            tick_y = mt + int((1 - frac) * ph)
            cv2.line(buf, (ml - 5, tick_y), (ml, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.putText(buf, f'{vmin + frac * v_range:.0f}', (ml - 35, tick_y + 4),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
        
        # Draw traces
        for i in range(len(xs) - 1):
            if not (np.isnan(xs[i]) or np.isnan(xs[i + 1])):
                cv2.line(buf, (tx(times[i]), py(xs[i])), (tx(times[i + 1]), py(xs[i + 1])), 
                        (180, 100, 20, 255), 2, self.LINE_AA)
            if not (np.isnan(ys[i]) or np.isnan(ys[i + 1])):
                cv2.line(buf, (tx(times[i]), py(ys[i])), (tx(times[i + 1]), py(ys[i + 1])), 
                        (20, 100, 180, 255), 2, self.LINE_AA)

        self.time_xy_label.setPixmap(self._buffer_to_pixmap(buf))

    def _draw_time_zv(self, z: List[float], voltage: List[float]):
        """Draw Z and Voltage vs time (combined plot)."""
        z = np.array(z, dtype=float) if z else np.array([])
        voltage = np.array(voltage, dtype=float) if voltage else np.array([])
        
        # Setup buffer
        w = max(300, self.time_zv_label.width())
        h = max(150, self.time_zv_label.height())
        buf = self._create_transparent_buffer(w, h)
        
        # Plot area
        ml, mt, mr, mb = 50, 20, 30, 40
        pw, ph = w - ml - mr, h - mt - mb
        
        # Draw axes
        cv2.line(buf, (ml, mt + ph), (ml + pw, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.line(buf, (ml, mt), (ml, mt + ph), (0, 0, 0, 255), 1, self.LINE_AA)
        
        # Labels
        cv2.putText(buf, 'Time (s)', (ml + pw // 2 - 25, mt + ph + 30), 
                   self.FONT, self.FONT_LABEL, (0, 0, 0, 255), 1, self.LINE_AA)
        cv2.putText(buf, 'Z,V', (10, mt + ph // 2), 
                   self.FONT, self.FONT_LABEL, (0, 0, 0, 255), 1, self.LINE_AA)
        
        # Check for valid data
        has_z = z.size > 0 and not np.all(np.isnan(z))
        has_v = voltage.size > 0 and not np.all(np.isnan(voltage))
        
        if has_z or has_v:
            n = max(len(z), len(voltage))
            tmin, tmax = 0.0, float(n - 1) if n > 1 else 1.0
            
            # Combined value range
            all_vals = []
            if has_z:
                all_vals.extend(z[~np.isnan(z)])
            if has_v:
                all_vals.extend(voltage[~np.isnan(voltage)])
            
            vmin, vmax = min(all_vals), max(all_vals)
            v_range = max(vmax - vmin, 1e-6)
            
            def tx(i):
                return int(ml + i / max(tmax - tmin, 1e-6) * pw)
            
            def py(v):
                return int(mt + ph - (v - vmin) / v_range * ph) if not np.isnan(v) else mt + ph // 2
            
            # Tick marks
            for i in range(5):
                frac = i / 4
                tick_x = ml + int(frac * pw)
                cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph + 5), 
                        (0, 0, 0, 255), 1, self.LINE_AA)
                cv2.putText(buf, f'{tmin + frac * (tmax - tmin):.0f}', 
                           (tick_x - 12, mt + ph + 18),
                           self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
                
                tick_y = mt + int((1 - frac) * ph)
                cv2.line(buf, (ml - 5, tick_y), (ml, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
                cv2.putText(buf, f'{vmin + frac * v_range:.1f}', (ml - 35, tick_y + 4),
                           self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
            
            # Draw traces
            if has_z:
                for i in range(len(z) - 1):
                    if not (np.isnan(z[i]) or np.isnan(z[i + 1])):
                        cv2.line(buf, (tx(i), py(z[i])), (tx(i + 1), py(z[i + 1])), 
                                (0, 200, 0, 255), 2, self.LINE_AA)
            
            if has_v:
                for i in range(len(voltage) - 1):
                    if not (np.isnan(voltage[i]) or np.isnan(voltage[i + 1])):
                        cv2.line(buf, (tx(i), py(voltage[i])), (tx(i + 1), py(voltage[i + 1])), 
                                (200, 0, 0, 255), 2, self.LINE_AA)
        else:
            # Empty plot - just show ticks
            for i in range(5):
                frac = i / 4
                tick_x = ml + int(frac * pw)
                cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph + 5), 
                        (0, 0, 0, 255), 1, self.LINE_AA)
                tick_y = mt + int((1 - frac) * ph)
                cv2.line(buf, (ml - 5, tick_y), (ml, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)

        self.time_zv_label.setPixmap(self._buffer_to_pixmap(buf))
