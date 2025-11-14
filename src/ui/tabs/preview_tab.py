"""Preview tab - optimized implementation.

Provides a bead list and three plots:
- XY trajectory (centered at origin)
- X/Y vs time (combined)
- Z/Voltage vs time (combined)
"""
# type: ignore

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QGroupBox, QSplitter, QCheckBox, QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QPixmap, QImage
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2


class ResizablePixmapLabel(QLabel):
    """QLabel that emits a signal when resized."""
    resized = pyqtSignal()

    def resizeEvent(self, a0) -> None:  # type: ignore[override]
        """Override QWidget.resizeEvent - parameter named to match base signature for analyzer.

        This does not change runtime behaviour; it satisfies the static checker.
        """
        super().resizeEvent(a0)
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
        # If timeline is provided as event rows, store them here as (times, amps, enabled)
        self._timeline_events = None
        self._init_ui()

    def _setup_palette(self):
        """Configure widget palette to match app theme."""
        app = QApplication.instance()
        if isinstance(app, QApplication):
            pal = self.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))
            self.setPalette(pal)
            self.setAutoFillBackground(True)

    def _init_ui(self):
        """Initialize UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 8)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_bead_list_panel())
        splitter.addWidget(self._create_plots_panel())
        splitter.setSizes([260, 640])
        layout.addWidget(splitter)

    def _create_bead_list_panel(self) -> QGroupBox:
        """Create bead list panel with checkboxes."""
        gb = QGroupBox("Tracked Beads")
        layout = QVBoxLayout(gb)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        self.bead_list = QListWidget()
        self.bead_list.currentItemChanged.connect(self._on_bead_selected)
        # Set background to match the rest of the UI and remove border
        self.bead_list.setStyleSheet("QListWidget { background-color: #f0f0f0; border: none; }")
        layout.addWidget(self.bead_list)
        
        # Controls row at bottom
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 4, 0, 0)
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
        gb = QGroupBox("Movement Beads")
        layout = QVBoxLayout(gb)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create three plot labels
        self.xy_label = self._create_plot_label(150)
        self.time_xy_label = self._create_plot_label(150)
        self.time_zv_label = self._create_plot_label(150)
        
        # Add plots first
        layout.addWidget(self.xy_label, 1)
        layout.addSpacing(30)
        layout.addWidget(self.time_xy_label, 1)
        layout.addWidget(self.time_zv_label, 1)
        
        # Checkbox for stuck bead - positioned at bottom left
        self.stuck_bead_checkbox = QCheckBox("Stuck Bead")
        self.stuck_bead_checkbox.setEnabled(False)  # Disabled until a bead is selected
        self.stuck_bead_checkbox.stateChanged.connect(self._on_stuck_bead_changed)
        
        # Create horizontal layout for checkbox (left-aligned at bottom)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setContentsMargins(0, 4, 0, 0)
        checkbox_layout.addWidget(self.stuck_bead_checkbox)
        checkbox_layout.addStretch()  # Push checkbox to the left
        
        layout.addLayout(checkbox_layout)

        return gb
    
    def _create_plot_label(self, min_height: int) -> ResizablePixmapLabel:
        """Create a resizable plot label."""
        label = ResizablePixmapLabel()
        label.setMinimumHeight(min_height)
        label.setScaledContents(False)
        # allow layout to expand labels equally vertically
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.resized.connect(self._on_plot_resize)
        return label

    def set_video_widget(self, vw):
        """Set video widget reference for FPS access."""
        self.video_widget = vw

    def on_video_loaded(self):
        """Called when a new video is loaded."""
        # Clear any existing data
        self.bead_list.clear()
        self.tracked_beads = {}
        self.count_label.setText("0 beads loaded")
        self.select_all.setChecked(False)

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
            item.setData(Qt.ItemDataRole.UserRole, bead_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.bead_list.addItem(item)
        
        count = len(tracking_data)
        self.count_label.setText(f"{count} bead{'s' if count != 1 else ''} loaded")
        self.select_all.setChecked(True)

        # Try to read a global function generator timeline from the opened HDF5
        # Search robustly for common dataset names under /raw_data or /data.
        self.shared_times = None
        self.function_generator_timeline = None
        try:
            self._load_function_generator_timeline()
        except Exception:
            # Non-fatal: timeline optional
            self.shared_times = None
            self.function_generator_timeline = None

    def _load_function_generator_timeline(self):
        """Attempt to locate and load a function-generator timeline dataset from the HDF5 file.

        This is separated out so plotting can call it lazily if the file wasn't available
        at the time tracking data was loaded.
        """
        # reset
        self.shared_times = None
        self.function_generator_timeline = None
        if not (self.video_widget and hasattr(self.video_widget, 'get_hdf5_file')):
            return
        h5 = self.video_widget.get_hdf5_file()
        if h5 is None:
            return

        # prefer raw_data, fall back to data
        data_group = None
        if 'raw_data' in h5:
            data_group = h5['raw_data']
        elif 'data' in h5:
            data_group = h5['data']

        candidate = None
        if data_group is not None:
            if 'function_generator_timeline' in data_group:
                candidate = 'function_generator_timeline'
            else:
                for k in data_group.keys():
                    kl = k.lower()
                    if 'function' in kl and ('timeline' in kl or 'time' in kl or 'fg' in kl or 'func' in kl):
                        candidate = k
                        break

        if candidate is None:
            for k in h5.keys():
                kl = k.lower()
                if 'function' in kl and ('timeline' in kl or 'time' in kl or 'fg' in kl or 'func' in kl):
                    candidate = k
                    break

        if candidate is None:
            return

        # read dataset
        if data_group is not None and candidate in data_group:
            ds = data_group[candidate][:]
        else:
            ds = h5[candidate][:]

        arr = np.array(ds)
        print(f"[PreviewTab] found candidate timeline dataset '{candidate}' with shape {arr.shape} dtype={arr.dtype}")

        # If dataset is stored as text lines (CSV-like), parse it here
        if arr.dtype.kind in ('S', 'U', 'O') and arr.ndim == 1:
            try:
                lines = [x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
                lines = [ln.strip() for ln in lines if ln is not None and str(ln).strip() != '']
                if len(lines) >= 2 and (',' in lines[0]):
                    header = [h.strip() for h in lines[0].split(',')]
                    cols = {h: [] for h in header}
                    for ln in lines[1:]:
                        parts = [p.strip() for p in ln.split(',')]
                        if len(parts) < len(header):
                            continue
                        for i, h in enumerate(header):
                            cols[h].append(parts[i])

                    def _to_float_list(key):
                        if key in cols:
                            out = []
                            for v in cols[key]:
                                try:
                                    out.append(float(v))
                                except Exception:
                                    out.append(np.nan)
                            return np.array(out, dtype=float)
                        return None

                    def _to_int_list(key):
                        if key in cols:
                            out = []
                            for v in cols[key]:
                                try:
                                    out.append(int(float(v)))
                                except Exception:
                                    out.append(0)
                            return np.array(out, dtype=int)
                        return None

                    times_col = _to_float_list('timestamp') or _to_float_list('time') or _to_float_list('t')
                    amp_col = _to_float_list('amplitude_vpp') or _to_float_list('amplitude') or _to_float_list('vpp')
                    enabled_col = _to_int_list('output_enabled') or _to_int_list('enabled')

                    if times_col is not None:
                        print(f"[PreviewTab] parsed CSV-like timeline: {len(times_col)} rows, fields={header}")
                        parsed = np.empty(len(times_col), dtype=[('time', float), ('amp', float), ('enabled', int)])
                        parsed['time'] = times_col
                        parsed['amp'] = amp_col if amp_col is not None else np.zeros(len(times_col), dtype=float)
                        parsed['enabled'] = enabled_col if enabled_col is not None else np.zeros(len(times_col), dtype=int)
                        arr = parsed
            except Exception as e:
                print(f"[PreviewTab] failed parsing text timeline: {e}")

        # determine video/frame count and fps if available
        n_frames = None
        fps = 1.0
        try:
            if self.video_widget and hasattr(self.video_widget, 'controller'):
                n_frames = int(self.video_widget.controller.get_total_frames())
                fps = float(self.video_widget.controller.get_fps())
        except Exception:
            n_frames = None

        # If arr now has named fields
        if getattr(arr, 'dtype', None) is not None and getattr(arr.dtype, 'names', None):
            # Robust field selection for structured timeline datasets. Prefer explicit names
            field_names = list(arr.dtype.names)
            lname_map = {n.lower(): n for n in field_names}

            def pick_field(preds, default=None):
                for n in field_names:
                    nl = n.lower()
                    for p in preds:
                        if p in nl:
                            return n
                return default

            tname = pick_field(['timestamp', 'time', 't'], field_names[0])
            aname = pick_field(['amplitude', 'vpp', 'amp'])
            ename = pick_field(['output_enabled', 'output', 'enabled', 'enable', 'on'])

            # prefer exact common names if present
            if 'amplitude_vpp' in lname_map:
                aname = lname_map['amplitude_vpp']
            if 'output_enabled' in lname_map:
                ename = lname_map['output_enabled']

            print(f"[PreviewTab] structured fields detected: t={tname}, amp={aname}, enabled={ename}")

            times_col = np.array(arr[tname], dtype=float) if tname is not None else np.arange(len(arr), dtype=float)
            amp_col = None
            if aname is not None:
                try:
                    amp_col = np.array(arr[aname], dtype=float)
                except Exception:
                    amp_col = None
            enabled_col = None
            if ename is not None:
                try:
                    enabled_col = np.array(arr[ename], dtype=int)
                except Exception:
                    # boolean-like
                    enabled_col = np.array(arr[ename], dtype=bool).astype(int)

            # Fallbacks if amplitude or enabled not found
            if amp_col is None:
                # try common numeric fields that are not the timestamp
                for n in field_names:
                    if n == tname:
                        continue
                    try:
                        cand = np.array(arr[n], dtype=float)
                        # pick a field that doesn't equal the timestamps
                        if not np.allclose(cand, times_col):
                            amp_col = cand
                            aname = n
                            break
                    except Exception:
                        continue
            if enabled_col is None:
                # try to infer a boolean-like field
                for n in field_names:
                    if n == tname or n == aname:
                        continue
                    vals = arr[n]
                    if vals.dtype == np.bool_ or vals.dtype == bool:
                        enabled_col = np.array(vals, dtype=int)
                        ename = n
                        break

            if amp_col is None:
                amp_col = np.zeros_like(times_col)

            # Debug: show small sample of parsed fields
            try:
                print(f"[PreviewTab] parsed timeline fields sample: times={times_col[:6]}, amp={amp_col[:6]}, enabled={enabled_col[:6]}")
            except Exception:
                pass
            
            if enabled_col is None:
                enabled_col = np.zeros(len(times_col), dtype=int)

            # Now build per-frame timeline if we have frame info, else keep event times
            if n_frames is not None:
                t_arr = np.arange(n_frames) / max(1.0, fps)
                tl = np.full(len(t_arr), np.nan, dtype=float)
                cur_amp = 0.0
                cur_enabled = 0
                ev_idx = 0
                order = np.argsort(times_col)
                times_sorted = times_col[order]
                amp_sorted = amp_col[order]
                enabled_sorted = enabled_col[order]
                for i, t in enumerate(t_arr):
                    while ev_idx < len(times_sorted) and times_sorted[ev_idx] <= t:
                        cur_amp = amp_sorted[ev_idx] if amp_col is not None else cur_amp
                        cur_enabled = int(enabled_sorted[ev_idx]) if enabled_col is not None else cur_enabled
                        ev_idx += 1
                    tl[i] = cur_amp if cur_enabled else 0.0
                self.function_generator_timeline = tl
                self.shared_times = t_arr
                # also keep raw events so we can re-sample if needed
                self._timeline_events = (times_sorted, amp_sorted, enabled_sorted)
                print(f"[PreviewTab] structured timeline converted to per-frame step function; frames={len(tl)}; amp_unique={np.unique(amp_sorted)}")
            else:
                # no frame info: use event times as shared times
                self.shared_times = times_col
                # store events for later sampling into plot time axis
                order = np.argsort(times_col)
                times_sorted = times_col[order]
                amp_sorted = amp_col[order]
                enabled_sorted = enabled_col[order]
                self._timeline_events = (times_sorted, amp_sorted, enabled_sorted)
                # keep a compact representation too (amplitude only at event times)
                self.function_generator_timeline = (amp_sorted * enabled_sorted.astype(float))
                print(f"[PreviewTab] structured timeline loaded as events; events={len(times_col)}; amp_unique={np.unique(amp_sorted)}")
            return

        # If 2D with two columns -> (time, value)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            # Nx2: assume first column is time (s), second is amplitude (V)
            self.shared_times = arr[:, 0].astype(float)
            self.function_generator_timeline = arr[:, 1].astype(float)
            print(f"[PreviewTab] using timeline as (time, value) pairs; times len={len(self.shared_times)}")
            return

        # 1D numeric arrays
        if arr.ndim == 1:
            if n_frames is not None and len(arr) == n_frames:
                self.function_generator_timeline = arr.astype(float)
                self.shared_times = np.arange(len(arr)) / max(1.0, fps)
                print(f"[PreviewTab] using timeline as one value per frame; frames={n_frames}, fps={fps}")
                return
            if n_frames is not None and len(arr) > n_frames:
                chunk_size = int(np.ceil(len(arr) / n_frames))
                values = []
                for i in range(n_frames):
                    start = i * chunk_size
                    end = min(len(arr), start + chunk_size)
                    segment = arr[start:end]
                    if segment.size > 0:
                        vpp = float(np.nanmax(segment) - np.nanmin(segment))
                        values.append(vpp)
                    else:
                        values.append(np.nan)
                self.function_generator_timeline = np.array(values, dtype=float)
                self.shared_times = np.arange(len(values)) / max(1.0, fps)
                print(f"[PreviewTab] high-rate waveform -> computed Vpp per frame; out_len={len(values)}, fps={fps}")
                return

        # fallback: couldn't interpret dataset
        self.shared_times = None
        self.function_generator_timeline = None

    def _on_bead_selected(self, current, previous=None):
        """Handle bead selection change."""
        if current:
            bead_id = current.data(Qt.ItemDataRole.UserRole)
            self._update_plots_for_bead(bead_id)
            
            # Update stuck bead checkbox based on bead's stuck state
            self.stuck_bead_checkbox.setEnabled(True)
            bead = self.tracked_beads.get(bead_id)
            if bead:
                # Block signals to avoid triggering the handler
                self.stuck_bead_checkbox.blockSignals(True)
                self.stuck_bead_checkbox.setChecked(bead.get('stuck', False))
                self.stuck_bead_checkbox.blockSignals(False)
        else:
            # No bead selected, disable checkbox
            self.stuck_bead_checkbox.setEnabled(False)
            self.stuck_bead_checkbox.setChecked(False)

    def _on_stuck_bead_changed(self, state):
        """Handle stuck bead checkbox state change."""
        current = self.bead_list.currentItem()
        if not current:
            return
        bead_id = current.data(Qt.ItemDataRole.UserRole)
        bead = self.tracked_beads.get(bead_id)
        if not bead:
            return

        # Update stuck state
        is_stuck = (state == Qt.CheckState.Checked)
        bead['stuck'] = is_stuck

        # Update the bead list item text to show/hide lock icon
        base_name = f"Bead {bead_id + 1}"
        if is_stuck:
            current.setText(f"{base_name} 🔒")
        else:
            current.setText(base_name)

    def _toggle_select_all(self, state):
        """Toggle all bead checkboxes."""
        checked = (state == Qt.CheckState.Checked)
        for i in range(self.bead_list.count()):
            self.bead_list.item(i).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def _on_plot_resize(self):
        """Redraw plots on resize."""
        # Let the layout distribute the available vertical space equally.
        # Just trigger redraws for the current selection.
        try:
            current = self.bead_list.currentItem()
            if current:
                self._update_plots_for_bead(current.data(Qt.ItemDataRole.UserRole))
            else:
                # Redraw empty placeholders if nothing selected
                self.xy_label.clear()
                self.time_xy_label.clear()
                self.time_zv_label.clear()
        except Exception:
            pass

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
    
    @staticmethod
    def _get_nice_tick_values(vmin: float, vmax: float, num_ticks: int = 5) -> list:
        """Calculate nice round tick values for axis labels.
        
        Returns list of (fraction, value, label_text) tuples where:
        - fraction is the position along the axis (0 to 1)
        - value is the actual data value
        - label_text is the formatted string to display
        """
        data_range = vmax - vmin
        if data_range < 1e-6:
            # Degenerate range, just use equal spacing
            return [(i/4.0, vmin + i/4.0 * data_range, f'{vmin + i/4.0 * data_range:.1f}') 
                    for i in range(num_ticks)]
        
        # Calculate nice step size
        rough_step = data_range / (num_ticks - 1)
        magnitude = 10 ** np.floor(np.log10(rough_step))
        
        # Normalize and find nice step (1, 2, 2.5, 5, or 10 of the magnitude)
        normalized = rough_step / magnitude
        if normalized <= 1.0:
            nice_step = 1.0 * magnitude
        elif normalized <= 2.0:
            nice_step = 2.0 * magnitude
        elif normalized <= 2.5:
            nice_step = 2.5 * magnitude
        elif normalized <= 5.0:
            nice_step = 5.0 * magnitude
        else:
            nice_step = 10.0 * magnitude
        
        # Find nice start value (round down to nearest nice_step)
        nice_min = np.floor(vmin / nice_step) * nice_step
        
        # Generate tick values
        ticks = []
        tick_val = nice_min
        while tick_val <= vmax + nice_step * 0.01:  # Small tolerance
            if tick_val >= vmin - nice_step * 0.01:  # Within range
                # Calculate fraction along axis
                frac = (tick_val - vmin) / data_range
                
                # Format label: use .0f for integers, .1f for half-steps, .2f otherwise
                if abs(tick_val - round(tick_val)) < 1e-6:
                    label = f'{int(round(tick_val))}'
                elif abs(tick_val * 2 - round(tick_val * 2)) < 1e-6:
                    label = f'{tick_val:.1f}'
                else:
                    label = f'{tick_val:.2f}'
                
                ticks.append((frac, tick_val, label))
            tick_val += nice_step
        
        return ticks
    
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

        # Scale data to fit (centered around 0) and add padding so data doesn't touch the frame
        # Store computed range for use in time-series plot to ensure matching axes
        padding_factor = 1.15
        max_extent = max(np.nanmax(np.abs(xs)), np.nanmax(np.abs(ys)), 1e-6)
        max_extent_padded = max_extent * padding_factor
        # Store for shared use with time plot
        self._xy_vmin = -max_extent_padded
        self._xy_vmax = max_extent_padded
        total_span = max_extent_padded * 2
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
        self._draw_rotated_text(buf, 'Y (px)', max(0, ox - 66), cy - 20,
                                font_scale=self.FONT_LABEL, color=(0, 0, 0, 255), rotate_angle=90)

        # Tick marks and numbers
        for i in range(5):
            frac = i / 4.0
            val = (frac - 0.5) * total_span

            # X ticks (bottom and top)
            tick_x = ox + int(frac * plot_size)
            # draw ticks inward (into the plot) instead of outward
            cv2.line(buf, (tick_x, oy + plot_size), (tick_x, oy + plot_size - 5), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (tick_x, oy), (tick_x, oy + 5), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.putText(buf, f'{val:.0f}', (tick_x - 12, oy + plot_size + 20), self.FONT, self.FONT_TICK,
                        (0, 0, 0, 255), 1, self.LINE_AA)

            # Y ticks (left and right)
            tick_y = int(oy + (1 - frac) * plot_size)
            # draw ticks inward (into the plot)
            cv2.line(buf, (ox, tick_y), (ox + 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (ox + plot_size, tick_y), (ox + plot_size - 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
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

        # Data ranges: use shared limits from XY plot if available, otherwise compute
        # If XY plot was drawn first, reuse its limits for consistency
        if hasattr(self, '_xy_vmin') and hasattr(self, '_xy_vmax'):
            vmin = self._xy_vmin
            vmax = self._xy_vmax
        else:
            # Compute independently with padding
            pad = 1.15
            vmin_raw = min(xs.min(), ys.min())
            vmax_raw = max(xs.max(), ys.max())
            max_abs = max(abs(vmin_raw), abs(vmax_raw), 1e-6) * pad
            vmin = -max_abs
            vmax = max_abs
        
        tmin, tmax = 0.0, times.max() if times.size else 1.0
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

        # Calculate nice tick values for time axis (more ticks for better resolution)
        time_ticks = self._get_nice_tick_values(tmin, tmax, 11)
        
        # Draw time ticks (bottom and top) - major ticks with nice labels
        for frac, tick_val, label in time_ticks:
            tick_x = ml + int(frac * pw)
            # draw major ticks inward into the plot area (longer)
            cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph - 5), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (tick_x, mt), (tick_x, mt + 5), (0, 0, 0, 255), 1, self.LINE_AA)
            # Center the label based on its length
            label_offset = len(label) * 3
            cv2.putText(buf, label, (tick_x - label_offset, mt + ph + 22),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)

        # Calculate nice tick values for value axis (y-axis)
        value_ticks = self._get_nice_tick_values(vmin, vmax, 9)
        
        # Draw value ticks (left and right)
        for frac, tick_val, label in value_ticks:
            tick_y = mt + int((1 - frac) * ph)
            # draw ticks inward into the plot area
            cv2.line(buf, (ml, tick_y), (ml + 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            cv2.line(buf, (ml + pw, tick_y), (ml + pw - 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
            # left-side numbers: move a couple pixels further left
            cv2.putText(buf, label, (ml - 28, tick_y + 4),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
            # right-side numbers: move many pixels to the left (closer to plot)
            cv2.putText(buf, label, (ml + pw + 8, tick_y + 4),
                       self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
        
        # Minor ticks on time axis (x-axis) - add subdivisions between major ticks
        if len(time_ticks) > 1:
            for i in range(len(time_ticks) - 1):
                frac1 = time_ticks[i][0]
                frac2 = time_ticks[i + 1][0]
                # Add 4 minor ticks between each pair of major ticks
                for j in range(1, 5):
                    minor_frac = frac1 + (frac2 - frac1) * j / 5.0
                    tick_x = ml + int(minor_frac * pw)
                    # draw shorter minor ticks (3 pixels instead of 5)
                    cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph - 3), (0, 0, 0, 255), 1, self.LINE_AA)
                    cv2.line(buf, (tick_x, mt), (tick_x, mt + 3), (0, 0, 0, 255), 1, self.LINE_AA)

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
        # Ensure timeline loaded (lazy load if video/HDF5 became available after tracking load)
        if self.function_generator_timeline is None:
            try:
                self._load_function_generator_timeline()
            except Exception:
                pass

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
                pad = 1.15
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
                # Prefer reconstructing from raw events
                events = getattr(self, '_timeline_events', None)
                tl_used = None
                if events:
                    # Re-sample raw events into the plotting time axis
                    times_sorted, amp_sorted, enabled_sorted = events
                    print(f"[PreviewTab] resampling events into plot t_arr: times={times_sorted}, amp={amp_sorted}, enabled={enabled_sorted}")
                    tl_ev = np.full(len(t_arr), 0.0, dtype=float)
                    ev_idx = 0
                    cur_amp = 0.0
                    cur_enabled = 0
                    for i, t in enumerate(t_arr):
                        while ev_idx < len(times_sorted) and times_sorted[ev_idx] <= t:
                            cur_amp = float(amp_sorted[ev_idx])
                            cur_enabled = int(enabled_sorted[ev_idx])
                            ev_idx += 1
                        tl_ev[i] = cur_amp if cur_enabled else 0.0
                    tl_used = tl_ev
                else:
                    # Fallback to per-frame timeline array
                    if self.function_generator_timeline is not None:
                        tl = np.array(self.function_generator_timeline, dtype=float)
                        if len(tl) >= len(t_arr):
                            tl_used = tl[:len(t_arr)]
                        else:
                            tl_used = np.full(len(t_arr), 0.0)
                            tl_used[:len(tl)] = tl

                # Compute visible range for right axis from event amplitudes with padding
                events = getattr(self, '_timeline_events', None)
                if events is not None:
                    _, amp_sorted, _ = events
                    amp_max = float(np.nanmax(amp_sorted)) if len(amp_sorted) else 4.0
                    t_vmin_final = 0.0
                    # Add 15% padding above the max amplitude
                    t_vmax_final = float(max(1.0, np.ceil(amp_max * 1.15)))
                    print(f"[PreviewTab] timeline amp_max={amp_max} -> axis 0..{t_vmax_final}")
                elif tl_used is not None:
                    t_vmin = float(np.nanmin(tl_used)) if not np.all(np.isnan(tl_used)) else 0.0
                    t_vmax = float(np.nanmax(tl_used)) if not np.all(np.isnan(tl_used)) else 1.0
                    t_vmin_final = 0.0
                    # Add 15% padding above the max value
                    t_vmax_final = float(max(1.0, np.ceil(t_vmax * 1.15)))
                else:
                    t_vmin_final = 0.0
                    t_vmax_final = 1.0

                def py_right(v):
                    return int(mt + ph - (v - t_vmin_final) / max((t_vmax_final - t_vmin_final), 1e-6) * ph)

            # Calculate nice tick values for time and value axes (more ticks for better resolution)
            time_ticks = self._get_nice_tick_values(tmin, tmax, 11)
            value_ticks = self._get_nice_tick_values(vmin_sym, vmax_sym, 9)
            
            # Draw time ticks (bottom and top) - major ticks with nice labels
            for frac, tick_val, label in time_ticks:
                tick_x = ml + int(frac * pw)
                # draw major time ticks inward (longer)
                cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph - 5), (0, 0, 0, 255), 1, self.LINE_AA)
                cv2.line(buf, (tick_x, mt), (tick_x, mt + 5), (0, 0, 0, 255), 1, self.LINE_AA)
                # Center the label based on its length
                label_offset = len(label) * 3
                cv2.putText(buf, label, (tick_x - label_offset, mt + ph + 22),
                           self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)

            # Draw value ticks (left and right)
            for frac, tick_val, label in value_ticks:
                tick_y = mt + int((1 - frac) * ph)
                # draw value ticks inward
                cv2.line(buf, (ml, tick_y), (ml + 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
                cv2.line(buf, (ml + pw, tick_y), (ml + pw - 5, tick_y), (0, 0, 0, 255), 1, self.LINE_AA)
                # left-side numbers
                cv2.putText(buf, label, (ml - 28, tick_y + 4),
                           self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
                # right-side: timeline scale if present, otherwise mirror left values
                if has_timeline:
                    # Calculate timeline value at this fraction
                    timeline_val = t_vmin_final + frac * (t_vmax_final - t_vmin_final)
                    # prefer integer-looking labels for amplitude (e.g. 0, 4)
                    if abs(timeline_val - round(timeline_val)) < 1e-6:
                        timeline_lbl = str(int(round(timeline_val)))
                    else:
                        timeline_lbl = f'{timeline_val:.2f}'
                    cv2.putText(buf, timeline_lbl, (ml + pw + 8, tick_y + 4),
                               self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
                else:
                    # No timeline: show only a single '0' label at the bottom tick on the right axis
                    if frac < 0.01:  # First tick
                        cv2.putText(buf, '0', (ml + pw + 8, tick_y + 4),
                                   self.FONT, self.FONT_SMALL, (0, 0, 0, 255), 1, self.LINE_AA)
                    # otherwise leave right-side blank
            
            # Minor ticks on time axis (x-axis) - add subdivisions between major ticks
            if len(time_ticks) > 1:
                for i in range(len(time_ticks) - 1):
                    frac1 = time_ticks[i][0]
                    frac2 = time_ticks[i + 1][0]
                    # Add 4 minor ticks between each pair of major ticks
                    for j in range(1, 5):
                        minor_frac = frac1 + (frac2 - frac1) * j / 5.0
                        tick_x = ml + int(minor_frac * pw)
                        # draw shorter minor ticks (3 pixels instead of 5)
                        cv2.line(buf, (tick_x, mt + ph), (tick_x, mt + ph - 3), (0, 0, 0, 255), 1, self.LINE_AA)
                        cv2.line(buf, (tick_x, mt), (tick_x, mt + 3), (0, 0, 0, 255), 1, self.LINE_AA)

            # Draw traces: Z in blue, Voltage in red (left axis)
            if has_z:
                for i in range(min(len(z), len(t_arr)) - 1):
                    if not (np.isnan(z[i]) or np.isnan(z[i + 1])):
                        cv2.line(buf, (tx_time(t_arr[i]), py_left(z[i])), (tx_time(t_arr[i + 1]), py_left(z[i + 1])), blue, 1, self.LINE_AA)

            if has_v:
                for i in range(min(len(voltage), len(t_arr)) - 1):
                    if not (np.isnan(voltage[i]) or np.isnan(voltage[i + 1])):
                        cv2.line(buf, (tx_time(t_arr[i]), py_left(voltage[i])), (tx_time(t_arr[i + 1]), py_left(voltage[i + 1])), red, 1, self.LINE_AA)

            # Draw timeline on right axis if present
            if has_timeline and tl_used is not None:
                # Diagnostics
                try:
                    valid = tl_used[tl_used > 0]
                    print(f"[PreviewTab] plotting timeline: min={float(np.min(tl_used))}, max={float(np.max(tl_used))}, unique={np.unique(tl_used)}")
                except Exception as e:
                    print(f"[PreviewTab] timeline diagnostics failed: {e}")

                for i in range(len(t_arr) - 1):
                    v1 = float(tl_used[i]) if i < len(tl_used) else 0.0
                    v2 = float(tl_used[i + 1]) if (i + 1) < len(tl_used) else 0.0
                    cv2.line(buf, (tx_time(t_arr[i]), py_right(v1)), (tx_time(t_arr[i + 1]), py_right(v2)), (0, 0, 255, 255), 1, self.LINE_AA)

            # Right rotated label for Amplitude: center vertically based on text size
            try:
                ts = cv2.getTextSize('Amplitude (Vpp)', self.FONT, self.FONT_LABEL, self.LINE_THICKNESS)[0]
                # when rotated 90deg, the rotated height equals the text width + padding (text_buf width = text_size[0] + 10)
                rotated_h = ts[0] + 10
                y_pos = int(mt + ph // 2 - rotated_h // 2)
                self._draw_rotated_text(buf, 'Amplitude (Vpp)', ml + pw + 40, y_pos, color=red, rotate_angle=90)
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
