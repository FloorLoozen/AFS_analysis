"""Info tab for displaying measurement metadata and GPU information."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QScrollArea, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from datetime import datetime
import os
from src.utils.gpu_config import get_gpu_info


class InfoTab(QWidget):
    """Tab for displaying measurement information and metadata."""

    def __init__(self):
        """Initialize info tab."""
        super().__init__()
        # Ensure this tab uses the application's Window palette color so it matches other tabs
        app = QApplication.instance()
        if app is not None:
            pal = self.palette()
            pal.setColor(QPalette.Window, app.palette().color(QPalette.Window))  # type: ignore
            self.setPalette(pal)
            self.setAutoFillBackground(True)
        self.video_widget = None
        # Track maximum label width across info sections so we can align columns
        self._max_info_label_width = 0
        self._init_ui()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget
    
    def _init_ui(self):
        """Initialize the user interface with scrollable metadata."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scroll area without frame
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)  # Remove black border
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container widget for scroll area
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setContentsMargins(8, 24, 8, 8)
        layout.setSpacing(10)
        
        # GPU/System Info Section (always visible)
        system_group = QGroupBox("System Info")
        self.system_layout = QFormLayout(system_group)
        self.system_layout.setContentsMargins(8, 8, 8, 8)
        self.system_layout.setSpacing(5)
        self.system_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.system_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # type: ignore
        layout.addWidget(system_group)
        
        # Populate GPU info immediately
        self._populate_system_info()
        
        # Measurement Info Section
        measurement_group = QGroupBox("Measurement Info")
        self.measurement_layout = QFormLayout(measurement_group)
        self.measurement_layout.setContentsMargins(8, 8, 8, 8)
        self.measurement_layout.setSpacing(5)
        self.measurement_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.measurement_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # type: ignore
        layout.addWidget(measurement_group)
        
        # Video Info Section
        video_group = QGroupBox("Video Info")
        self.video_layout = QFormLayout(video_group)
        self.video_layout.setContentsMargins(8, 8, 8, 8)
        self.video_layout.setSpacing(5)
        self.video_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.video_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # type: ignore
        layout.addWidget(video_group)
        
        # File Info Section
        file_group = QGroupBox("File Info")
        self.file_layout = QFormLayout(file_group)
        self.file_layout.setContentsMargins(8, 8, 8, 8)
        self.file_layout.setSpacing(5)
        self.file_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.file_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # type: ignore
        layout.addWidget(file_group)
        
        # Additional Metadata Section (for all other HDF5 attributes)
        metadata_group = QGroupBox("Additional Metadata")
        self.metadata_layout = QFormLayout(metadata_group)
        self.metadata_layout.setContentsMargins(8, 8, 8, 8)
        self.metadata_layout.setSpacing(5)
        self.metadata_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.metadata_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # type: ignore
        layout.addWidget(metadata_group)
        
        layout.addStretch()
        
        # Set scroll widget and add to main layout
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def _add_info_row(self, layout, label_text, value_text=""):
        """Add a label-value row to a form layout."""
        # Label with fixed width for perfect alignment
        label = QLabel(label_text)
        label.setStyleSheet("color: #666;")
        # Left-align label text; we'll compute a consistent column width after rows are added
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)  # type: ignore
        # Update maximum label width seen so far (use font metrics)
        fm = label.fontMetrics()
        w = fm.boundingRect(label_text).width() + 8
        if w > self._max_info_label_width:
            self._max_info_label_width = w
        
        # Value
        value = QLabel(value_text)
        value.setWordWrap(True)
        value.setStyleSheet("color: #222;")
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)  # type: ignore
        
        # Add to form layout
        layout.addRow(label, value)
        
        return value

    def _apply_label_widths(self):
        """Apply the computed maximum label width to all label widgets in the info layouts.

        This ensures all label columns align to the same left edge based on the longest label.
        """
        if self._max_info_label_width <= 0:
            return

        target_w = self._max_info_label_width
        for layout in (self.system_layout, self.measurement_layout, self.video_layout, self.file_layout, self.metadata_layout):
            # Iterate over items and set label widths for those ending with ':' which are the labels
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is None:
                    continue
                w = item.widget()
                if isinstance(w, QLabel) and w.text().strip().endswith(':'):
                    w.setFixedWidth(target_w)
                    w.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    
    def _clear_layout(self, layout):
        """Clear all rows from a form layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _populate_system_info(self):
        """Populate system and GPU information (called once during init)."""
        self._clear_layout(self.system_layout)
        
        # Get GPU information
        gpu_info = get_gpu_info()
        
        # GPU Acceleration Status - simple text without special styling
        if gpu_info['opencl_enabled']:
            status_text = "Enabled"
        elif gpu_info['opencl_available']:
            status_text = "Available but not enabled"
        else:
            status_text = "Not Available (CPU only)"
        
        self._add_info_row(self.system_layout, "GPU Acceleration:", status_text)
        
        # GPU Device Name
        if gpu_info['device_name'] != 'CPU':
            self._add_info_row(self.system_layout, "GPU Device:", gpu_info['device_name'])
        
        # Acceleration Type
        if gpu_info['acceleration'] != 'None':
            self._add_info_row(self.system_layout, "Acceleration Type:", gpu_info['acceleration'])
    
    def on_video_loaded(self):
        """Called when a new video is loaded."""
        self.update_info()
    
    def update_info(self):
        """Update measurement information display with detailed sections."""
        if not self.video_widget:
            self._clear_layout(self.measurement_layout)
            self._clear_layout(self.video_layout)
            self._clear_layout(self.file_layout)
            self._clear_layout(self.metadata_layout)
            return
        
        metadata = self.video_widget.get_metadata()
        
        if not metadata:
            self._clear_layout(self.measurement_layout)
            self._clear_layout(self.video_layout)
            self._clear_layout(self.file_layout)
            self._clear_layout(self.metadata_layout)
            return
        
        # Clear existing content
        self._clear_layout(self.measurement_layout)
        self._clear_layout(self.video_layout)
        self._clear_layout(self.file_layout)
        self._clear_layout(self.metadata_layout)
        
        # Track which keys we've already displayed
        displayed_keys = set()
        
        # === MEASUREMENT INFO ===
        # Sample name
        if 'user_sample_name' in metadata and metadata['user_sample_name']:
            self._add_info_row(self.measurement_layout, "Sample:", str(metadata['user_sample_name']))
            displayed_keys.add('user_sample_name')
        
        # Operator
        if 'user_operator' in metadata and metadata['user_operator']:
            self._add_info_row(self.measurement_layout, "Operator:", str(metadata['user_operator']))
            displayed_keys.add('user_operator')
        
        # System name
        if 'user_system_name' in metadata and metadata['user_system_name']:
            self._add_info_row(self.measurement_layout, "System:", str(metadata['user_system_name']))
            displayed_keys.add('user_system_name')
        
        # Timestamp
        timestamp_field = 'created_at' if 'created_at' in metadata else 'timestamp'
        if timestamp_field in metadata:
            try:
                timestamp = metadata[timestamp_field]
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, str):
                    if 'T' in timestamp:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        time_str = timestamp
                else:
                    time_str = str(timestamp)
                self._add_info_row(self.measurement_layout, "Recorded:", time_str)
                displayed_keys.add(timestamp_field)
            except Exception:
                pass
        
        # Description
        if 'description' in metadata and metadata['description']:
            self._add_info_row(self.measurement_layout, "Description:", str(metadata['description']))
            displayed_keys.add('description')
        
        # === VIDEO INFO ===
        # Resolution
        if 'frame_shape' in metadata:
            shape = metadata['frame_shape']
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                self._add_info_row(self.video_layout, "Resolution:", f"{shape[1]} × {shape[0]} px")
            displayed_keys.add('frame_shape')
        
        # Frame count
        if 'total_frames' in metadata:
            self._add_info_row(self.video_layout, "Total Frames:", f"{metadata['total_frames']:,}")
            displayed_keys.add('total_frames')
        
        # FPS - show actual if available, otherwise target
        fps = None
        if 'actual_fps' in metadata:
            fps = metadata['actual_fps']
            self._add_info_row(self.video_layout, "Frame Rate (fps):", f"{fps:.2f}")
            displayed_keys.add('actual_fps')
            displayed_keys.add('fps')  # Don't show target if we have actual
        elif 'fps' in metadata:
            fps = metadata['fps']
            self._add_info_row(self.video_layout, "Frame Rate (fps):", f"{fps:.2f}")
            displayed_keys.add('fps')
        
        # Duration
        if fps and fps > 0 and 'total_frames' in metadata:
            duration_sec = metadata['total_frames'] / fps
            hours = int(duration_sec // 3600)
            minutes = int((duration_sec % 3600) // 60)
            seconds = duration_sec % 60
            if hours > 0:
                duration_str = f"{hours}:{minutes:02d}:{seconds:05.2f}"
            else:
                duration_str = f"{minutes}:{seconds:05.2f}"
            self._add_info_row(self.video_layout, "Duration (h:mm:ss):", duration_str)
        
        # Data type
        if 'dtype' in metadata:
            self._add_info_row(self.video_layout, "Data Type:", str(metadata['dtype']))
            displayed_keys.add('dtype')
        
        # Color format
        if 'color_format' in metadata:
            self._add_info_row(self.video_layout, "Color Format:", str(metadata['color_format']))
            displayed_keys.add('color_format')
        
        # Color channels
        if 'frame_shape' in metadata:
            shape = metadata['frame_shape']
            if isinstance(shape, (list, tuple)):
                if len(shape) == 3:
                    channels = shape[2]
                    self._add_info_row(self.video_layout, "Channels:", f"{channels} (Color)")
                else:
                    self._add_info_row(self.video_layout, "Channels:", "1 (Grayscale)")
        
        # Compression info
        if 'compression' in metadata and metadata['compression']:
            self._add_info_row(self.video_layout, "Compression:", str(metadata['compression']))
            displayed_keys.add('compression')
            
            if 'compression_opts' in metadata and metadata['compression_opts']:
                self._add_info_row(self.video_layout, "Compression Level:", str(metadata['compression_opts']))
                displayed_keys.add('compression_opts')
        
        # === FILE INFO ===
        # File path
        if 'file_path' in metadata:
            file_path = metadata['file_path']
            self._add_info_row(self.file_layout, "Path:", file_path)
            displayed_keys.add('file_path')
            
            # File size
            try:
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                if size_mb > 1024:
                    size_gb = size_mb / 1024
                    self._add_info_row(self.file_layout, "File Size (GB):", f"{size_gb:.2f}")
                else:
                    self._add_info_row(self.file_layout, "File Size (MB):", f"{size_mb:.2f}")
                
                # Calculate uncompressed size estimate
                if 'total_frames' in metadata and 'frame_shape' in metadata and 'dtype' in metadata:
                    shape = metadata['frame_shape']
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        # Get bytes per pixel from dtype
                        import numpy as np
                        dtype_obj = np.dtype(metadata['dtype'])
                        bytes_per_pixel = dtype_obj.itemsize
                        
                        # Calculate total pixels
                        total_pixels = metadata['total_frames'] * shape[0] * shape[1]
                        if len(shape) == 3:
                            total_pixels *= shape[2]
                        
                        uncompressed_size = total_pixels * bytes_per_pixel
                        uncompressed_mb = uncompressed_size / (1024 * 1024)
                        
                        # Compression ratio
                        compression_ratio = uncompressed_size / file_size
                        self._add_info_row(self.file_layout, "Compression Ratio:", f"{compression_ratio:.1f}×")
            except Exception:
                pass
        
        # Format version
        if 'format_version' in metadata:
            self._add_info_row(self.file_layout, "Format Version:", str(metadata['format_version']))
            displayed_keys.add('format_version')
        
        # === ADDITIONAL METADATA ===
        # Skip only truly redundant/duplicate data
        skip_keys = {
            # Duplicates of data already shown above
            'actual_fps',  # Already shown as Frame Rate
            'fps',  # Already shown as Frame Rate
            'frame_shape',  # Already shown as Resolution
            'total_frames',  # Already shown as Total Frames
            'dtype',  # Already shown as Data Type
            'color_format',  # Already shown as Color Format
            'compression',  # Already shown as Compression
            'compression_opts',  # Already shown as Compression Level
            'file_path',  # Already shown as Path
            'format_version',  # Already shown as Format Version
            'user_sample_name',  # Already shown as Sample
            'user_operator',  # Already shown as Operator
            'user_system_name',  # Already shown as System
            'created_at',  # Already shown as Recorded
            'timestamp',  # Already shown as Recorded
            'description',  # Already shown as Description
            # True duplicates - same data with different names
            'original_frame_shape',  # Duplicate of frame_shape
            'frame_size',  # Calculated from frame_shape
            'user_save_path',  # Duplicate of file_path
        }
        
        # Sort additional metadata in logical order by category
        def metadata_sort_key(key):
            # Define logical ordering by category with specific field priorities
            key_lower = key.lower()
            
            # Category 0: GPU/Processing configuration (setup info)
            if 'gpu_acceleration' in key_lower:
                return (0, 0, key)
            elif 'gpu_frames_processed' in key_lower:
                return (0, 1, key)
            elif 'chunk_size' in key_lower:
                return (0, 2, key)
            elif 'downscale_factor' in key_lower:
                return (0, 3, key)
            
            # Category 1: Compression settings
            elif 'original_compression' in key_lower:
                return (1, 0, key)
            elif 'post_compressed' in key_lower:
                return (1, 1, key)
            elif 'post_compression' in key_lower:
                return (1, 2, key)
            elif 'post_processed' in key_lower:
                return (1, 3, key)
            elif 'quality_reduction' in key_lower:
                return (1, 4, key)
            
            # Category 2: Timing and performance
            elif 'avg_downscale_time' in key_lower or 'avg' in key_lower:
                return (2, 0, key)
            elif 'fps_efficiency' in key_lower:
                return (2, 1, key)
            elif 'processing_time' in key_lower:
                return (2, 2, key)
            elif 'finished_at' in key_lower or 'finished' in key_lower:
                return (2, 3, key)
            elif 'post_processing_timestamp' in key_lower:
                return (2, 4, key)
            
            # Category 3: Frame statistics (min, max, mean)
            elif key_lower == 'min':
                return (3, 0, key)
            elif key_lower == 'max':
                return (3, 1, key)
            elif key_lower == 'mean':
                return (3, 2, key)
            elif 'median' in key_lower or 'std' in key_lower:
                return (3, 3, key)
            
            # Category 4: Data size info
            elif 'frame_size' in key_lower:
                return (4, 0, key)
            elif 'recording_duration' in key_lower:
                return (4, 1, key)
            elif 'total_data' in key_lower:
                return (4, 2, key)
            
            # Category 5: System info
            elif 'system' in key_lower:
                return (5, 0, key)
            elif 'compression_level' in key_lower:
                return (5, 1, key)
            elif 'data_type' in key_lower:
                return (5, 2, key)
            
            # Category 6: Everything else (alphabetically)
            else:
                return (6, 0, key)
        
        additional_keys = sorted(
            [k for k in metadata.keys() if k not in displayed_keys and k not in skip_keys],
            key=metadata_sort_key
        )
        
        for key in additional_keys:
            value = metadata[key]
            
            # Format the key nicely
            display_key = key.replace('_', ' ').title()
            
            # Special handling for specific field names that need units
            key_lower = key.lower()
            
            # Check if display_key already ends with a unit suffix that needs to be converted
            unit_conversions = {
                ' Fps': ' (fps)',
                ' Mb': ' (MB)',
                ' Gb': ' (GB)',
                ' Bytes': ' (bytes)',
                ' S': ' (s)',
                ' Ms': ' (ms)',
            }
            
            converted = False
            for suffix, bracket_unit in unit_conversions.items():
                if display_key.endswith(suffix):
                    display_key = display_key[:-len(suffix)] + bracket_unit + ':'
                    converted = True
                    break
            
            if not converted:
                # Add units for fields that don't have them in the name
                if 'avg_downscale_time' in key_lower or 'processing_time' in key_lower:
                    display_key = display_key + ' (ms):'
                elif key_lower == 'min':
                    display_key = 'Min Pixel Intensity:'
                elif key_lower == 'max':
                    display_key = 'Max Pixel Intensity:'
                elif key_lower == 'mean':
                    display_key = 'Mean Pixel Intensity:'
                elif 'frame_size' in key_lower:
                    display_key = display_key + ' (bytes):'
                elif 'total_data' in key_lower and 'mb' not in key_lower:
                    display_key = display_key + ' (MB):'
                elif 'recording_duration' in key_lower and 's' not in key_lower:
                    display_key = display_key + ' (s):'
                elif 'fps_efficiency' in key_lower:
                    display_key = display_key + ' (%):'
                elif 'chunk_size' in key_lower:
                    display_key = 'Chunk Size:'  # No unit - it's dimensions
                elif 'downscale_factor' in key_lower:
                    display_key = 'Downscale Factor:'  # No unit - it's a ratio
                elif 'compression_level' in key_lower:
                    display_key = display_key + ':'  # No unit - it's a level (0-9)
                elif 'timestamp' in key_lower or 'finished_at' in key_lower:
                    display_key = display_key + ':'  # Timestamps don't need units
                elif 'gpu_frames_processed' in key_lower:
                    display_key = 'GPU Frames Processed:'  # No unit - it's a count
                else:
                    display_key = display_key + ':'
            
            # Format the value appropriately
            if isinstance(value, (list, tuple)):
                # For lists/tuples, format nicely
                if len(value) > 10:
                    display_value = f"[{len(value)} items]"
                elif len(value) <= 3:
                    # For small tuples (like chunk_size), show with × separator
                    display_value = " × ".join(str(v) for v in value)
                else:
                    display_value = ", ".join(str(v) for v in value)
            elif isinstance(value, bytes):
                # For bytes, show as hex or length
                if len(value) < 100:
                    display_value = value.decode('utf-8', errors='ignore')
                else:
                    display_value = f"[{len(value)} bytes]"
            elif isinstance(value, (int, float)):
                # Format numbers nicely
                if isinstance(value, float):
                    display_value = f"{value:.6g}"
                else:
                    display_value = f"{value:,}"
            else:
                # Convert to string and clean up any brackets or array notation
                display_value = str(value)
                # If it looks like an array/list string, try to clean it up
                if display_value.startswith('[') and display_value.endswith(']'):
                    # Remove brackets and split
                    clean_str = display_value[1:-1].strip()
                    # Check if it's space-separated numbers (like "1 1024 1296")
                    parts = clean_str.split()
                    if len(parts) <= 3 and all(p.replace('.', '').replace('-', '').isdigit() for p in parts):
                        display_value = " × ".join(parts)
                    else:
                        display_value = clean_str
            
            # Truncate very long values
            if len(display_value) > 200:
                display_value = display_value[:197] + "..."
            
            self._add_info_row(self.metadata_layout, display_key, display_value)

        # After populating all sections, apply a uniform label column width based on the longest label
        self._apply_label_widths()
