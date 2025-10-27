"""Info tab for displaying measurement metadata."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QScrollArea
)
from PyQt5.QtCore import Qt
from datetime import datetime
import os


class InfoTab(QWidget):
    """Tab for displaying measurement information and metadata."""

    def __init__(self):
        """Initialize info tab."""
        super().__init__()
        self.video_widget = None
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
        
        # Measurement Info Section
        measurement_group = QGroupBox("Measurement Info")
        self.measurement_layout = QFormLayout(measurement_group)
        self.measurement_layout.setContentsMargins(8, 8, 8, 8)
        self.measurement_layout.setSpacing(5)
        self.measurement_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.measurement_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.addWidget(measurement_group)
        
        # Video Info Section
        video_group = QGroupBox("Video Info")
        self.video_layout = QFormLayout(video_group)
        self.video_layout.setContentsMargins(8, 8, 8, 8)
        self.video_layout.setSpacing(5)
        self.video_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.video_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.addWidget(video_group)
        
        # File Info Section
        file_group = QGroupBox("File Info")
        self.file_layout = QFormLayout(file_group)
        self.file_layout.setContentsMargins(8, 8, 8, 8)
        self.file_layout.setSpacing(5)
        self.file_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.file_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.addWidget(file_group)
        
        # Additional Metadata Section (for all other HDF5 attributes)
        metadata_group = QGroupBox("Additional Metadata")
        self.metadata_layout = QFormLayout(metadata_group)
        self.metadata_layout.setContentsMargins(8, 8, 8, 8)
        self.metadata_layout.setSpacing(5)
        self.metadata_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.metadata_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.addWidget(metadata_group)
        
        layout.addStretch()
        
        # Set scroll widget and add to main layout
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def _add_info_row(self, layout, label_text, value_text=""):
        """Add a label-value row to a form layout."""
        # Label with fixed width for perfect alignment
        label = QLabel(label_text)
        label.setFixedWidth(160)  # Fixed width ensures perfect alignment
        label.setStyleSheet("color: #666;")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        # Value
        value = QLabel(value_text)
        value.setWordWrap(True)
        value.setStyleSheet("color: #222;")
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Add to form layout
        layout.addRow(label, value)
        
        return value
    
    def _clear_layout(self, layout):
        """Clear all rows from a form layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
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
                self._add_info_row(self.video_layout, "Resolution (px):", f"{shape[1]} × {shape[0]}")
            displayed_keys.add('frame_shape')
        
        # Frame count
        if 'total_frames' in metadata:
            self._add_info_row(self.video_layout, "Total Frames:", f"{metadata['total_frames']:,}")
            displayed_keys.add('total_frames')
        
        # FPS
        fps = None
        if 'actual_fps' in metadata:
            fps = metadata['actual_fps']
            self._add_info_row(self.video_layout, "Actual FPS (fps):", f"{fps:.3f}")
            displayed_keys.add('actual_fps')
        
        if 'fps' in metadata:
            target_fps = metadata['fps']
            self._add_info_row(self.video_layout, "Target FPS (fps):", f"{target_fps:.3f}")
            displayed_keys.add('fps')
            if fps is None:
                fps = target_fps
        
        # Duration
        if fps and fps > 0 and 'total_frames' in metadata:
            duration_sec = metadata['total_frames'] / fps
            minutes = int(duration_sec // 60)
            seconds = duration_sec % 60
            duration_str = f"{minutes}:{seconds:05.2f}"
            self._add_info_row(self.video_layout, "Duration:", duration_str)
        
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
                    size_str = f"{size_mb / 1024:.2f}"
                    self._add_info_row(self.file_layout, "File Size (GB):", size_str)
                else:
                    size_str = f"{size_mb:.2f}"
                    self._add_info_row(self.file_layout, "File Size (MB):", size_str)
                
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
                        
                        if uncompressed_mb > 1024:
                            uncompressed_str = f"{uncompressed_mb / 1024:.2f}"
                            self._add_info_row(self.file_layout, "Uncompressed (GB):", uncompressed_str)
                        else:
                            uncompressed_str = f"{uncompressed_mb:.2f}"
                            self._add_info_row(self.file_layout, "Uncompressed (MB):", uncompressed_str)
                        
                        # Compression ratio
                        compression_ratio = uncompressed_size / file_size
                        self._add_info_row(self.file_layout, "Compression Ratio:", f"{compression_ratio:.2f}×")
                        
                        # Space saved
                        space_saved = (1 - file_size / uncompressed_size) * 100
                        self._add_info_row(self.file_layout, "Space Saved (%):", f"{space_saved:.1f}")
            except Exception:
                pass
        
        # Format version
        if 'format_version' in metadata:
            self._add_info_row(self.file_layout, "Format Version:", str(metadata['format_version']))
            displayed_keys.add('format_version')
        
        # === ADDITIONAL METADATA ===
        # Display all remaining metadata that hasn't been shown yet
        additional_keys = sorted([k for k in metadata.keys() if k not in displayed_keys])
        
        for key in additional_keys:
            value = metadata[key]
            
            # Format the key nicely and extract units
            display_key = key.replace('_', ' ').title()
            
            # Detect and move units to brackets
            unit_mappings = {
                'Fps': '(fps)',
                'Mb': '(MB)',
                'Gb': '(GB)',
                'Bytes': '(bytes)',
                'S': '(s)',  # seconds
                'Ms': '(ms)',  # milliseconds
            }
            
            # Check if key ends with a unit indicator
            for unit_suffix, bracket_unit in unit_mappings.items():
                if display_key.endswith(f' {unit_suffix}'):
                    # Remove unit from key and add to brackets
                    display_key = display_key[:-len(unit_suffix)-1].strip() + f' {bracket_unit}:'
                    break
            else:
                # No unit found, just add colon
                display_key = display_key + ':'
            
            # Format the value appropriately
            if isinstance(value, (list, tuple)):
                # For lists/tuples, show as comma-separated
                if len(value) > 10:
                    display_value = f"[{len(value)} items]"
                else:
                    display_value = str(value)
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
                display_value = str(value)
            
            # Truncate very long values
            if len(display_value) > 200:
                display_value = display_value[:197] + "..."
            
            self._add_info_row(self.metadata_layout, display_key, display_value)
