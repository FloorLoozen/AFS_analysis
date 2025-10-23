"""Info tab for displaying measurement metadata."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout
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
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 24, 8, 8)
        layout.setSpacing(10)
        
        # Measurement Info Section
        measurement_group = QGroupBox("Measurement Info")
        self.measurement_layout = QFormLayout(measurement_group)
        self.measurement_layout.setContentsMargins(8, 8, 8, 8)
        self.measurement_layout.setSpacing(5)
        self.measurement_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(measurement_group)
        
        # Video Info Section
        video_group = QGroupBox("Video Info")
        self.video_layout = QFormLayout(video_group)
        self.video_layout.setContentsMargins(8, 8, 8, 8)
        self.video_layout.setSpacing(5)
        self.video_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(video_group)
        
        # File Info Section
        file_group = QGroupBox("File Info")
        self.file_layout = QFormLayout(file_group)
        self.file_layout.setContentsMargins(8, 8, 8, 8)
        self.file_layout.setSpacing(5)
        self.file_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(file_group)
        
        layout.addStretch()
    
    def _add_info_row(self, layout, label_text, value_text=""):
        """Add a label-value row to a form layout."""
        # Label
        label = QLabel(label_text)
        label.setStyleSheet("color: #666;")
        
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
            return
        
        metadata = self.video_widget.get_metadata()
        
        if not metadata:
            self._clear_layout(self.measurement_layout)
            self._clear_layout(self.video_layout)
            self._clear_layout(self.file_layout)
            return
        
        # Clear existing content
        self._clear_layout(self.measurement_layout)
        self._clear_layout(self.video_layout)
        self._clear_layout(self.file_layout)
        
        # === MEASUREMENT INFO ===
        # Sample name
        if 'user_sample_name' in metadata and metadata['user_sample_name']:
            self._add_info_row(self.measurement_layout, "Sample:", str(metadata['user_sample_name']))
        
        # Operator
        if 'user_operator' in metadata and metadata['user_operator']:
            self._add_info_row(self.measurement_layout, "Operator:", str(metadata['user_operator']))
        
        # System name
        if 'user_system_name' in metadata and metadata['user_system_name']:
            self._add_info_row(self.measurement_layout, "System:", str(metadata['user_system_name']))
        
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
            except Exception:
                pass
        
        # Description
        if 'description' in metadata and metadata['description']:
            self._add_info_row(self.measurement_layout, "Description:", str(metadata['description']))
        
        # === VIDEO INFO ===
        # Resolution
        if 'frame_shape' in metadata:
            shape = metadata['frame_shape']
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                self._add_info_row(self.video_layout, "Resolution:", f"{shape[1]} × {shape[0]} px")
        
        # Frame count
        if 'total_frames' in metadata:
            self._add_info_row(self.video_layout, "Total Frames:", f"{metadata['total_frames']:,}")
        
        # FPS
        fps = None
        if 'actual_fps' in metadata:
            fps = metadata['actual_fps']
            self._add_info_row(self.video_layout, "Frame Rate:", f"{fps:.3f} fps")
        
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
            
            if 'compression_opts' in metadata and metadata['compression_opts']:
                self._add_info_row(self.video_layout, "Compression Level:", str(metadata['compression_opts']))
        
        # === FILE INFO ===
        # File path
        if 'file_path' in metadata:
            file_path = metadata['file_path']
            self._add_info_row(self.file_layout, "Path:", file_path)
            
            # File size
            try:
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                if size_mb > 1024:
                    size_str = f"{size_mb / 1024:.2f} GB ({file_size:,} bytes)"
                else:
                    size_str = f"{size_mb:.2f} MB ({file_size:,} bytes)"
                self._add_info_row(self.file_layout, "File Size:", size_str)
                
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
                            uncompressed_str = f"{uncompressed_mb / 1024:.2f} GB"
                        else:
                            uncompressed_str = f"{uncompressed_mb:.2f} MB"
                        self._add_info_row(self.file_layout, "Uncompressed:", uncompressed_str)
                        
                        # Compression ratio
                        compression_ratio = uncompressed_size / file_size
                        self._add_info_row(self.file_layout, "Compression Ratio:", f"{compression_ratio:.2f}×")
                        
                        # Space saved
                        space_saved = (1 - file_size / uncompressed_size) * 100
                        self._add_info_row(self.file_layout, "Space Saved:", f"{space_saved:.1f}%")
            except Exception:
                pass
        
        # Format version
        if 'format_version' in metadata:
            self._add_info_row(self.file_layout, "Format Version:", str(metadata['format_version']))
