"""Measurement information widget for AFS Analysis."""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QFrame, QGridLayout, QWidget
)
from PyQt5.QtCore import Qt
from datetime import datetime


class MeasurementInfoWidget(QGroupBox):
    """Widget for displaying measurement information and metadata."""

    def __init__(self):
        """Initialize measurement info widget."""
        super().__init__("Measurement Information")
        self.video_widget = None  # Will be set by main window
        self._init_ui()
    
    def set_video_widget(self, video_widget):
        """Set reference to video widget."""
        self.video_widget = video_widget
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create info frame with consistent styling
        frame = self._create_info_frame()
        main_layout.addWidget(frame)

    def _create_info_frame(self):
        """Create the main info frame with consistent styling."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Container widget for the grid to control alignment
        grid_container = QWidget()
        grid_layout = QVBoxLayout(grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create grid layout for aligned columns
        self.info_grid = QGridLayout()
        self.info_grid.setSpacing(5)
        self.info_grid.setContentsMargins(0, 0, 0, 0)
        self.info_grid.setColumnStretch(1, 1)  # Value column stretches
        
        # Create labels dictionary for easy updates
        self.info_labels = {}
        
        grid_layout.addLayout(self.info_grid)
        grid_layout.addStretch()
        
        # Add container with right alignment
        layout.addWidget(grid_container)
        layout.addStretch()
        
        return frame
    
    def _add_info_row(self, label_text: str, value_text: str, row: int):
        """Add a row to the info grid with right-aligned label."""
        # Create label with right alignment
        label = QLabel(f"{label_text}:")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setStyleSheet("color: #888; padding-right: 8px;")
        label.setMinimumWidth(80)
        
        # Create value label with left alignment
        value = QLabel(value_text)
        value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        value.setStyleSheet("color: #333;")
        
        # Add to grid
        self.info_grid.addWidget(label, row, 0)
        self.info_grid.addWidget(value, row, 1)
        
        # Store reference
        self.info_labels[label_text] = value
    
    def _clear_info_grid(self):
        """Clear all widgets from the info grid."""
        while self.info_grid.count():
            item = self.info_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.info_labels.clear()
    
    def update_info(self):
        """Update measurement information display - column format with more details."""
        if not self.video_widget:
            self._clear_info_grid()
            return
        
        metadata = self.video_widget.get_metadata()
        
        if not metadata:
            self._clear_info_grid()
            return
        
        # Clear existing content
        self._clear_info_grid()
        
        row = 0
        
        # Video information
        if 'total_frames' in metadata:
            self._add_info_row("Frames", str(metadata['total_frames']), row)
            row += 1
        
        if 'actual_fps' in metadata:
            fps = metadata['actual_fps']
            self._add_info_row("FPS", f"{fps:.1f}", row)
            row += 1
            
            # Duration
            if 'total_frames' in metadata:
                duration = metadata['total_frames'] / fps
                self._add_info_row("Duration", f"{duration:.2f} s", row)
                row += 1
        
        if 'frame_shape' in metadata:
            shape = metadata['frame_shape']
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                self._add_info_row("Resolution", f"{shape[1]} × {shape[0]}", row)
                row += 1
        
        # HDF5 Metadata - Sample information
        if 'user_sample_name' in metadata and metadata['user_sample_name']:
            self._add_info_row("Sample", str(metadata['user_sample_name']), row)
            row += 1
        
        # Timestamp - check both field names
        timestamp_field = None
        if 'created_at' in metadata:
            timestamp_field = 'created_at'
        elif 'timestamp' in metadata:
            timestamp_field = 'timestamp'
        
        if timestamp_field:
            try:
                # Try to format timestamp if it's a number or string
                timestamp = metadata[timestamp_field]
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, str):
                    # Try to parse ISO format or just display as-is
                    try:
                        # Try ISO format first
                        if 'T' in timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            time_str = timestamp
                    except:
                        time_str = timestamp
                else:
                    time_str = str(timestamp)
                self._add_info_row("Recorded", time_str, row)
                row += 1
            except Exception as e:
                pass
        
        # Operator - check both field names
        operator_field = None
        if 'user_operator' in metadata and metadata['user_operator']:
            operator_field = 'user_operator'
        elif 'operator' in metadata and metadata['operator']:
            operator_field = 'operator'
        
        if operator_field:
            self._add_info_row("Operator", str(metadata[operator_field]), row)
            row += 1
        
        # System name
        if 'user_system_name' in metadata and metadata['user_system_name']:
            self._add_info_row("System", str(metadata['user_system_name']), row)
            row += 1
        
        # Description/Notes
        if 'description' in metadata and metadata['description']:
            notes_text = str(metadata['description'])
            # Truncate if too long
            if len(notes_text) > 50:
                notes_text = notes_text[:47] + "..."
            self._add_info_row("Description", notes_text, row)
            row += 1
        elif 'notes' in metadata and metadata['notes']:
            notes_text = str(metadata['notes'])
            # Truncate if too long
            if len(notes_text) > 50:
                notes_text = notes_text[:47] + "..."
            self._add_info_row("Notes", notes_text, row)
            row += 1
