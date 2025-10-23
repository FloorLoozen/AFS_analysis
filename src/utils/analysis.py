"""Analysis algorithms for video data."""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any


class TrackingResult:
    """Container for tracking results."""
    
    def __init__(self):
        self.positions: List[Tuple[float, float]] = []
        self.frame_indices: List[int] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_position(self, frame_index: int, x: float, y: float):
        """Add a tracked position."""
        self.frame_indices.append(frame_index)
        self.positions.append((x, y))
    
    def get_x_positions(self) -> np.ndarray:
        """Get X positions as array."""
        return np.array([pos[0] for pos in self.positions])
    
    def get_y_positions(self) -> np.ndarray:
        """Get Y positions as array."""
        return np.array([pos[1] for pos in self.positions])
    
    def get_displacement(self) -> np.ndarray:
        """Calculate total displacement from origin."""
        positions = np.array(self.positions)
        if len(positions) == 0:
            return np.array([])
        
        origin = positions[0]
        return np.sqrt(np.sum((positions - origin) ** 2, axis=1))
    
    def get_velocity(self, dt: float = 1.0) -> np.ndarray:
        """
        Calculate velocity between frames.
        
        Args:
            dt: Time step between frames
        
        Returns:
            Array of velocities
        """
        positions = np.array(self.positions)
        if len(positions) < 2:
            return np.array([])
        
        diff = np.diff(positions, axis=0)
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        return distances / dt


class XYTracker:
    """Tracks XY position of features in video frames."""
    
    def __init__(self):
        self.tracking_result = TrackingResult()
    
    def track_centroid(self, frames: np.ndarray) -> TrackingResult:
        """
        Track centroid of brightest region across frames.
        
        Args:
            frames: Array of frames (num_frames, height, width, channels) or (num_frames, height, width)
        
        Returns:
            TrackingResult with positions
        """
        self.tracking_result = TrackingResult()
        
        for i, frame in enumerate(frames):
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2).astype(np.uint8)
            else:
                gray = frame
            
            # Find brightest region (simple centroid)
            # Threshold at 50% of max intensity
            threshold = np.max(gray) * 0.5
            mask = gray > threshold
            
            if np.any(mask):
                # Calculate centroid
                y_coords, x_coords = np.where(mask)
                x_center = np.mean(x_coords)
                y_center = np.mean(y_coords)
                
                self.tracking_result.add_position(i, float(x_center), float(y_center))
            else:
                # No bright region found, use previous position or center
                if len(self.tracking_result.positions) > 0:
                    prev_pos = self.tracking_result.positions[-1]
                    self.tracking_result.add_position(i, prev_pos[0], prev_pos[1])
                else:
                    # Use frame center
                    h, w = gray.shape
                    self.tracking_result.add_position(i, w / 2, h / 2)
        
        self.tracking_result.metadata['method'] = 'centroid'
        return self.tracking_result
    
    def track_template(self, frames: np.ndarray, template: np.ndarray) -> TrackingResult:
        """
        Track using template matching.
        
        Args:
            frames: Array of frames
            template: Template to match
        
        Returns:
            TrackingResult with positions
        """
        # Placeholder for template matching implementation
        # TODO: Implement cv2.matchTemplate based tracking
        self.tracking_result = TrackingResult()
        self.tracking_result.metadata['method'] = 'template'
        return self.tracking_result


class ZTracker:
    """Tracks Z-axis (vertical) displacement in video frames."""
    
    def __init__(self):
        self.z_values: List[float] = []
        self.frame_indices: List[int] = []
    
    def track_intensity_variance(self, frames: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[List[int], List[float]]:
        """
        Track Z displacement by measuring focus/sharpness (variance of Laplacian).
        
        Args:
            frames: Array of frames
            roi: Optional ROI as (x, y, width, height)
        
        Returns:
            Tuple of (frame_indices, z_values)
        """
        self.z_values = []
        self.frame_indices = []
        
        for i, frame in enumerate(frames):
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2).astype(np.uint8)
            else:
                gray = frame
            
            # Extract ROI if specified
            if roi:
                x, y, w, h = roi
                gray = gray[y:y+h, x:x+w]
            
            # Calculate variance of Laplacian (focus measure)
            if cv2 is not None:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                variance = float(laplacian.var())
            else:
                # Fallback: use standard deviation as focus measure
                variance = float(np.std(gray))
            
            self.z_values.append(variance)
            self.frame_indices.append(i)
        
        return self.frame_indices, self.z_values
    
    def get_relative_z(self, reference_frame: int = 0) -> np.ndarray:
        """
        Get Z values relative to a reference frame.
        
        Args:
            reference_frame: Index of frame to use as reference
        
        Returns:
            Array of relative Z values
        """
        z_array = np.array(self.z_values)
        if len(z_array) == 0:
            return np.array([])
        
        if reference_frame >= len(z_array):
            reference_frame = 0
        
        reference_value = z_array[reference_frame]
        return z_array - reference_value
