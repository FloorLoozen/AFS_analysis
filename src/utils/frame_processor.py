"""Frame processing utilities - separate from UI display logic."""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class FrameProcessor:
    """Handles frame processing operations."""
    
    @staticmethod
    def resize_to_fit(frame: np.ndarray, target_width: int, target_height: int, 
                      keep_aspect_ratio: bool = True) -> Tuple[np.ndarray, float]:
        """
        Resize frame to fit within target dimensions.
        
        Args:
            frame: Input frame (RGB or BGR)
            target_width: Maximum width
            target_height: Maximum height
            keep_aspect_ratio: Whether to maintain aspect ratio (default True)
        
        Returns:
            Tuple of (resized_frame, scale_factor)
        """
        if frame is None or frame.size == 0:
            return frame, 1.0
            
        h, w = frame.shape[:2]
        
        if keep_aspect_ratio:
            # Calculate scaling to fit target size while maintaining aspect ratio
            scale = min(target_width / w, target_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = target_width, target_height
            scale = min(target_width / w, target_height / h)
        
        # Resize frame using fast interpolation
        if scale < 1.0:
            # Downsampling - use AREA for better quality
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Upsampling - use LINEAR for speed
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return resized, scale
    
    @staticmethod
    def draw_bead_overlays(frame: np.ndarray, bead_positions: Dict[int, Tuple[int, int]], 
                          bead_traces: Dict[int, list] = None, box_size: int = 30, box_thickness: int = 1) -> np.ndarray:
        """
        Draw tracking overlays on frame with live traces.
        
        Args:
            frame: Input frame (will be copied, not modified)
            bead_positions: Dictionary mapping bead_id to (x, y) position
            bead_traces: Dictionary mapping bead_id to list of (x, y) positions for trace history
            box_size: Size of box around bead
            box_thickness: Thickness of box lines
        
        Returns:
            Frame with overlays drawn
        """
        if not bead_positions or frame is None:
            return frame
        
        # Work on a copy to not modify original
        frame_with_overlay = frame.copy()
        half_size = box_size // 2
        
        # Color for overlays (green in RGB)
        color = (0, 255, 0)
        trace_color = (255, 255, 0)  # Cyan for traces
        
        # Draw traces first (so they appear behind boxes)
        if bead_traces:
            for bead_id, trace in bead_traces.items():
                if len(trace) > 1:
                    # Draw trace as connected lines
                    for i in range(len(trace) - 1):
                        # Convert to integers for OpenCV (handles sub-pixel float coordinates)
                        pt1 = (int(round(trace[i][0])), int(round(trace[i][1])))
                        pt2 = (int(round(trace[i + 1][0])), int(round(trace[i + 1][1])))
                        cv2.line(frame_with_overlay, pt1, pt2, trace_color, 1)
        
        # Draw boxes and labels on top of traces
        for bead_id, (x, y) in bead_positions.items():
            # Convert to integers for OpenCV (handles sub-pixel float coordinates)
            x_int, y_int = int(round(x)), int(round(y))
            
            # Draw box around bead
            pt1 = (x_int - half_size, y_int - half_size)
            pt2 = (x_int + half_size, y_int + half_size)
            cv2.rectangle(frame_with_overlay, pt1, pt2, color, box_thickness)
            
            # Draw label
            label_pos = (x_int - half_size, y_int - half_size - 5)
            cv2.putText(frame_with_overlay, str(bead_id + 1), 
                       label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame_with_overlay
    
    @staticmethod
    def prepare_for_display(frame: np.ndarray, target_width: int, target_height: int,
                           bead_positions: Optional[Dict[int, Tuple[int, int]]] = None) -> np.ndarray:
        """
        Prepare frame for display: resize and optionally add overlays.
        
        Args:
            frame: Input frame
            target_width: Target display width
            target_height: Target display height
            bead_positions: Optional bead positions to draw
        
        Returns:
            Processed frame ready for display
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Add overlays first (on full resolution for accuracy)
        if bead_positions:
            frame = FrameProcessor.draw_bead_overlays(frame, bead_positions)
        
        # Then resize for display
        resized, _ = FrameProcessor.resize_to_fit(frame, target_width, target_height)
        
        return resized
    
    @staticmethod
    def extract_roi(frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Extract region of interest from frame.
        
        Args:
            frame: Input frame
            x, y: Top-left corner coordinates
            width, height: ROI dimensions
        
        Returns:
            Cropped frame
        """
        h, w = frame.shape[:2]
        
        # Clamp coordinates
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x2 = min(x + width, w)
        y2 = min(y + height, h)
        
        return frame[y:y2, x:x2].copy()
    
    @staticmethod
    def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        if len(frame.shape) == 2:
            return frame  # Already grayscale
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    @staticmethod
    def apply_contrast(frame: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
        """
        Adjust frame contrast and brightness.
        
        Args:
            frame: Input frame
            alpha: Contrast control (1.0 = no change)
            beta: Brightness control (0 = no change)
        
        Returns:
            Adjusted frame
        """
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    @staticmethod
    def get_frame_info(frame: np.ndarray) -> dict:
        """
        Get information about a frame.
        
        Returns:
            Dictionary with frame properties
        """
        if frame is None:
            return {}
        
        info = {
            'shape': frame.shape,
            'dtype': str(frame.dtype),
            'min': float(np.min(frame)),
            'max': float(np.max(frame)),
            'mean': float(np.mean(frame))
        }
        
        if len(frame.shape) == 3:
            info['channels'] = frame.shape[2]
        else:
            info['channels'] = 1
        
        return info
