"""Frame processing utilities - separate from UI display logic."""

import cv2
import numpy as np
from typing import Tuple


class FrameProcessor:
    """Handles frame processing operations."""
    
    @staticmethod
    def resize_to_fit(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize frame to fit within target dimensions while maintaining aspect ratio.
        
        Args:
            frame: Input frame (RGB or BGR)
            target_width: Maximum width
            target_height: Maximum height
        
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        
        # Calculate scaling to fit target size while maintaining aspect ratio
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
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
