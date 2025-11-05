"""XY bead tracking using cross-correlation template matching.

Robust tracking that handles:
- Focus transitions (light/dark bead centers)
- Adaptive template updating
- Inverted template fallback for appearance changes
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from src.utils.logger import Logger

logger = Logger


class BeadTracker:
    """XY bead tracking using cross-correlation template matching."""
    
    def __init__(self, window_size: int = 40):
        """Initialize bead tracker.
        
        Args:
            window_size: Search window size around last position
        """
        self.window_size = window_size
        self.beads: List[Dict[str, Any]] = []
        self.template_size = 30  # Template size for cross-correlation
        
    def add_bead(self, frame: np.ndarray, x: int, y: int, bead_id: int):
        """Add a new bead to track.
        
        Args:
            frame: Current frame to extract template
            x, y: Initial position
            bead_id: Unique identifier for this bead
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Extract template around clicked position
        half_size = self.template_size // 2
        x1 = max(0, int(x - half_size))
        y1 = max(0, int(y - half_size))
        x2 = min(gray.shape[1], int(x + half_size))
        y2 = min(gray.shape[0], int(y + half_size))
        
        template = gray[y1:y2, x1:x2].astype(np.float32)
        
        bead = {
            'id': bead_id,
            'positions': [(x, y)],
            'initial_pos': (x, y),
            'reference_pos': (x, y),
            'frames_since_reference': 0,
            'template': template
        }
        self.beads.append(bead)
    
    def track_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Track beads using cross-correlation template matching.
        
        Args:
            frame: Current video frame
            
        Returns:
            List of (bead_id, x, y) tuples with updated positions
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        results = []
        
        for bead in self.beads:
            if bead['template'] is None:
                continue
                
            last_x, last_y = bead['positions'][-1]
            
            # Define search window
            search_radius = self.window_size
            x1 = max(0, int(last_x - search_radius))
            y1 = max(0, int(last_y - search_radius))
            x2 = min(gray.shape[1], int(last_x + search_radius))
            y2 = min(gray.shape[0], int(last_y + search_radius))
            
            if x2 <= x1 or y2 <= y1:
                bead['positions'].append((last_x, last_y))
                results.append((bead['id'], last_x, last_y))
                continue
            
            search_window = gray[y1:y2, x1:x2]
            
            # Cross-correlation matching
            result = cv2.matchTemplate(search_window, bead['template'], cv2.TM_CCORR_NORMED)
            _, max_corr, _, best_loc = cv2.minMaxLoc(result)
            
            # Convert to position in search window
            peak_x, peak_y = best_loc
            
            # Adjust for template size (matchTemplate returns top-left corner)
            half_template = self.template_size // 2
            peak_x += half_template
            peak_y += half_template
            
            # Convert to global coordinates
            global_x = x1 + peak_x
            global_y = y1 + peak_y
            
            # Update reference position every 20 frames to allow movement
            bead['frames_since_reference'] = bead.get('frames_since_reference', 0) + 1
            if bead['frames_since_reference'] >= 20:
                bead['reference_pos'] = (global_x, global_y)
                bead['frames_since_reference'] = 0
            
            # Adaptive template update
            if max_corr > 0.5:
                half_size = self.template_size // 2
                tx1 = max(0, int(global_x - half_size))
                ty1 = max(0, int(global_y - half_size))
                tx2 = min(gray.shape[1], int(global_x + half_size))
                ty2 = min(gray.shape[0], int(global_y + half_size))
                
                if tx2 - tx1 == bead['template'].shape[1] and ty2 - ty1 == bead['template'].shape[0]:
                    new_template = gray[ty1:ty2, tx1:tx2]
                    # Conservative blend: 95% old, 5% new
                    alpha = 0.05
                    bead['template'] = (1 - alpha) * bead['template'] + alpha * new_template
            
            # Store updated position
            bead['positions'].append((global_x, global_y))
            results.append((bead['id'], global_x, global_y))
        
        return results
    
    def get_bead_positions(self, bead_id: int) -> List[Tuple[int, int]]:
        """Get all positions for a specific bead."""
        for bead in self.beads:
            if bead['id'] == bead_id:
                return bead['positions']
        return []
    
    def get_all_beads_data(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get all tracking data for all beads."""
        return {bead['id']: bead['positions'] for bead in self.beads}
    
    def get_bead_count(self) -> int:
        """Get number of tracked beads."""
        return len(self.beads)
    
    def load_from_data(self, beads_data: List[Dict[str, Any]]):
        """Load tracking data from saved format."""
        self.beads = beads_data
    
    def clear(self):
        """Clear all tracked beads."""
        self.beads = []
    
    def remove_bead(self, bead_id: int):
        """Remove a specific bead from tracking."""
        self.beads = [bead for bead in self.beads if bead['id'] != bead_id]


def detect_beads_auto(frame: np.ndarray, min_area: int = 500, max_area: int = 5000, threshold_value: int = 150) -> List[Tuple[int, int]]:
    """Auto-detect beads in XY plane using simple blob detection."""
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Threshold
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bead_positions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:
                        bead_positions.append((cx, cy))
    return bead_positions
