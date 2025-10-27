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


def detect_beads_auto(frame: np.ndarray, min_area: int = 50, max_area: int = 5000, 
                       threshold_value: int = 150) -> List[Tuple[int, int]]:
    """
    Automatically detect bright circular beads in a frame.
    
    Args:
        frame: Input frame (RGB or grayscale)
        min_area: Minimum area of bead in pixels
        max_area: Maximum area of bead in pixels
        threshold_value: Brightness threshold (0-255)
    
    Returns:
        List of (x, y) center positions of detected beads
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Apply threshold to get bright regions
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bead_positions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if min_area <= area <= max_area:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Check circularity (optional, helps filter out non-bead objects)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Reasonably circular
                        bead_positions.append((cx, cy))
    
    return bead_positions


class BeadTracker:
    """Tracks multiple beads using adaptive template matching."""
    
    def __init__(self, window_size: int = 40):
        """
        Initialize bead tracker.
        
        Args:
            window_size: Size of tracking window around bead
        """
        self.window_size = window_size
        self.beads: List[Dict[str, Any]] = []  # List of beads with their data
        
    def add_bead(self, frame: np.ndarray, x: int, y: int, bead_id: int):
        """
        Add a new bead to track.
        
        Args:
            frame: First frame to extract template from
            x, y: Initial position of bead center
            bead_id: Unique ID for this bead
        """
        # Extract template around the bead
        half_size = self.window_size // 2
        h, w = frame.shape[:2]
        
        # Ensure we don't go out of bounds
        y1 = max(0, y - half_size)
        y2 = min(h, y + half_size)
        x1 = max(0, x - half_size)
        x2 = min(w, x + half_size)
        
        template = frame[y1:y2, x1:x2].copy()
        
        # Convert to grayscale if needed
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        else:
            template_gray = template
        
        bead = {
            'id': bead_id,
            'template': template_gray,
            'positions': [(x, y)],  # List of (x, y) positions for each frame
            'initial_pos': (x, y),
            'lost_frames': 0,  # Counter for consecutive lost frames
            'last_good_match': 1.0  # Last good match score
        }
        
        self.beads.append(bead)
    
    def track_frame(self, frame: np.ndarray, search_radius: int = 120, 
                    min_match_score: float = 0.25, update_template: bool = True,
                    max_lost_frames: int = 50) -> List[Tuple[int, int, int]]:
        """
        Track all beads in a new frame with adaptive template matching.
        
        Args:
            frame: New frame to track beads in
            search_radius: How far from last position to search (increased default)
            min_match_score: Minimum correlation score to accept (0-1, lowered for robustness)
            update_template: Whether to update template with good matches
            max_lost_frames: Maximum frames to keep searching before giving up
        
        Returns:
            List of (bead_id, x, y) positions
        """
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            frame_gray = frame
        
        results = []
        
        for bead in self.beads:
            template = bead['template']
            last_x, last_y = bead['positions'][-1]
            
            # Define search region
            h, w = frame_gray.shape
            th, tw = template.shape
            
            # Increase search radius if bead was recently lost
            effective_radius = search_radius
            if bead['lost_frames'] > 0:
                # Expand search for lost beads - up to 3x the normal radius
                effective_radius = min(search_radius * (1 + bead['lost_frames'] // 10), 300)
            
            # Search region around last known position
            x1 = max(0, last_x - effective_radius)
            y1 = max(0, last_y - effective_radius)
            x2 = min(w, last_x + effective_radius + tw)
            y2 = min(h, last_y + effective_radius + th)
            
            search_region = frame_gray[y1:y2, x1:x2]
            
            # Template matching
            if search_region.shape[0] < th or search_region.shape[1] < tw:
                # Search region too small, use last position
                new_x, new_y = last_x, last_y
                match_score = bead.get('last_good_match', 0.0) * 0.95  # Decay confidence
            else:
                # Try multiple matching methods for robustness
                result_normed = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                _, max_val_normed, _, max_loc_normed = cv2.minMaxLoc(result_normed)
                
                # Also try TM_CCORR_NORMED for better handling of brightness changes
                result_ccorr = cv2.matchTemplate(search_region, template, cv2.TM_CCORR_NORMED)
                _, max_val_ccorr, _, max_loc_ccorr = cv2.minMaxLoc(result_ccorr)
                
                # Use the better match
                if max_val_normed > max_val_ccorr:
                    max_val = max_val_normed
                    max_loc = max_loc_normed
                else:
                    max_val = max_val_ccorr
                    max_loc = max_loc_ccorr
                
                # Convert to frame coordinates
                match_x = x1 + max_loc[0] + tw // 2
                match_y = y1 + max_loc[1] + th // 2
                match_score = max_val
                
                # Distance from last position - penalize large jumps between beads
                dx = match_x - last_x
                dy = match_y - last_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Adaptive threshold: lower for stationary beads, higher for moving
                adaptive_threshold = min_match_score
                if bead['lost_frames'] == 0 and distance < 20:
                    # Bead moving slowly - be more strict
                    adaptive_threshold = max(min_match_score, 0.4)
                elif distance > effective_radius * 0.7:
                    # Large jump - could be tracking wrong bead
                    adaptive_threshold = max(min_match_score, 0.5)
                
                # Check if match is good enough
                if match_score >= adaptive_threshold:
                    # Good match - accept new position
                    new_x, new_y = match_x, match_y
                    
                    # Update template if match is strong
                    if update_template and match_score > 0.55:
                        # Extract new template region
                        ny1 = max(0, new_y - th // 2)
                        ny2 = min(h, new_y + th // 2)
                        nx1 = max(0, new_x - tw // 2)
                        nx2 = min(w, new_x + tw // 2)
                        
                        new_template = frame_gray[ny1:ny2, nx1:nx2]
                        
                        # Adaptive template update: blend with old template
                        if new_template.shape == template.shape:
                            # Higher blending for better matches
                            alpha = 0.2 if match_score > 0.7 else 0.1
                            bead['template'] = cv2.addWeighted(
                                template.astype(np.float32), 1 - alpha,
                                new_template.astype(np.float32), alpha, 0
                            ).astype(np.uint8)
                    
                    bead['lost_frames'] = 0
                    bead['last_good_match'] = match_score
                else:
                    # Poor match - might be lost, keep last position
                    new_x, new_y = last_x, last_y
                    bead['lost_frames'] += 1
                    bead['last_good_match'] = max(match_score, bead.get('last_good_match', 0.0) * 0.95)
            
            # Store new position (always store, maintaining bead identity)
            bead['positions'].append((new_x, new_y))
            results.append((bead['id'], new_x, new_y))
        
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
        """
        Load tracking data from saved format.
        
        Args:
            beads_data: List of bead dictionaries with id, positions, template
        """
        self.beads = beads_data
    
    def clear(self):
        """Clear all tracked beads."""
        self.beads = []
    
    def remove_bead(self, bead_id: int):
        """Remove a specific bead from tracking."""
        self.beads = [bead for bead in self.beads if bead['id'] != bead_id]


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
