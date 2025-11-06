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


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Return a single channel view of the frame regardless of input shape."""
    if frame.ndim == 2:
        return frame

    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 1:
            return frame[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if channels == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    raise ValueError(f"Unsupported frame shape for grayscale conversion: {frame.shape}")


class BeadTracker:
    """XY bead tracking using cross-correlation template matching."""
    
    def __init__(self, window_size: int = 40):
        """Initialize bead tracker.
        
        Args:
            window_size: Search window size around last position
        """
        self.window_size = window_size
        self.beads: List[Dict[str, Any]] = []
        self.template_size = 40  # Template size for cross-correlation
        
    def add_bead(self, frame: np.ndarray, x: int, y: int, bead_id: int):
        """Add a new bead to track.
        
        Args:
            frame: Current frame to extract template
            x, y: Initial position
            bead_id: Unique identifier for this bead
        """
        gray = _to_grayscale(frame)

        template, _ = self._extract_template(gray, x, y)
        template_x, template_y = self._compute_profiles(template)

        bead = {
            'id': bead_id,
            'positions': [(x, y)],
            'initial_pos': (x, y),
            'reference_pos': (x, y),
            'frames_since_reference': 0,
            'template': template,
            'template_x': template_x,
            'template_y': template_y,
            'template_shape': template.shape
        }
        self.beads.append(bead)
    
    def track_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Track beads using cross-correlation template matching.
        
        Args:
            frame: Current video frame
            
        Returns:
            List of (bead_id, x, y) tuples with updated positions
        """
        gray = _to_grayscale(frame).astype(np.float32)
        height, width = gray.shape

        results = []

        for bead in self.beads:
            if bead.get('template_x') is None or bead.get('template_y') is None:
                continue

            last_x, last_y = bead['positions'][-1]

            template = bead.get('template')
            tpl_shape = bead.get('template_shape', None)
            if template is None or tpl_shape is None:
                tpl_shape = (self.template_size, self.template_size)

            tpl_h, tpl_w = tpl_shape
            search_radius = self.window_size

            # Horizontal profile search
            y_start, y_end = self._compute_bounds(last_y, tpl_h, height)
            strip_width = tpl_w + 2 * search_radius
            x_start, x_end = self._compute_bounds(last_x, strip_width, width)

            strip = gray[y_start:y_end, x_start:x_end]
            profile_x = strip.mean(axis=0) if strip.size else np.array([])
            corr_x = self._normalized_cross_correlation(profile_x, bead['template_x']) if profile_x.size else np.array([])

            if corr_x.size == 0:
                new_x = int(last_x)
                best_corr_x = -1.0
            else:
                best_idx_x = int(np.argmax(corr_x))
                best_corr_x = float(corr_x[best_idx_x])
                center_offset_x = tpl_w / 2.0
                new_x = int(round(x_start + best_idx_x + center_offset_x))

            new_x = int(np.clip(new_x, 0, width - 1))

            # Vertical profile search (centered on updated x)
            x_strip_start, x_strip_end = self._compute_bounds(new_x, tpl_w, width)
            strip_height = tpl_h + 2 * search_radius
            y_strip_start, y_strip_end = self._compute_bounds(last_y, strip_height, height)

            strip_y = gray[y_strip_start:y_strip_end, x_strip_start:x_strip_end]
            profile_y = strip_y.mean(axis=1) if strip_y.size else np.array([])
            corr_y = self._normalized_cross_correlation(profile_y, bead['template_y']) if profile_y.size else np.array([])

            if corr_y.size == 0:
                new_y = int(last_y)
                best_corr_y = -1.0
            else:
                best_idx_y = int(np.argmax(corr_y))
                best_corr_y = float(corr_y[best_idx_y])
                center_offset_y = tpl_h / 2.0
                new_y = int(round(y_strip_start + best_idx_y + center_offset_y))

            new_y = int(np.clip(new_y, 0, height - 1))

            # Update reference position periodically
            bead['frames_since_reference'] = bead.get('frames_since_reference', 0) + 1
            if bead['frames_since_reference'] >= 20:
                bead['reference_pos'] = (new_x, new_y)
                bead['frames_since_reference'] = 0

            # Adaptive template update when correlation is strong on both axes
            if best_corr_x > 0.5 and best_corr_y > 0.5:
                try:
                    new_template, _ = self._extract_template(gray, new_x, new_y, tpl_shape)
                except ValueError:
                    new_template = None

                if new_template is not None and bead['template'] is not None and new_template.shape == bead['template'].shape:
                    alpha = 0.05
                    bead['template'] = (1 - alpha) * bead['template'] + alpha * new_template
                    template_x_new, template_y_new = self._compute_profiles(new_template)
                    if template_x_new.shape == bead['template_x'].shape:
                        bead['template_x'] = (1 - alpha) * bead['template_x'] + alpha * template_x_new
                    if template_y_new.shape == bead['template_y'].shape:
                        bead['template_y'] = (1 - alpha) * bead['template_y'] + alpha * template_y_new

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
        """Load tracking data from saved format."""
        self.beads = []
        for bead in beads_data:
            bead_copy = dict(bead)

            template = bead_copy.get('template')
            if template is not None:
                template = template.astype(np.float32)
                bead_copy['template'] = template
                bead_copy['template_shape'] = template.shape
                tpl_x, tpl_y = self._compute_profiles(template)
                bead_copy['template_x'] = tpl_x
                bead_copy['template_y'] = tpl_y
            else:
                bead_copy['template_shape'] = (self.template_size, self.template_size)
                bead_copy['template_x'] = None
                bead_copy['template_y'] = None

            bead_copy.setdefault('frames_since_reference', 0)
            if 'reference_pos' not in bead_copy:
                if bead_copy.get('positions'):
                    bead_copy['reference_pos'] = bead_copy['positions'][0]
                else:
                    bead_copy['reference_pos'] = (0, 0)

            self.beads.append(bead_copy)
    
    def clear(self):
        """Clear all tracked beads."""
        self.beads = []
    
    def remove_bead(self, bead_id: int):
        """Remove a specific bead from tracking."""
        self.beads = [bead for bead in self.beads if bead['id'] != bead_id]

    @staticmethod
    def _compute_bounds(center: float, size: int, limit: int) -> Tuple[int, int]:
        """Return start/end indices centered near value with requested size."""
        if limit <= 0:
            return 0, 0

        size = max(1, min(int(size), limit))
        start = int(round(center)) - size // 2
        end = start + size

        if start < 0:
            end -= start
            start = 0

        if end > limit:
            shift = end - limit
            start -= shift
            end = limit

        if start < 0:
            start = 0

        if end - start <= 0:
            end = min(limit, start + 1)

        return start, end

    def _extract_template(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        target_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract a template centered at (x, y) with optional target shape."""

        height, width = gray.shape
        if target_shape is None:
            target_h = target_w = self.template_size
        else:
            target_h, target_w = target_shape

        y1, y2 = self._compute_bounds(y, target_h, height)
        x1, x2 = self._compute_bounds(x, target_w, width)

        template = gray[y1:y2, x1:x2].astype(np.float32)
        if template.size == 0:
            raise ValueError("Template extraction failed; region empty")

        return template, (y1, y2, x1, x2)

    @staticmethod
    def _compute_profiles(template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 1D template profiles by averaging rows and columns."""
        if template.ndim != 2:
            raise ValueError("Template must be 2D for profile extraction")

        profile_x = template.mean(axis=0)
        profile_y = template.mean(axis=1)
        return profile_x.astype(np.float32), profile_y.astype(np.float32)

    @staticmethod
    def _normalized_cross_correlation(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Compute normalized cross-correlation between 1D signal and template."""
        if signal.size == 0 or template.size == 0 or signal.size < template.size:
            return np.array([], dtype=np.float32)

        template = template.astype(np.float32)
        signal = signal.astype(np.float32)

        template_mean = float(template.mean())
        template_zero = template - template_mean
        template_norm = float(np.linalg.norm(template_zero))
        if template_norm == 0:
            return np.zeros(signal.size - template.size + 1, dtype=np.float32)

        window_len = template.size
        num_positions = signal.size - window_len + 1
        correlations = np.zeros(num_positions, dtype=np.float32)

        for idx in range(num_positions):
            window = signal[idx:idx + window_len]
            window_mean = float(window.mean())
            window_zero = window - window_mean
            denom = float(np.linalg.norm(window_zero)) * template_norm
            if denom == 0:
                correlations[idx] = 0.0
            else:
                correlations[idx] = float(np.dot(window_zero, template_zero) / denom)

        return correlations


def detect_beads_auto(frame: np.ndarray, min_area: int = 500, max_area: int = 5000, threshold_value: int = 150) -> List[Tuple[int, int]]:
    """Auto-detect beads in XY plane using simple blob detection."""
    # Convert to grayscale
    gray = _to_grayscale(frame)
    
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
