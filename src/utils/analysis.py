"""XY traces tracking - used by XY Traces tab only. No analysis performed."""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional


class BeadTracker:
    """Store XY traces for beads - used by XY Traces tab."""
    
    def __init__(self, window_size: int = 40):
        """Initialize bead tracker."""
        self.window_size = window_size
        self.beads: List[Dict[str, Any]] = []
        self.template_size = 30
        
    def add_bead(self, frame: np.ndarray, x: int, y: int, bead_id: int):
        """Add a new bead to track."""
        bead = {
            'id': bead_id,
            'positions': [(x, y)],
            'initial_pos': (x, y),
            'template': None,
            'velocity': (0, 0)
        }
        self.beads.append(bead)
    
    def track_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Track beads using COM + Quadrant Interpolation algorithm.
        Based on Cnossen et al., 2019 and qtrk implementation.
        
        Args:
            frame: Current video frame (grayscale or RGB)
            
        Returns:
            List of (bead_id, x, y) tuples with updated positions
        """
        # Convert to grayscale and float
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        results = []
        
        for bead in self.beads:
            last_x, last_y = bead['positions'][-1]
            
            # Extract ROI around last known position
            roi_size = self.window_size
            x1 = max(0, int(last_x - roi_size))
            y1 = max(0, int(last_y - roi_size))
            x2 = min(gray.shape[1], int(last_x + roi_size))
            y2 = min(gray.shape[0], int(last_y + roi_size))
            
            if x2 <= x1 or y2 <= y1:
                # ROI out of bounds, keep last position
                bead['positions'].append((last_x, last_y))
                results.append((bead['id'], last_x, last_y))
                continue
            
            roi = gray[y1:y2, x1:x2]
            
            # Step 1: Center-of-Mass computation for initial estimate
            com_x, com_y = self._compute_center_of_mass(roi)
            
            # Convert COM to global coordinates
            global_x = x1 + com_x
            global_y = y1 + com_y
            
            # Step 2: Quadrant Interpolation refinement (multiple iterations)
            # Parameters from qtrk: radialSteps ~80, angularSteps starts low and increases
            refined_x, refined_y = self._quadrant_interpolation(
                gray, global_x, global_y, 
                radial_steps=40,      # Number of radial samples
                angular_steps_per_q=32,  # Angular steps per quadrant (start low)
                min_radius=2.0,       # Inner radius
                max_radius=15.0,      # Outer radius  
                iterations=3,         # Number of iterations
                angular_step_factor=1.5  # Increase angular resolution each iteration
            )
            
            # Store updated position
            bead['positions'].append((refined_x, refined_y))
            results.append((bead['id'], refined_x, refined_y))
        
        return results
    
    def _compute_center_of_mass(self, roi: np.ndarray) -> Tuple[float, float]:
        """
        Compute center-of-mass of intensity in ROI.
        
        Args:
            roi: Region of interest image
            
        Returns:
            (x, y) center-of-mass coordinates relative to ROI
        """
        # Subtract background (minimum value)
        roi_bg_subtracted = roi - np.min(roi)
        
        # Compute total intensity
        total = np.sum(roi_bg_subtracted)
        
        if total == 0:
            # No signal, return center of ROI
            return roi.shape[1] / 2.0, roi.shape[0] / 2.0
        
        # Create coordinate grids
        y_coords, x_coords = np.indices(roi.shape)
        
        # Compute weighted average
        com_x = np.sum(x_coords * roi_bg_subtracted) / total
        com_y = np.sum(y_coords * roi_bg_subtracted) / total
        
        return com_x, com_y
    
    def _quadrant_interpolation(self, frame: np.ndarray, x: float, y: float, 
                                radial_steps: int = 40, angular_steps_per_q: int = 32,
                                min_radius: float = 2.0, max_radius: float = 15.0,
                                iterations: int = 3, angular_step_factor: float = 1.5) -> Tuple[float, float]:
        """
        Refine bead position using Quadrant Interpolation algorithm.
        
        Based on qtrk implementation by Cnossen et al.
        
        Args:
            frame: Full frame image
            x, y: Initial position estimate
            radial_steps: Number of radial samples (nr)
            angular_steps_per_q: Angular steps per quadrant (starts low, increases)
            min_radius: Minimum radius for sampling
            max_radius: Maximum radius for sampling  
            iterations: Number of QI refinement iterations
            angular_step_factor: Factor to increase angular steps each iteration
            
        Returns:
            (x, y) refined position
        """
        center_x, center_y = x, y
        nr = radial_steps
        
        # Build trigonometric table for quadrant (0 to 90 degrees)
        trig_table = []
        for j in range(angular_steps_per_q):
            ang = 0.5 * np.pi * (j + 0.5) / angular_steps_per_q
            trig_table.append((np.cos(ang), np.sin(ang)))
        
        pixels_per_prof_len = (max_radius - min_radius) / radial_steps
        angular_steps = angular_steps_per_q / (angular_step_factor ** iterations)
        
        for k in range(iterations):
            angular_steps = int(max(angular_steps, 10))  # Minimum 10 samples
            
            # Compute all 4 quadrants
            q0 = self._compute_quadrant_profile(frame, center_x, center_y, nr, angular_steps, 
                                               0, min_radius, max_radius, trig_table)
            q1 = self._compute_quadrant_profile(frame, center_x, center_y, nr, angular_steps,
                                               1, min_radius, max_radius, trig_table)
            q2 = self._compute_quadrant_profile(frame, center_x, center_y, nr, angular_steps,
                                               2, min_radius, max_radius, trig_table)
            q3 = self._compute_quadrant_profile(frame, center_x, center_y, nr, angular_steps,
                                               3, min_radius, max_radius, trig_table)
            
            if q0 is None or q1 is None or q2 is None or q3 is None:
                # Boundary hit
                break
            
            # Build X profile: Ix = [ qL(-r)  qR(r) ]
            # qL = q1 + q2 (left side, reversed)
            # qR = q0 + q3 (right side)
            x_profile = np.concatenate([
                (q1 + q2)[::-1],  # Left, reversed
                (q0 + q3)          # Right
            ])
            
            # Build Y profile: Iy = [ qB(-r)  qT(r) ]
            # qT = q0 + q1 (top/bottom depends on coordinate system)
            # qB = q2 + q3
            y_profile = np.concatenate([
                (q2 + q3)[::-1],  # Bottom, reversed
                (q0 + q1)          # Top
            ])
            
            # Compute offsets using FFT auto-correlation
            offset_x = self._qi_compute_offset(x_profile, nr)
            offset_y = self._qi_compute_offset(y_profile, nr)
            
            # Update position
            center_x += offset_x * pixels_per_prof_len
            center_y += offset_y * pixels_per_prof_len
            
            # Increase angular resolution for next iteration
            angular_steps *= angular_step_factor
        
        return center_x, center_y
    
    def _compute_quadrant_profile(self, frame: np.ndarray, cx: float, cy: float,
                                  radial_steps: int, angular_steps: int, quadrant: int,
                                  min_radius: float, max_radius: float,
                                  trig_table: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Compute radial profile for one quadrant.
        
        Based on qtrk ComputeQuadrantProfile.
        
        Args:
            frame: Image frame
            cx, cy: Center position
            radial_steps: Number of radial bins
            angular_steps: Number of angular samples
            quadrant: Quadrant index (0-3)
            min_radius: Minimum sampling radius
            max_radius: Maximum sampling radius
            trig_table: Precomputed cos/sin values for 0-90 degrees
            
        Returns:
            Radial profile array, or None if out of bounds
        """
        # Quadrant multipliers: controls which direction to sample
        # q0: (+x, +y), q1: (-x, +y), q2: (-x, -y), q3: (+x, -y)
        qmat = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        mx, my = qmat[quadrant]
        
        profile = np.zeros(radial_steps, dtype=np.float32)
        rstep = (max_radius - min_radius) / radial_steps
        
        # Compute mean for background
        mean = np.mean(frame)
        
        # For each radial bin
        for i in range(radial_steps):
            r = min_radius + rstep * i
            total = 0.0
            count = 0
            
            # Sample along angular direction using precomputed trig table
            angstepf = len(trig_table) / angular_steps
            for a in range(angular_steps):
                j = int(angstepf * a)
                cos_val, sin_val = trig_table[j]
                
                # Calculate sample position
                sample_x = cx + mx * cos_val * r
                sample_y = cy + my * sin_val * r
                
                # Bilinear interpolation
                value = self._bilinear_interpolate(frame, sample_x, sample_y)
                
                if value is not None:
                    total += value
                    count += 1
            
            # Need at least a few samples
            if count >= 3:
                profile[i] = total / count
            else:
                profile[i] = mean
        
        return profile
    
    def _qi_compute_offset(self, profile: np.ndarray, nr: int) -> float:
        """
        Compute offset using FFT-based auto-correlation.
        
        Based on qtrk QI_ComputeOffset implementation.
        
        Args:
            profile: Combined profile of length 2*nr
            nr: Number of radial steps
            
        Returns:
            Offset in units of profile bins
        """
        # Convert to complex for FFT
        prof_complex = profile.astype(np.complex64)
        prof_reverse = prof_complex[::-1]
        
        # Forward FFT of both
        fft_prof = np.fft.fft(prof_complex)
        fft_rev = np.fft.fft(prof_reverse)
        
        # Multiply with conjugate
        fft_prod = fft_prof * np.conj(fft_rev)
        
        # Inverse FFT to get auto-correlation
        autoconv = np.fft.ifft(fft_prod).real
        
        # Shift the autocorrelation (circular shift by nr)
        autoconv_shifted = np.roll(autoconv, nr)
        
        # Find maximum with sub-pixel interpolation
        max_pos = self._compute_max_interp(autoconv_shifted)
        
        # Convert to offset (from qtrk: (maxPos - nr) * (pi/4))
        offset = (max_pos - nr) * (np.pi / 4.0)
        
        return offset
    
    def _compute_max_interp(self, data: np.ndarray) -> float:
        """
        Find maximum position with sub-pixel interpolation using least-squares quadratic fit.
        
        Based on qtrk ComputeMaxInterp.
        
        Args:
            data: 1D array
            
        Returns:
            Sub-pixel position of maximum
        """
        # Find integer maximum
        i_max = np.argmax(data)
        
        # Use 5 points around maximum for quadratic fit (as in QI_LSQFIT_NWEIGHTS)
        num_pts = 5
        start_pos = max(i_max - num_pts // 2, 0)
        end_pos = min(i_max + (num_pts - num_pts // 2), len(data))
        num_points = end_pos - start_pos
        
        if num_points < 3:
            return float(i_max)
        
        # Extract points around maximum
        xs = np.arange(start_pos, end_pos) - i_max
        ys = data[start_pos:end_pos]
        
        # Fit quadratic: y = a*x^2 + b*x + c
        # Using least squares
        try:
            A = np.column_stack([xs**2, xs, np.ones_like(xs)])
            coeffs = np.linalg.lstsq(A, ys, rcond=None)[0]
            a, b, c = coeffs
            
            # Maximum of quadratic is at x = -b/(2a)
            if abs(a) > 1e-9:
                interp_max = -b / (2 * a)
                return float(i_max) + interp_max
            else:
                return float(i_max)
        except:
            return float(i_max)
    
    def _bilinear_interpolate(self, frame: np.ndarray, x: float, y: float) -> Optional[float]:
        """
        Bilinear interpolation at sub-pixel position.
        
        Args:
            frame: Image frame
            x, y: Sub-pixel coordinates
            
        Returns:
            Interpolated intensity value, or None if out of bounds
        """
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1
        
        # Check bounds
        if x0 < 0 or y0 < 0 or x1 >= frame.shape[1] or y1 >= frame.shape[0]:
            return None
        
        # Interpolation weights
        wx = x - x0
        wy = y - y0
        
        # Bilinear interpolation
        value = (1 - wx) * (1 - wy) * frame[y0, x0] + \
                wx * (1 - wy) * frame[y0, x1] + \
                (1 - wx) * wy * frame[y1, x0] + \
                wx * wy * frame[y1, x1]
        
        return float(value)
    
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


def detect_beads_auto(frame: np.ndarray, min_area: int = 50, max_area: int = 5000, threshold_value: int = 150) -> List[Tuple[int, int]]:
    """Auto-detect beads - used by XY Traces tab."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
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
