"""Analysis modules for AFS data.

This package contains specialized analysis modules for different aspects
of Atomic Force Spectroscopy measurements:

- xy_tracking_xcorr: XY-plane bead tracking using cross-correlation
- xy_tracking_simple: Simple, robust XY-plane bead tracking using COM
- xy_tracking_gaussian: Advanced tracking with 2D Gaussian PSF fitting
- xy_tracking: Legacy tracking implementation
- (future) z_tracking: Z-axis position tracking
- (future) force_analysis: Force curve analysis
- (future) stiffness: Stiffness calculations
- (future) thermal: Thermal calibration and analysis
"""

# Import main classes for convenience - using cross-correlation
from src.analysis.xy_tracking_xcorr import BeadTracker, detect_beads_auto

__all__ = ['BeadTracker', 'detect_beads_auto']
