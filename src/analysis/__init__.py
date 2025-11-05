"""Analysis modules for AFS data.

This package contains specialized analysis modules for different aspects
of Atomic Force Spectroscopy measurements:

- xy_tracking: XY-plane bead tracking using Cnossen methodology
- (future) z_tracking: Z-axis position tracking
- (future) force_analysis: Force curve analysis
- (future) stiffness: Stiffness calculations
- (future) thermal: Thermal calibration and analysis
"""

# Import main classes for convenience
from src.analysis.xy_tracking import BeadTracker, detect_beads_auto

__all__ = ['BeadTracker', 'detect_beads_auto']
