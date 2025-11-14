"""
DEPRECATED: This module is kept for backward compatibility.
Please import VideoController from src.ui.video_controller_qt instead.

The video controller has been refactored for better modularity:
- Core logic (pure Python): src.utils.video_controller_core.VideoControllerCore
- Qt wrapper: src.ui.video_controller_qt.VideoController

This file now just re-exports the Qt wrapper for backward compatibility.
"""

# Re-export VideoController from UI layer for backward compatibility
from src.ui.video_controller_qt import VideoController

__all__ = ['VideoController']
