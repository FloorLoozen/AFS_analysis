"""Qt wrapper for VideoControllerCore - provides Qt signals and timer integration."""

from typing import Optional
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import numpy as np

from src.utils.video_controller_core import VideoControllerCore
from src.utils.video_loader import HDF5VideoSource


class VideoController(QObject):
    """
    Qt-enabled video controller wrapper.
    
    Wraps VideoControllerCore and provides Qt signals/slots for UI integration.
    All core logic is in the pure Python VideoControllerCore class.
    """
    
    # Qt signals for UI updates
    frame_changed = pyqtSignal(int, np.ndarray)  # frame_index, frame_data
    playback_state_changed = pyqtSignal(bool)  # is_playing
    video_loaded = pyqtSignal()
    playback_finished = pyqtSignal()
    
    def __init__(self):
        """Initialize Qt video controller wrapper."""
        super().__init__()
        
        # Core controller (pure Python, no Qt)
        self._core = VideoControllerCore()
        
        # Set up callbacks to emit Qt signals
        self._core.set_frame_changed_callback(self._on_frame_changed)
        self._core.set_playback_state_changed_callback(self._on_playback_state_changed)
        self._core.set_video_loaded_callback(self._on_video_loaded)
        self._core.set_playback_finished_callback(self._on_playback_finished)
        
        # Timer for playback
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)
    
    def _on_frame_changed(self, frame_index: int, frame_data: np.ndarray) -> None:
        """Core callback: emit Qt signal for frame change."""
        self.frame_changed.emit(frame_index, frame_data)
    
    def _on_playback_state_changed(self, is_playing: bool) -> None:
        """Core callback: emit Qt signal for playback state change."""
        self.playback_state_changed.emit(is_playing)
    
    def _on_video_loaded(self) -> None:
        """Core callback: emit Qt signal for video loaded."""
        self.video_loaded.emit()
    
    def _on_playback_finished(self) -> None:
        """Core callback: emit Qt signal for playback finished."""
        self.playback_finished.emit()
    
    # Delegate properties to core
    @property
    def video_source(self) -> Optional[HDF5VideoSource]:
        """Get video source from core."""
        return self._core.video_source
    
    @property
    def is_playing(self) -> bool:
        """Get playback state from core."""
        return self._core.is_playing
    
    @property
    def current_frame_index(self) -> int:
        """Get current frame index from core."""
        return self._core.current_frame_index
    
    # Delegate methods to core
    def load_video(self, video_source: HDF5VideoSource) -> None:
        """Load a video source."""
        self._core.load_video(video_source)
    
    def get_metadata(self):
        """Get current video metadata."""
        return self._core.get_metadata()
    
    def get_total_frames(self) -> int:
        """Get total number of frames."""
        return self._core.get_total_frames()
    
    def get_fps(self) -> float:
        """Get video FPS."""
        return self._core.get_fps()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame data."""
        return self._core.get_current_frame()
    
    def seek_to_frame(self, frame_index: int) -> None:
        """Seek to specific frame."""
        self._core.seek_to_frame(frame_index)
    
    def play(self) -> None:
        """Start playback."""
        if not self._core.video_source or self._core.is_playing:
            return
        
        self._core.set_playing(True)
        
        # Calculate timer interval from FPS
        interval = int(1000 / self._core.get_fps())
        self.playback_timer.start(interval)
    
    def pause(self) -> None:
        """Pause playback."""
        if not self._core.is_playing:
            return
        
        self._core.set_playing(False)
        self.playback_timer.stop()
    
    def stop(self) -> None:
        """Stop playback and return to beginning."""
        self.pause()
        self.seek_to_frame(0)
    
    def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        if self._core.is_playing:
            self.pause()
        else:
            self.play()
    
    def _advance_frame(self) -> None:
        """Advance to next frame (called by timer)."""
        success = self._core.advance_frame()
        if not success:
            # Reached end - stop timer
            self.playback_timer.stop()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.pause()
        self._core.cleanup()
