"""Video playback controller."""

from typing import Optional
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import numpy as np

from src.utils.video_loader import HDF5VideoSource


class VideoController(QObject):
    """Controls video playback logic independently of UI."""
    
    # Signals for UI updates
    frame_changed = pyqtSignal(int, np.ndarray)  # frame_index, frame_data
    playback_state_changed = pyqtSignal(bool)  # is_playing
    video_loaded = pyqtSignal()
    playback_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.video_source: Optional[HDF5VideoSource] = None
        self.is_playing = False
        self.current_frame_index = 0
        
        # Timer for playback
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)
    
    def load_video(self, video_source: HDF5VideoSource):
        """Load a video source."""
        # Clean up previous video
        self.cleanup()
        
        # Set new video source
        self.video_source = video_source
        self.current_frame_index = 0
        
        # Load first frame
        self.seek_to_frame(0)
        
        # Emit signal
        self.video_loaded.emit()
    
    def get_metadata(self):
        """Get current video metadata."""
        if self.video_source:
            return self.video_source.get_metadata()
        return {}
    
    def get_total_frames(self) -> int:
        """Get total number of frames."""
        return self.video_source.total_frames if self.video_source else 0
    
    def get_fps(self) -> float:
        """Get video FPS."""
        return self.video_source.fps if self.video_source else 30.0
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame data."""
        if self.video_source:
            return self.video_source.get_frame(self.current_frame_index)
        return None
    
    def seek_to_frame(self, frame_index: int):
        """Seek to specific frame."""
        if not self.video_source:
            return
        
        # Clamp frame index
        frame_index = max(0, min(frame_index, self.video_source.total_frames - 1))
        
        # Get frame
        frame = self.video_source.get_frame(frame_index)
        if frame is not None:
            self.current_frame_index = frame_index
            self.frame_changed.emit(frame_index, frame)
    
    def play(self):
        """Start playback."""
        if not self.video_source or self.is_playing:
            return
        
        self.is_playing = True
        
        # Calculate timer interval from FPS
        interval = int(1000 / self.video_source.fps)
        self.playback_timer.start(interval)
        
        self.playback_state_changed.emit(True)
    
    def pause(self):
        """Pause playback."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        self.playback_timer.stop()
        self.playback_state_changed.emit(False)
    
    def stop(self):
        """Stop playback and return to beginning."""
        self.pause()
        self.seek_to_frame(0)
    
    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def _advance_frame(self):
        """Advance to next frame (called by timer)."""
        if not self.video_source:
            return
        
        next_frame = self.current_frame_index + 1
        
        if next_frame >= self.video_source.total_frames:
            # Reached end of video
            self.pause()
            self.playback_finished.emit()
            return
        
        self.seek_to_frame(next_frame)
    
    def cleanup(self):
        """Clean up resources."""
        self.pause()
        
        if self.video_source:
            self.video_source.cleanup()
            self.video_source = None
