"""Pure Python video playback controller core - no UI dependencies."""

from typing import Optional, Callable, Dict, Any, List
import numpy as np

from src.utils.video_loader import HDF5VideoSource


class VideoControllerCore:
    """Pure Python video controller with callback pattern (no Qt/UI dependencies)."""
    
    def __init__(self):
        """Initialize video controller core."""
        self.video_source: Optional[HDF5VideoSource] = None
        self.is_playing = False
        self.current_frame_index = 0
        
        # Frame cache for better performance (LRU cache)
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._cache_size = 50
        self._cache_order: List[int] = []
        
        # Callbacks for events (observer pattern)
        self._on_frame_changed: Optional[Callable[[int, np.ndarray], None]] = None
        self._on_playback_state_changed: Optional[Callable[[bool], None]] = None
        self._on_video_loaded: Optional[Callable[[], None]] = None
        self._on_playback_finished: Optional[Callable[[], None]] = None
    
    def set_frame_changed_callback(self, callback: Callable[[int, np.ndarray], None]) -> None:
        """Set callback for frame changes."""
        self._on_frame_changed = callback
    
    def set_playback_state_changed_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback for playback state changes."""
        self._on_playback_state_changed = callback
    
    def set_video_loaded_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for video loaded event."""
        self._on_video_loaded = callback
    
    def set_playback_finished_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for playback finished event."""
        self._on_playback_finished = callback
    
    def load_video(self, video_source: HDF5VideoSource) -> None:
        """Load a video source."""
        # Clean up previous video
        self.cleanup()
        
        # Clear cache
        self._frame_cache.clear()
        self._cache_order.clear()
        
        # Set new video source
        self.video_source = video_source
        self.current_frame_index = 0
        
        # Load first frame
        self.seek_to_frame(0)
        
        # Notify callback
        if self._on_video_loaded:
            self._on_video_loaded()
    
    def get_metadata(self) -> Dict[str, Any]:
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
    
    def _get_frame_cached(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get frame with LRU caching for better performance.
        
        Args:
            frame_index: Frame index to retrieve
        
        Returns:
            Frame data or None
        """
        # Check cache first
        if frame_index in self._frame_cache:
            # Move to end of cache order (most recently used)
            self._cache_order.remove(frame_index)
            self._cache_order.append(frame_index)
            return self._frame_cache[frame_index].copy()
        
        # Load from source
        if not self.video_source:
            return None
        
        frame = self.video_source.get_frame(frame_index)
        
        if frame is not None:
            # Evict oldest if cache full
            if len(self._cache_order) >= self._cache_size:
                oldest = self._cache_order.pop(0)
                if oldest in self._frame_cache:
                    del self._frame_cache[oldest]
            
            # Add to cache
            self._frame_cache[frame_index] = frame.copy()
            self._cache_order.append(frame_index)
        
        return frame
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame data."""
        if self.video_source:
            return self._get_frame_cached(self.current_frame_index)
        return None
    
    def seek_to_frame(self, frame_index: int) -> None:
        """Seek to specific frame."""
        if not self.video_source:
            return
        
        # Clamp frame index
        frame_index = max(0, min(frame_index, self.video_source.total_frames - 1))
        
        # Get frame using cache
        frame = self._get_frame_cached(frame_index)
        if frame is not None:
            self.current_frame_index = frame_index
            # Notify callback
            if self._on_frame_changed:
                self._on_frame_changed(frame_index, frame)
    
    def advance_frame(self) -> bool:
        """
        Advance to next frame.
        
        Returns:
            True if advanced, False if at end
        """
        if not self.video_source:
            return False
        
        next_frame = self.current_frame_index + 1
        
        if next_frame >= self.video_source.total_frames:
            # Reached end of video
            self.set_playing(False)
            if self._on_playback_finished:
                self._on_playback_finished()
            return False
        
        self.seek_to_frame(next_frame)
        return True
    
    def set_playing(self, playing: bool) -> None:
        """
        Set playback state.
        
        Args:
            playing: True to play, False to pause
        """
        if self.is_playing != playing:
            self.is_playing = playing
            if self._on_playback_state_changed:
                self._on_playback_state_changed(playing)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.set_playing(False)
        
        # Clear cache
        self._frame_cache.clear()
        self._cache_order.clear()
        
        if self.video_source:
            self.video_source.cleanup()
            self.video_source = None
