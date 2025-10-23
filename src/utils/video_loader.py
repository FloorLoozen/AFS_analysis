"""Video loading and management logic - HDF5 only."""

import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any


class HDF5VideoSource:
    """Video source from HDF5 files (AFS_acquisition format)."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.hdf5_file = None
        self.video_data = None
        self.total_frames = 0
        self.fps = 30.0
        self.metadata = {}
        self._load(file_path)
    
    def _load(self, file_path: str):
        """Load HDF5 video file."""
        try:
            self.hdf5_file = h5py.File(file_path, 'r')
            
            # Try to find video data in common locations
            video_paths = ['data/main_video', 'main_video', 'video']
            
            for path in video_paths:
                if path in self.hdf5_file:
                    self.video_data = self.hdf5_file[path]
                    break
            
            # If not found, search for any 4D dataset
            if self.video_data is None:
                for key in self.hdf5_file.keys():
                    if isinstance(self.hdf5_file[key], h5py.Dataset):
                        if len(self.hdf5_file[key].shape) == 4:
                            self.video_data = self.hdf5_file[key]
                            break
            
            if self.video_data is None:
                raise ValueError("No video data found in HDF5 file")
            
            self.total_frames = self.video_data.shape[0]
            
            # Prefer actual_fps over target fps
            if 'actual_fps' in self.video_data.attrs:
                self.fps = float(self.video_data.attrs['actual_fps'])
            elif 'fps' in self.video_data.attrs:
                self.fps = float(self.video_data.attrs['fps'])
            else:
                self.fps = 30.0
            
            self.metadata = dict(self.video_data.attrs)
            
        except Exception as e:
            if self.hdf5_file:
                self.hdf5_file.close()
            raise RuntimeError(f"Failed to load HDF5 video: {e}")
    
    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get frame at specified index."""
        if self.video_data is None or not (0 <= frame_index < self.total_frames):
            return None
        
        try:
            frame = self.video_data[frame_index]
            
            # Convert BGR to RGB if needed
            if 'color_format' in self.metadata and self.metadata['color_format'] == 'BGR':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame
        except Exception as e:
            print(f"Error reading frame {frame_index}: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.hdf5_file:
            try:
                self.hdf5_file.close()
            except:
                pass
            self.hdf5_file = None
        self.video_data = None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get video metadata including file info, video properties, and HDF5 attributes."""
        metadata = {
            'file_path': self.file_path,
            'total_frames': self.total_frames,
            'actual_fps': self.fps,
            'dtype': str(self.video_data.dtype) if self.video_data is not None else None,
            'frame_shape': list(self.video_data.shape[1:]) if self.video_data is not None else None,
        }
        
        # Add compression info if available
        if self.video_data is not None:
            if hasattr(self.video_data, 'compression'):
                metadata['compression'] = self.video_data.compression
            if hasattr(self.video_data, 'compression_opts'):
                metadata['compression_opts'] = self.video_data.compression_opts
        
        # Add video dataset attributes
        metadata.update(self.metadata)
        
        # Add data group attributes if available
        if self.hdf5_file is not None:
            try:
                if 'data' in self.hdf5_file:
                    data_group = self.hdf5_file['data']
                    for key in data_group.attrs.keys():
                        if key not in metadata:
                            value = data_group.attrs[key]
                            # Decode bytes to string
                            if isinstance(value, bytes):
                                value = value.decode('utf-8')
                            metadata[key] = value
                
                # Add root level attributes
                for key in self.hdf5_file.attrs.keys():
                    if key not in metadata:
                        value = self.hdf5_file.attrs[key]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        metadata[key] = value
            except Exception as e:
                print(f"Error reading HDF5 attributes: {e}")
        
        return metadata


class VideoLoader:
    """Load HDF5 video files."""
    
    @staticmethod
    def load(file_path: str) -> HDF5VideoSource:
        """Load HDF5 video from file path."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in ['.hdf5', '.h5']:
            raise ValueError(f"Only HDF5 files (.hdf5, .h5) are supported")
        
        return HDF5VideoSource(str(path))
