"""Input/Output operations for tracking data with HDF5 files."""

import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from src.utils.logger import Logger


class TrackingDataIO:
    """Handle saving and loading tracking data to/from HDF5 files in analysed_data folder."""
    
    @staticmethod
    def save_to_hdf5(hdf5_path: str, beads_data: List[Dict[str, Any]], 
                     metadata: Optional[Dict[str, Any]] = None, hdf5_file_handle: Optional[h5py.File] = None):
        """
        Save tracking data to HDF5 file under /analysed_data/xy_tracking.
        
        Args:
            hdf5_path: Path to HDF5 file (used if hdf5_file_handle is None)
            beads_data: List of bead dictionaries from BeadTracker
            metadata: Optional metadata to save
            hdf5_file_handle: Optional already-open HDF5 file handle
        """
        if not beads_data:
            Logger.debug("No bead data to save", "TRACKING_IO")
            return
        
        try:
            Logger.debug(f"Saving to HDF5: {hdf5_path}", "TRACKING_IO")
            
            # Use provided file handle or open new one
            if hdf5_file_handle is not None:
                f = hdf5_file_handle
                should_close = False
            else:
                f = h5py.File(hdf5_path, 'a')
                should_close = True
            
            try:
                # Create analysed_data group if it doesn't exist
                if 'analysed_data' not in f:
                    analysis_group = f.create_group('analysed_data')
                else:
                    analysis_group = f['analysed_data']  # type: ignore
                
                # Create or overwrite xy_tracking group
                if 'xy_tracking' in analysis_group:  # type: ignore
                    del analysis_group['xy_tracking']  # type: ignore
                
                xy_group = analysis_group.create_group('xy_tracking')  # type: ignore
                
                # Prepare tracking data
                max_frames = max(len(bead['positions']) for bead in beads_data)
                num_beads = len(beads_data)
                
                # Create dataset: shape (num_frames, num_beads, 2) for (x, y) positions
                tracking_data = np.full((max_frames, num_beads, 2), -1, dtype=np.int32)
                
                for bead_idx, bead in enumerate(beads_data):
                    positions = bead['positions']
                    for frame_idx, (x, y) in enumerate(positions):
                        tracking_data[frame_idx, bead_idx, 0] = x
                        tracking_data[frame_idx, bead_idx, 1] = y
                
                # Save positions dataset
                positions_dataset = xy_group.create_dataset('positions', data=tracking_data)
                
                # Save metadata as group attributes
                xy_group.attrs['num_beads'] = num_beads
                xy_group.attrs['num_frames'] = max_frames
                xy_group.attrs['description'] = 'XY bead tracking data'
                
                if metadata:
                    for key, value in metadata.items():
                        if value is not None:
                            xy_group.attrs[key] = value
                
                # Save individual bead metadata
                for bead_idx, bead in enumerate(beads_data):
                    bead_id = bead['id']
                    initial_x, initial_y = bead['positions'][0] if bead['positions'] else (0, 0)
                    
                    xy_group.attrs[f'bead_{bead_idx}_id'] = bead_id
                    xy_group.attrs[f'bead_{bead_idx}_initial_x'] = initial_x
                    xy_group.attrs[f'bead_{bead_idx}_initial_y'] = initial_y
                    
                    # Save template if available
                    if 'template' in bead and bead['template'] is not None:
                        template_name = f'bead_{bead_idx}_template'
                        xy_group.create_dataset(template_name, data=bead['template'])
                
                Logger.success(f"Saved {num_beads} beads, {max_frames} frames to /analysed_data/xy_tracking/", "TRACKING_IO")
                
            finally:
                # Only close if we opened it ourselves
                if should_close:
                    f.close()
                
        except Exception as e:
            Logger.error(f"Error saving tracking data: {e}", "TRACKING_IO")
            import traceback
            traceback.print_exc()
            raise
    
    @staticmethod
    def load_from_hdf5(hdf5_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load tracking data from HDF5 file from /analysed_data/xy_tracking.
        
        Args:
            hdf5_path: Path to HDF5 file
        
        Returns:
            Tuple of (beads_data, metadata)
        """
        beads_data = []
        metadata = {}
        
        try:
            Logger.debug(f"Loading from HDF5: {hdf5_path}", "TRACKING_IO")
            with h5py.File(hdf5_path, 'r') as f:
                if 'analysed_data' not in f or 'xy_tracking' not in f['analysed_data']:  # type: ignore
                    return beads_data, metadata
                
                xy_group = f['analysed_data']['xy_tracking']  # type: ignore
                
                # Check if it's a group (new format) or dataset (old format)
                if isinstance(xy_group, h5py.Group):
                    # New format
                    if 'positions' not in xy_group:
                        return beads_data, metadata
                    
                    tracking_data = xy_group['positions'][:]  # type: ignore
                    
                    # Load metadata from group attributes
                    for key, value in xy_group.attrs.items():
                        metadata[key] = value
                    
                    num_beads = int(metadata.get('num_beads', tracking_data.shape[1]))  # type: ignore
                    
                    # Reconstruct bead dictionaries
                    for bead_idx in range(num_beads):
                        positions = []
                        for frame_idx in range(tracking_data.shape[0]):  # type: ignore
                            x, y = tracking_data[frame_idx, bead_idx, :]  # type: ignore
                            if x >= 0 and y >= 0:  # type: ignore
                                positions.append((int(x), int(y)))  # type: ignore
                            else:
                                break
                        
                        if not positions:
                            continue
                        
                        bead_id = int(metadata.get(f'bead_{bead_idx}_id', bead_idx))
                        initial_x = int(metadata.get(f'bead_{bead_idx}_initial_x', positions[0][0]))
                        initial_y = int(metadata.get(f'bead_{bead_idx}_initial_y', positions[0][1]))
                        
                        template = None
                        template_name = f'bead_{bead_idx}_template'
                        if template_name in xy_group:
                            template = xy_group[template_name][:]  # type: ignore
                        
                        bead = {
                            'id': bead_id,
                            'positions': positions,
                            'initial_pos': (initial_x, initial_y),
                            'template': template
                        }
                        
                        beads_data.append(bead)
                else:
                    # Old format: backward compatibility
                    tracking_data = xy_group[:]  # type: ignore
                    
                    for key, value in xy_group.attrs.items():
                        metadata[key] = value
                    
                    num_beads = int(metadata.get('num_beads', tracking_data.shape[1]))  # type: ignore
                    
                    for bead_idx in range(num_beads):
                        positions = []
                        for frame_idx in range(tracking_data.shape[0]):  # type: ignore
                            x, y = tracking_data[frame_idx, bead_idx, :]  # type: ignore
                            if x >= 0 and y >= 0:  # type: ignore
                                positions.append((int(x), int(y)))  # type: ignore
                            else:
                                break
                        
                        if not positions:
                            continue
                        
                        bead_id = int(metadata.get(f'bead_{bead_idx}_id', bead_idx))
                        initial_x = int(metadata.get(f'bead_{bead_idx}_initial_x', positions[0][0]))
                        initial_y = int(metadata.get(f'bead_{bead_idx}_initial_y', positions[0][1]))
                        
                        template = None
                        template_name = f'xy_tracking_bead_{bead_idx}_template'
                        if template_name in f['analysed_data']:  # type: ignore
                            template = f['analysed_data'][template_name][:]  # type: ignore
                        
                        bead = {
                            'id': bead_id,
                            'positions': positions,
                            'initial_pos': (initial_x, initial_y),
                            'template': template
                        }
                        
                        beads_data.append(bead)
                
                Logger.success(f"Loaded {len(beads_data)} beads", "TRACKING_IO")
                
        except Exception as e:
            Logger.error(f"Error loading tracking data: {e}", "TRACKING_IO")
            import traceback
            traceback.print_exc()
        
        return beads_data, metadata
    
    @staticmethod
    def has_tracking_data(hdf5_path: str) -> bool:
        """
        Check if HDF5 file contains tracking data in /analysed_data/xy_tracking.
        
        Args:
            hdf5_path: Path to HDF5 file
        
        Returns:
            True if tracking data exists
        """
        try:
            with h5py.File(hdf5_path, 'r') as f:
                has_data = 'analysed_data' in f and 'xy_tracking' in f['analysed_data']  # type: ignore
                if has_data:
                    Logger.debug("Found existing tracking data", "TRACKING_IO")
                return has_data
        except Exception as e:
            Logger.error(f"Error checking tracking data: {e}", "TRACKING_IO")
            return False
    
    @staticmethod
    def export_to_csv(hdf5_path: str, output_csv_path: Optional[str] = None) -> str:
        """
        Export tracking data to CSV file.
        
        Args:
            hdf5_path: Path to HDF5 file with tracking data
            output_csv_path: Optional output path, defaults to same name as HDF5
        
        Returns:
            Path to created CSV file
        """
        if output_csv_path is None:
            output_csv_path = str(Path(hdf5_path).with_suffix('.csv'))
        
        Logger.debug("Loading data from HDF5 for export", "TRACKING_IO")
        beads_data, metadata = TrackingDataIO.load_from_hdf5(hdf5_path)
        
        if not beads_data:
            raise ValueError("No tracking data available to export")
        
        Logger.debug(f"Exporting to CSV", "TRACKING_IO")
        # Create CSV content
        import csv
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header without time column
            header = ['frame']
            for bead in beads_data:
                bead_id = bead['id']
                header.extend([f'bead_{bead_id}_x', f'bead_{bead_id}_y'])
            writer.writerow(header)
            
            # Find max number of frames
            max_frames = max(len(bead['positions']) for bead in beads_data)
            
            # Write data row by row
            for frame_idx in range(max_frames):
                row: List[Any] = [frame_idx]
                for bead in beads_data:
                    if frame_idx < len(bead['positions']):
                        x, y = bead['positions'][frame_idx]
                        row.extend([x, y])
                    else:
                        row.extend(['', ''])  # Empty if this bead doesn't have this frame
                writer.writerow(row)
        
        Logger.success(f"Exported {len(beads_data)} beads to {output_csv_path}", "TRACKING_IO")
        return output_csv_path
