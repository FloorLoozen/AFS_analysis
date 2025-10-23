"""Input/Output operations for tracking data with HDF5 files."""

import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class TrackingDataIO:
    """Handle saving and loading tracking data to/from HDF5 files in analysis folder."""
    
    @staticmethod
    def save_to_hdf5(hdf5_path: str, beads_data: List[Dict[str, Any]], 
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Save tracking data to HDF5 file under /analysis/xy_tracking.
        
        Args:
            hdf5_path: Path to HDF5 file
            beads_data: List of bead dictionaries from BeadTracker
            metadata: Optional metadata to save
        """
        if not beads_data:
            print("⚠ No bead data to save")
            return
        
        try:
            print(f"[SAVE] Opening HDF5: {hdf5_path}")
            with h5py.File(hdf5_path, 'a') as f:
                # Create analysis group if it doesn't exist
                if 'analysis' not in f:
                    print("[SAVE] Creating /analysis group")
                    analysis_group = f.create_group('analysis')
                else:
                    print("[SAVE] Using existing /analysis group")
                    analysis_group = f['analysis']
                
                # Create or overwrite xy_tracking dataset
                if 'xy_tracking' in analysis_group:
                    print("[SAVE] Deleting existing xy_tracking dataset")
                    del analysis_group['xy_tracking']
                
                # Prepare tracking data as structured array
                max_frames = max(len(bead['positions']) for bead in beads_data)
                num_beads = len(beads_data)
                print(f"[SAVE] Preparing data: {num_beads} beads, {max_frames} frames")
                
                # Create dataset: shape (num_frames, num_beads, 2) for (x, y) positions
                tracking_data = np.full((max_frames, num_beads, 2), -1, dtype=np.int32)
                
                for bead_idx, bead in enumerate(beads_data):
                    positions = bead['positions']
                    for frame_idx, (x, y) in enumerate(positions):
                        tracking_data[frame_idx, bead_idx, 0] = x
                        tracking_data[frame_idx, bead_idx, 1] = y
                
                # Save as dataset
                print("[SAVE] Creating xy_tracking dataset")
                tracking_dataset = analysis_group.create_dataset('xy_tracking', data=tracking_data)
                
                # Save metadata as attributes
                tracking_dataset.attrs['num_beads'] = num_beads
                tracking_dataset.attrs['num_frames'] = max_frames
                tracking_dataset.attrs['description'] = 'XY bead tracking positions (frame, bead, xy)'
                
                if metadata:
                    for key, value in metadata.items():
                        if value is not None:
                            tracking_dataset.attrs[key] = value
                
                # Save individual bead metadata and templates
                for bead_idx, bead in enumerate(beads_data):
                    bead_id = bead['id']
                    initial_x, initial_y = bead['positions'][0] if bead['positions'] else (0, 0)
                    
                    tracking_dataset.attrs[f'bead_{bead_idx}_id'] = bead_id
                    tracking_dataset.attrs[f'bead_{bead_idx}_initial_x'] = initial_x
                    tracking_dataset.attrs[f'bead_{bead_idx}_initial_y'] = initial_y
                    
                    # Save template if available
                    if 'template' in bead and bead['template'] is not None:
                        template_name = f'xy_tracking_bead_{bead_idx}_template'
                        if template_name in analysis_group:
                            del analysis_group[template_name]
                        analysis_group.create_dataset(template_name, data=bead['template'])
                
                print(f"✓ Saved to /analysis/xy_tracking: {num_beads} beads, {max_frames} frames")
                
        except Exception as e:
            print(f"✗ Error saving tracking data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @staticmethod
    def load_from_hdf5(hdf5_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load tracking data from HDF5 file from /analysis/xy_tracking.
        
        Args:
            hdf5_path: Path to HDF5 file
        
        Returns:
            Tuple of (beads_data, metadata)
        """
        beads_data = []
        metadata = {}
        
        try:
            print(f"[LOAD] Opening HDF5: {hdf5_path}")
            with h5py.File(hdf5_path, 'r') as f:
                if 'analysis' not in f:
                    print("[LOAD] No /analysis group found")
                    return beads_data, metadata
                
                if 'xy_tracking' not in f['analysis']:
                    print("[LOAD] No /analysis/xy_tracking dataset found")
                    return beads_data, metadata
                
                print("[LOAD] Found /analysis/xy_tracking")
                tracking_dataset = f['analysis']['xy_tracking']
                tracking_data = tracking_dataset[:]  # Shape: (num_frames, num_beads, 2)
                print(f"[LOAD] Dataset shape: {tracking_data.shape}")
                
                # Load metadata from attributes
                for key, value in tracking_dataset.attrs.items():
                    metadata[key] = value
                
                num_beads = int(metadata.get('num_beads', tracking_data.shape[1]))
                print(f"[LOAD] Loading {num_beads} beads")
                
                # Reconstruct bead dictionaries
                for bead_idx in range(num_beads):
                    # Extract positions for this bead
                    positions = []
                    for frame_idx in range(tracking_data.shape[0]):
                        x, y = tracking_data[frame_idx, bead_idx, :]
                        if x >= 0 and y >= 0:  # Valid position (not -1 fill value)
                            positions.append((int(x), int(y)))
                        else:
                            break  # No more valid frames for this bead
                    
                    if not positions:
                        print(f"[LOAD] Bead {bead_idx} has no valid positions, skipping")
                        continue
                    
                    # Get bead metadata
                    bead_id = int(metadata.get(f'bead_{bead_idx}_id', bead_idx))
                    initial_x = int(metadata.get(f'bead_{bead_idx}_initial_x', positions[0][0]))
                    initial_y = int(metadata.get(f'bead_{bead_idx}_initial_y', positions[0][1]))
                    
                    # Load template if available
                    template = None
                    template_name = f'xy_tracking_bead_{bead_idx}_template'
                    if template_name in f['analysis']:
                        template = f['analysis'][template_name][:]
                    
                    bead = {
                        'id': bead_id,
                        'positions': positions,
                        'initial_pos': (initial_x, initial_y),
                        'template': template
                    }
                    
                    beads_data.append(bead)
                
                print(f"✓ Loaded {len(beads_data)} beads from /analysis/xy_tracking")
                
        except Exception as e:
            print(f"✗ Error loading tracking data: {e}")
            import traceback
            traceback.print_exc()
        
        return beads_data, metadata
    
    @staticmethod
    def has_tracking_data(hdf5_path: str) -> bool:
        """
        Check if HDF5 file contains tracking data in /analysis/xy_tracking.
        
        Args:
            hdf5_path: Path to HDF5 file
        
        Returns:
            True if tracking data exists
        """
        try:
            with h5py.File(hdf5_path, 'r') as f:
                has_data = 'analysis' in f and 'xy_tracking' in f['analysis']
                print(f"[CHECK] Has tracking data: {has_data}")
                return has_data
        except Exception as e:
            print(f"[CHECK] Error checking tracking data: {e}")
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
        
        print(f"[EXPORT] Loading data from HDF5")
        beads_data, metadata = TrackingDataIO.load_from_hdf5(hdf5_path)
        
        if not beads_data:
            raise ValueError("No tracking data available to export")
        
        print(f"[EXPORT] Creating CSV: {output_csv_path}")
        # Create CSV content
        import csv
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
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
        
        print(f"✓ Exported {len(beads_data)} beads to CSV")
        return output_csv_path
