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
                     metadata: Optional[Dict[str, Any]] = None, hdf5_file_handle: Optional[h5py.File] = None,
                     save_compact: bool = True, drop_templates: bool = True):
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
        
        f = None
        should_close = False
        
        try:
            Logger.debug(f"Saving to HDF5: {hdf5_path}", "TRACKING_IO")
            
            # Always open our own file handle for writing
            # Don't reuse read-only handles from video loader
            f = h5py.File(hdf5_path, 'a')
            should_close = True
            
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

            # Detect whether positions include z (3) or only x,y (2)
            coord_dims = 2
            for bead in beads_data:
                pos = bead.get('positions')
                if pos and len(pos) > 0:
                    first = pos[0]
                    if isinstance(first, (list, tuple)) and len(first) == 3:
                        coord_dims = 3
                        break

            # Build positions array (num_frames, num_beads, coord_dims)
            tracking_data = np.full((max_frames, num_beads, coord_dims), -1, dtype=np.int32)
            templates_list = []
            # optional per-frame per-bead fields
            forces_list = []
            rms_list = []
            symmetry_list = []
            amplitude_list = []
            stuck_list = []
            frequency_list = []

            for bead_idx, bead in enumerate(beads_data):
                positions = bead['positions']
                for frame_idx, coords in enumerate(positions):
                    # coords may be (x,y) or (x,y,z)
                    try:
                        if isinstance(coords, (list, tuple)):
                            if len(coords) >= 2:
                                tracking_data[frame_idx, bead_idx, 0] = int(coords[0])
                                tracking_data[frame_idx, bead_idx, 1] = int(coords[1])
                            if coord_dims == 3:
                                if len(coords) >= 3:
                                    tracking_data[frame_idx, bead_idx, 2] = int(coords[2])
                        else:
                            # try to unpack
                            x, y = coords
                            tracking_data[frame_idx, bead_idx, 0] = int(x)
                            tracking_data[frame_idx, bead_idx, 1] = int(y)
                    except Exception:
                        # ignore malformed entries
                        pass

                # collect templates if present
                tpl = bead.get('template')
                if tpl is not None:
                    templates_list.append(tpl)
                else:
                    templates_list.append(None)
                # collect optional time-series if present (expect list of per-frame values)
                fseries = bead.get('forces')
                rms_series = bead.get('rms')
                sym_series = bead.get('symmetry')
                amp_series = bead.get('amplitude')
                stuck_series = bead.get('stuck')
                freq_series = bead.get('frequency')

                forces_list.append(fseries)
                rms_list.append(rms_series)
                symmetry_list.append(sym_series)
                amplitude_list.append(amp_series)
                stuck_list.append(stuck_series)
                frequency_list.append(freq_series)

            # Save positions and templates in compact format if requested
            if save_compact:
                # choose safe integer dtype for positions
                mi = int(tracking_data[tracking_data >= 0].min()) if np.any(tracking_data >= 0) else 0
                ma = int(tracking_data.max()) if np.any(tracking_data >= 0) else 0
                if mi >= 0 and ma <= 65535:
                    pos_dtype = np.uint16
                elif mi >= -32768 and ma <= 32767:
                    pos_dtype = np.int16
                else:
                    pos_dtype = tracking_data.dtype

                dset_pos = xy_group.create_dataset('positions', data=tracking_data.astype(pos_dtype),
                                                   dtype=pos_dtype, chunks=(1, num_beads, coord_dims), compression=None)

                # Save metadata as group attributes
                xy_group.attrs['num_beads'] = num_beads
                xy_group.attrs['num_frames'] = max_frames
                xy_group.attrs['description'] = 'XY bead tracking data (compact layout)'

                if metadata:
                    for key, value in metadata.items():
                        if value is not None:
                            xy_group.attrs[key] = value

                # Save per-bead attrs
                for bead_idx, bead in enumerate(beads_data):
                    bead_id = bead['id']
                    initial_x = 0
                    initial_y = 0
                    initial_z = None
                    if bead.get('positions'):
                        init = bead['positions'][0]
                        if isinstance(init, (list, tuple)) and len(init) >= 2:
                            initial_x = init[0]
                            initial_y = init[1]
                            if len(init) >= 3:
                                initial_z = init[2]
                    xy_group.attrs[f'bead_{bead_idx}_id'] = bead_id
                    xy_group.attrs[f'bead_{bead_idx}_initial_x'] = initial_x
                    xy_group.attrs[f'bead_{bead_idx}_initial_y'] = initial_y
                    if initial_z is not None:
                        xy_group.attrs[f'bead_{bead_idx}_initial_z'] = initial_z
                    if 'stuck' in bead:
                        xy_group.attrs[f'bead_{bead_idx}_stuck'] = bool(bead['stuck'])

                # Stack templates if present and not dropped
                if not drop_templates and any(t is not None for t in templates_list):
                    # find template shape from first non-None
                    first_tpl = next(t for t in templates_list if t is not None)
                    tshape = first_tpl.shape
                    stacked = np.zeros((num_beads, tshape[0], tshape[1]), dtype=first_tpl.dtype)
                    for i, t in enumerate(templates_list):
                        if t is None:
                            stacked[i, :, :] = 0
                        else:
                            stacked[i, :, :] = t
                    # store as float16 to save space
                    xy_group.create_dataset('templates', data=stacked.astype(np.float16),
                                            dtype=np.float16, chunks=(1,) + tshape, compression='gzip', compression_opts=4)

                # Build and save optional arrays if any bead provided them
                # forces: (F, B, 3)
                if any(f is not None for f in forces_list):
                    fshape = (max_frames, num_beads, 3)
                    farr = np.zeros(fshape, dtype=np.float32)
                    for bi, fs in enumerate(forces_list):
                        if fs is None:
                            farr[:, bi, :] = 0.0
                        else:
                            for fi, vv in enumerate(fs):
                                farr[fi, bi, :] = vv
                    xy_group.create_dataset('forces', data=farr, dtype=np.float32,
                                             chunks=(1, num_beads, 3), compression='gzip', compression_opts=4)

                # rms: store as static per-bead value (B,). If provided as per-frame series,
                # reduce to per-bead mean across frames.
                if any(r is not None for r in rms_list):
                    r_bead = np.zeros((num_beads,), dtype=np.float32)
                    for bi, rs in enumerate(rms_list):
                        if rs is None:
                            r_bead[bi] = 0.0
                        else:
                            arr = np.asarray(rs)
                            if arr.ndim == 0 or arr.size == 1:
                                r_bead[bi] = float(arr)
                            else:
                                r_bead[bi] = float(np.nanmean(arr))
                    xy_group.create_dataset('rms_per_bead', data=r_bead, dtype=np.float32,
                                             chunks=(num_beads,), compression='gzip', compression_opts=4)

                # symmetry: store as static per-bead value (B,). Reduce per-frame series by mean.
                if any(s is not None for s in symmetry_list):
                    s_bead = np.zeros((num_beads,), dtype=np.float32)
                    for bi, ss in enumerate(symmetry_list):
                        if ss is None:
                            s_bead[bi] = 0.0
                        else:
                            arr = np.asarray(ss)
                            if arr.ndim == 0 or arr.size == 1:
                                s_bead[bi] = float(arr)
                            else:
                                s_bead[bi] = float(np.nanmean(arr))
                    xy_group.create_dataset('symmetry_per_bead', data=s_bead, dtype=np.float32,
                                             chunks=(num_beads,), compression='gzip', compression_opts=4)

                # amplitude: (F, B)
                if any(a is not None for a in amplitude_list):
                    aarr = np.zeros((max_frames, num_beads), dtype=np.float32)
                    for bi, aa in enumerate(amplitude_list):
                        if aa is None:
                            aarr[:, bi] = 0.0
                        else:
                            for fi, vv in enumerate(aa):
                                aarr[fi, bi] = vv
                    xy_group.create_dataset('amplitude', data=aarr, dtype=np.float32,
                                             chunks=(1, num_beads), compression='gzip', compression_opts=4)

                # frequency: (F, B) optional per-frame frequency estimate (Hz)
                if any(fr is not None for fr in frequency_list):
                    farr = np.zeros((max_frames, num_beads), dtype=np.float32)
                    for bi, frs in enumerate(frequency_list):
                        if frs is None:
                            farr[:, bi] = 0.0
                        else:
                            for fi, vv in enumerate(frs):
                                farr[fi, bi] = vv
                    xy_group.create_dataset('frequency', data=farr, dtype=np.float32,
                                             chunks=(1, num_beads), compression='gzip', compression_opts=4)

                # stuck: store static per-bead flag (B,) as uint8. If provided per-frame,
                # reduce with logical OR (any True -> stuck=1).
                if any(st is not None for st in stuck_list):
                    st_bead = np.zeros((num_beads,), dtype=np.uint8)
                    for bi, stl in enumerate(stuck_list):
                        if stl is None:
                            st_bead[bi] = 0
                        else:
                            arr = np.asarray(stl).astype(bool)
                            st_bead[bi] = 1 if np.any(arr) else 0
                    xy_group.create_dataset('stuck_per_bead', data=st_bead, dtype=np.uint8,
                                             chunks=(num_beads,), compression='gzip', compression_opts=1)
                    # also store per-bead attr for backward compatibility
                    for bi in range(num_beads):
                        xy_group.attrs[f'bead_{bi}_stuck'] = bool(st_bead[bi])

            else:
                # Legacy behavior: store positions as int32 and individual templates per bead
                positions_dataset = xy_group.create_dataset('positions', data=tracking_data)

                xy_group.attrs['num_beads'] = num_beads
                xy_group.attrs['num_frames'] = max_frames
                xy_group.attrs['description'] = 'XY bead tracking data'

                if metadata:
                    for key, value in metadata.items():
                        if value is not None:
                            xy_group.attrs[key] = value

                for bead_idx, bead in enumerate(beads_data):
                    bead_id = bead['id']
                    initial_x = 0
                    initial_y = 0
                    initial_z = None
                    if bead.get('positions'):
                        init = bead['positions'][0]
                        if isinstance(init, (list, tuple)) and len(init) >= 2:
                            initial_x = init[0]
                            initial_y = init[1]
                            if len(init) >= 3:
                                initial_z = init[2]
                    xy_group.attrs[f'bead_{bead_idx}_id'] = bead_id
                    xy_group.attrs[f'bead_{bead_idx}_initial_x'] = initial_x
                    xy_group.attrs[f'bead_{bead_idx}_initial_y'] = initial_y
                    if initial_z is not None:
                        xy_group.attrs[f'bead_{bead_idx}_initial_z'] = initial_z
                    if 'stuck' in bead:
                        xy_group.attrs[f'bead_{bead_idx}_stuck'] = bool(bead['stuck'])

                    if 'template' in bead and bead['template'] is not None:
                        template_name = f'bead_{bead_idx}_template'
                        xy_group.create_dataset(template_name, data=bead['template'])
            
            # Flush to ensure data is written
            if should_close:
                f.flush()
            
            Logger.success(f"Saved {num_beads} beads, {max_frames} frames to /analysed_data/xy_tracking/", "TRACKING_IO")
                
        except Exception as e:
            Logger.error(f"Error saving tracking data: {e}", "TRACKING_IO")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Always close if we opened it ourselves
            if should_close and f is not None:
                try:
                    f.close()
                    Logger.debug("HDF5 file closed after save", "TRACKING_IO")
                except Exception as e:
                    Logger.error(f"Error closing HDF5 file: {e}", "TRACKING_IO")
    
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

                    # Load per-bead static arrays if present
                    rms_pb = xy_group['rms_per_bead'][:] if 'rms_per_bead' in xy_group else None
                    sym_pb = xy_group['symmetry_per_bead'][:] if 'symmetry_per_bead' in xy_group else None
                    stuck_pb = xy_group['stuck_per_bead'][:] if 'stuck_per_bead' in xy_group else None

                    # Reconstruct bead dictionaries
                    for bead_idx in range(num_beads):
                        positions = []
                        for frame_idx in range(tracking_data.shape[0]):  # type: ignore
                            coords = tracking_data[frame_idx, bead_idx, :]  # type: ignore
                            if coords.size == 3:
                                x, y, z = coords
                                if x >= 0 and y >= 0:
                                    positions.append((int(x), int(y), int(z)))
                                else:
                                    break
                            else:
                                x, y = coords[:2]
                                if x >= 0 and y >= 0:
                                    positions.append((int(x), int(y)))
                                else:
                                    break

                        if not positions:
                            continue

                        bead_id = int(metadata.get(f'bead_{bead_idx}_id', bead_idx))
                        initial_x = int(metadata.get(f'bead_{bead_idx}_initial_x', positions[0][0]))
                        initial_y = int(metadata.get(f'bead_{bead_idx}_initial_y', positions[0][1]))

                        template = None
                        # Support stacked templates ('templates') (compact layout) or per-bead datasets
                        if 'templates' in xy_group:
                            try:
                                template = xy_group['templates'][bead_idx][:]  # type: ignore
                            except Exception:
                                template = None
                        else:
                            template_name = f'bead_{bead_idx}_template'
                            if template_name in xy_group:
                                template = xy_group[template_name][:]  # type: ignore

                        bead = {
                            'id': bead_id,
                            'positions': positions,
                            'initial_pos': (initial_x, initial_y),
                            'template': template,
                            'stuck': bool(stuck_pb[bead_idx]) if stuck_pb is not None else bool(xy_group.attrs.get(f'bead_{bead_idx}_stuck', False)),
                            'rms': float(rms_pb[bead_idx]) if rms_pb is not None else None,
                            'symmetry': float(sym_pb[bead_idx]) if sym_pb is not None else None,
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
                            coords = tracking_data[frame_idx, bead_idx, :]  # type: ignore
                            if coords.size == 3:
                                x, y, z = coords
                                if x >= 0 and y >= 0:
                                    positions.append((int(x), int(y), int(z)))
                                else:
                                    break
                            else:
                                x, y = coords[:2]
                                if x >= 0 and y >= 0:
                                    positions.append((int(x), int(y)))
                                else:
                                    break
                        
                        if not positions:
                            continue
                        
                        bead_id = int(metadata.get(f'bead_{bead_idx}_id', bead_idx))
                        initial_x = int(metadata.get(f'bead_{bead_idx}_initial_x', positions[0][0]))
                        initial_y = int(metadata.get(f'bead_{bead_idx}_initial_y', positions[0][1]))
                        
                        template = None
                        template = None
                        # Old-style per-dataset template name
                        template_name = f'xy_tracking_bead_{bead_idx}_template'
                        if template_name in f['analysed_data']:  # type: ignore
                            template = f['analysed_data'][template_name][:]  # type: ignore
                        
                        bead = {
                            'id': bead_id,
                            'positions': positions,
                            'initial_pos': (initial_x, initial_y),
                            'template': template,
                            'stuck': bool(metadata.get(f'bead_{bead_idx}_stuck', False))
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

        # Prefer compact group locations; fallback to xy_tracking or legacy load
        group_candidates = ['analysed_data/xy_tracking', 'analysed_data_compact', 'analysed_data']
        found_group = None
        with h5py.File(hdf5_path, 'r') as f:
            for g in group_candidates:
                if g in f and isinstance(f[g], h5py.Group):
                    # prefer groups that contain positions dataset
                    if 'positions' in f[g]:
                        found_group = g
                        break
                    # if last candidate and group exists, accept it
                    if found_group is None:
                        found_group = g

        # If no suitable group found, fall back to load_from_hdf5 and old behavior
        if found_group is None:
            beads_data, metadata = TrackingDataIO.load_from_hdf5(hdf5_path)
            if not beads_data:
                raise ValueError("No tracking data available to export")

            Logger.debug(f"Exporting to CSV (legacy load)", "TRACKING_IO")
            import csv
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['frame']
                for bead in beads_data:
                    bead_id = bead['id']
                    header.extend([f'bead_{bead_id}_x', f'bead_{bead_id}_y'])
                writer.writerow(header)
                max_frames = max(len(bead['positions']) for bead in beads_data)
                for frame_idx in range(max_frames):
                    row: List[Any] = [frame_idx]
                    for bead in beads_data:
                        if frame_idx < len(bead['positions']):
                            x, y = bead['positions'][frame_idx]
                            row.extend([x, y])
                        else:
                            row.extend(['', ''])
                    writer.writerow(row)

            Logger.success(f"Exported {len(beads_data)} beads to {output_csv_path}", "TRACKING_IO")
            return output_csv_path

        # Read directly from the compact group
        with h5py.File(hdf5_path, 'r') as f:
            g = f[found_group]
            positions = g['positions'][...] if 'positions' in g else None
            
            frequency = g['frequency'][...] if 'frequency' in g else None
            forces = g['forces'][...] if 'forces' in g else None
            rms_pb = g['rms_per_bead'][...] if 'rms_per_bead' in g else None
            if rms_pb is None and 'rms' in g:
                try:
                    rms_pb = np.nanmean(g['rms'][...], axis=0)
                except Exception:
                    rms_pb = None

            sym_pb = g['symmetry_per_bead'][...] if 'symmetry_per_bead' in g else None
            if sym_pb is None and 'symmetry' in g:
                try:
                    sym_pb = np.nanmean(g['symmetry'][...], axis=0)
                except Exception:
                    sym_pb = None
            amplitude = g['amplitude'][...] if 'amplitude' in g else None
            stuck_pb = g['stuck_per_bead'][...] if 'stuck_per_bead' in g else None
            if stuck_pb is None and 'stuck' in g:
                try:
                    stuck_arr = g['stuck'][...]
                    stuck_pb = np.any(stuck_arr.astype(bool), axis=0).astype(np.uint8)
                except Exception:
                    stuck_pb = None
            num_frames = int(g.attrs.get('num_frames', positions.shape[0] if positions is not None else 0))
            num_beads = int(g.attrs.get('num_beads', positions.shape[1] if positions is not None else 0))
            bead_ids = [int(g.attrs.get(f'bead_{i}_id', i)) for i in range(num_beads)]
            # fallback per-bead stuck flags from attrs if dataset not present
            stuck_attr_fallback = [bool(g.attrs.get(f'bead_{i}_stuck', False)) for i in range(num_beads)]

        # Build CSV header fields based on available datasets
        field_map = {
            'position': positions is not None,
            'forces': forces is not None,
            'rms': rms_pb is not None,
            'symmetry': sym_pb is not None,
            'amplitude': amplitude is not None,
            'stuck': (stuck_pb is not None) or any(stuck_attr_fallback),
            'frequency': frequency is not None,
        }

        import csv
        Logger.debug(f"Exporting to CSV from group {found_group}", "TRACKING_IO")
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            header = ['frame']
            for bi, bead_id in enumerate(bead_ids):
                if field_map['position']:
                    # positions may be (x,y) or (x,y,z)
                    if positions is not None and positions.ndim == 3 and positions.shape[2] == 3:
                        header.extend([f'bead_{bead_id}_x', f'bead_{bead_id}_y', f'bead_{bead_id}_z'])
                    else:
                        header.extend([f'bead_{bead_id}_x', f'bead_{bead_id}_y'])
                if field_map['forces']:
                    header.extend([f'bead_{bead_id}_fx', f'bead_{bead_id}_fy', f'bead_{bead_id}_fz'])
                if field_map['rms']:
                    header.append(f'bead_{bead_id}_rms')
                if field_map['symmetry']:
                    header.append(f'bead_{bead_id}_symmetry')
                if field_map['amplitude']:
                    header.append(f'bead_{bead_id}_amplitude')
                if field_map['frequency']:
                    header.append(f'bead_{bead_id}_frequency')
                if field_map['stuck']:
                    header.append(f'bead_{bead_id}_stuck')
            writer.writerow(header)

            # Rows per frame
            for frame_idx in range(num_frames):
                row: List[Any] = [frame_idx]
                for bi in range(num_beads):
                    # positions
                    if field_map['position']:
                        if positions is not None and positions.ndim == 3 and positions.shape[2] == 3:
                            x, y, z = positions[frame_idx, bi]  # type: ignore
                            if x >= 0 and y >= 0:
                                row.extend([int(x), int(y), int(z)])
                            else:
                                row.extend(['', '', ''])
                        else:
                            x, y = positions[frame_idx, bi]  # type: ignore
                            if x >= 0 and y >= 0:
                                row.extend([int(x), int(y)])
                            else:
                                row.extend(['', ''])
                    # forces
                    if field_map['forces']:
                        fx, fy, fz = forces[frame_idx, bi]
                        row.extend([float(fx), float(fy), float(fz)])
                    # rms (static per-bead)
                    if field_map['rms']:
                        row.append(float(rms_pb[bi]))
                    # symmetry (static per-bead)
                    if field_map['symmetry']:
                        row.append(float(sym_pb[bi]))
                    # amplitude
                    if field_map['amplitude']:
                        row.append(float(amplitude[frame_idx, bi]))
                    # frequency
                    if field_map['frequency']:
                        row.append(float(frequency[frame_idx, bi]))
                    # stuck (static per-bead)
                    if field_map['stuck']:
                        if stuck_pb is not None:
                            row.append(int(stuck_pb[bi]))
                        else:
                            row.append(int(stuck_attr_fallback[bi]))

                writer.writerow(row)

        Logger.success(f"Exported {num_beads} beads ({num_frames} frames) to {output_csv_path}", "TRACKING_IO")
        return output_csv_path
