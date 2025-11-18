"""Helper functions to read/write the per-bead stuck flags in HDF5 files.

This module centralizes writes so the UI can call a single helper instead of
doing ad-hoc HDF5 writes inline. The helpers are minimal and "best-effort":
they raise on I/O errors so callers can choose how to report failures.
"""
from pathlib import Path
from typing import Sequence, Optional
import numpy as np
import h5py


def write_stuck_per_bead(hdf5_path: str, stuck: Sequence[int]) -> None:
    """Write a full stuck_per_bead array to the file.

    Args:
        hdf5_path: Path to the HDF5 file.
        stuck: Sequence of 0/1 values (or truthy/falsy) per bead.

    Raises:
        OSError / IOError on file write problems.
    """
    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    arr = np.asarray(stuck, dtype=np.uint8)

    with h5py.File(str(path), 'a') as hf:
        # Ensure analysed_data/per_bead exist
        ad = hf.require_group('analysed_data')
        pb = ad.require_group('per_bead')

        if 'stuck_per_bead' in pb:
            ds = pb['stuck_per_bead']
            # Resize if necessary
            if ds.shape != arr.shape:
                # Overwrite with new dataset to avoid complex resizing semantics
                del pb['stuck_per_bead']
                pb.create_dataset('stuck_per_bead', data=arr, dtype='u1')
            else:
                ds[:] = arr
        else:
            pb.create_dataset('stuck_per_bead', data=arr, dtype='u1')


def set_stuck_flag(hdf5_path: str, bead_id: int, is_stuck: bool, num_beads: Optional[int] = None) -> None:
    """Set a single bead's stuck flag.

    If the dataset doesn't exist it will be created. If the dataset is shorter
    than required it will be extended (zero-filled) to accommodate the bead_id.

    Args:
        hdf5_path: Path to the HDF5 file.
        bead_id: Index of the bead (0-based).
        is_stuck: True to mark stuck, False to clear.
        num_beads: Optional number of beads to size the dataset to if it needs creation
                   or extension. If omitted, dataset will be sized to bead_id+1.

    Raises:
        FileNotFoundError if file missing.
        ValueError for invalid bead_id.
    """
    if bead_id < 0:
        raise ValueError('bead_id must be >= 0')

    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(str(path), 'a') as hf:
        ad = hf.require_group('analysed_data')
        pb = ad.require_group('per_bead')

        desired_len = (num_beads if (num_beads is not None and num_beads > 0) else (bead_id + 1))

        if 'stuck_per_bead' in pb:
            ds = pb['stuck_per_bead']
            cur_len = ds.shape[0]
            if cur_len <= bead_id or cur_len < desired_len:
                # create new dataset with desired length, copy old data
                old = np.array(ds[:], dtype=np.uint8)
                newlen = max(desired_len, bead_id + 1)
                new = np.zeros((newlen,), dtype=np.uint8)
                new[:old.shape[0]] = old
                # remove and recreate
                del pb['stuck_per_bead']
                pb.create_dataset('stuck_per_bead', data=new, dtype='u1')
                ds = pb['stuck_per_bead']
        else:
            # Create new dataset sized to desired_len (at least bead_id+1)
            newlen = max(desired_len, bead_id + 1)
            ds = pb.create_dataset('stuck_per_bead', data=np.zeros((newlen,), dtype=np.uint8), dtype='u1')

        # Write the flag
        ds[bead_id] = 1 if is_stuck else 0


def read_stuck_per_bead(hdf5_path: str) -> np.ndarray:
    """Read and return the stuck_per_bead array, or an empty array if not present.

    Args:
        hdf5_path: Path to HDF5 file.

    Returns:
        numpy array of dtype uint8
    """
    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(str(path), 'r') as hf:
        if 'analysed_data' in hf and 'per_bead' in hf['analysed_data'] and 'stuck_per_bead' in hf['analysed_data']['per_bead']:
            return np.array(hf['analysed_data']['per_bead']['stuck_per_bead'][:], dtype=np.uint8)
        return np.zeros((0,), dtype=np.uint8)
