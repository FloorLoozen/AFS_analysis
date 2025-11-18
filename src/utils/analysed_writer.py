"""Helper to write analysed_data group in the recommended compact layout.

Provides a single function `write_analysed_data` that creates/overwrites
`/analysed_data` in an open HDF5 file object.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import h5py
import numpy as np


def write_analysed_data(h5file: h5py.File,
                        positions: np.ndarray,
                        forces: Optional[np.ndarray] = None,
                        rms: Optional[np.ndarray] = None,
                        symmetry: Optional[np.ndarray] = None,
                        amplitude: Optional[np.ndarray] = None,
                        frequency: Optional[np.ndarray] = None,
                        stuck: Optional[np.ndarray] = None,
                        templates: Optional[np.ndarray] = None,
                        metaattrs: Optional[dict] = None,
                        drop_templates: bool = True,
                        group_path: str = 'analysed_data/xy_tracking'):
    """Create or overwrite `/analysed_data` group with compact layout.

    positions: (F, B, 2) integer array
    forces: (F, B, 3) float array or None
    rms: (F, B) float array or None
    symmetry: (F, B) float array or None
    templates: (B, H, W) float array or None
    metaattrs: dict of attributes to store at /analysed_data
    drop_templates: if True, don't write templates to file
    """
    # normalize group path
    group_path = group_path.strip('/')

    # remove existing target group if present
    if group_path in h5file:
        del h5file[group_path]

    g = h5file.create_group(group_path)
    attrs = metaattrs.copy() if metaattrs else {}
    attrs.setdefault('created_at', datetime.utcnow().isoformat())
    attrs.setdefault('tool_version', 'unknown')
    attrs.setdefault('description', 'Compact analysed_data layout')
    attrs['num_frames'] = int(positions.shape[0])
    attrs['num_beads'] = int(positions.shape[1])
    for k, v in attrs.items():
        g.attrs[k] = v
    # positions: allow 2D (x,y) or 3D (x,y,z) coords; choose safe integer dtype
    if positions.ndim != 3 or positions.shape[2] not in (2, 3):
        raise ValueError('positions must be shaped (F, B, 2) or (F, B, 3)')

    mi = int(np.nanmin(positions)) if np.any(~np.isnan(positions)) else 0
    ma = int(np.nanmax(positions)) if np.any(~np.isnan(positions)) else 0
    if mi >= 0 and ma <= 65535:
        pos_dtype = np.uint16
    elif mi >= -32768 and ma <= 32767:
        pos_dtype = np.int16
    else:
        pos_dtype = positions.dtype

    coord_dim = positions.shape[2]
    g.create_dataset('positions', data=positions.astype(pos_dtype),
                     dtype=pos_dtype, chunks=(1, positions.shape[1], coord_dim), compression=None)
    g['positions'].attrs['units'] = 'px'

    # forces
    if forces is not None:
        f = forces.astype(np.float32)
        g.create_dataset('forces', data=f, dtype=np.float32,
                         chunks=(1, forces.shape[1], 3), compression='gzip', compression_opts=4)
        g['forces'].attrs['units'] = 'pN'

    # rms — static per-bead data. Accept either per-bead (B,) or per-frame (F,B).
    # If a per-frame array is provided we reduce it to a per-bead summary (mean).
    if rms is not None:
        r = np.asarray(rms)
        if r.ndim == 2:
            # reduce to per-bead mean across frames
            r_bead = np.nanmean(r, axis=0).astype(np.float32)
        else:
            r_bead = r.astype(np.float32)
        # store as per-bead dataset
        g.create_dataset('rms_per_bead', data=r_bead, dtype=np.float32,
                         chunks=(r_bead.shape[0],), compression='gzip', compression_opts=4)

    # symmetry — static per-bead data. Same handling as rms.
    if symmetry is not None:
        s = np.asarray(symmetry)
        if s.ndim == 2:
            s_bead = np.nanmean(s, axis=0).astype(np.float32)
        else:
            s_bead = s.astype(np.float32)
        g.create_dataset('symmetry_per_bead', data=s_bead, dtype=np.float32,
                         chunks=(s_bead.shape[0],), compression='gzip', compression_opts=4)

    # amplitude
    if amplitude is not None:
        a = amplitude.astype(np.float32)
        g.create_dataset('amplitude', data=a, dtype=np.float32,
                         chunks=(1, amplitude.shape[1]), compression='gzip', compression_opts=4)

    # frequency per-frame per-bead (Hz) optional
    if frequency is not None:
        fr = frequency.astype(np.float32)
        g.create_dataset('frequency', data=fr, dtype=np.float32,
                         chunks=(1, frequency.shape[1]), compression='gzip', compression_opts=4)

    # stuck flags — static per-bead. Accept per-bead (B,) or per-frame (F,B).
    # If per-frame provided, we reduce with logical OR (if stuck at any frame -> stuck=1).
    if stuck is not None:
        st = np.asarray(stuck)
        if st.ndim == 2:
            st_bead = (np.any(st.astype(bool), axis=0)).astype(np.uint8)
        else:
            st_bead = st.astype(np.uint8)
        g.create_dataset('stuck_per_bead', data=st_bead, dtype=np.uint8,
                         chunks=(st_bead.shape[0],), compression='gzip', compression_opts=1)

    # templates: optional, but dropped by default to keep layout minimal
    if templates is not None and not drop_templates:
        t = templates.astype(np.float16)
        g.create_dataset('templates', data=t, dtype=np.float16,
                         chunks=(1,) + t.shape[1:], compression='gzip', compression_opts=4)

    return g
