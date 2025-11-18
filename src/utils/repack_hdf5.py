"""Repack HDF5 files with low-risk precision reductions and dataset normalization.

Usage:
  python -m src.utils.repack_hdf5 <source.hdf5>

What it does:
  - Creates a new file next to the source named <source>_repacked.hdf5
  - Converts bead templates (analysed_data/xy_tracking/bead_*_template) from float32 -> float16
    to halve storage for those small arrays.
  - Converts positions (analysed_data/xy_tracking/positions) from int32 -> uint16/int16
    if values fit in that range (checked automatically). If not safe, leaves as-is.
  - Ensures datasets that mention 'force', 'forces', 'rms', or 'symmetry' use float32
    (downcasts float64 -> float32).
  - Does NOT rewrite large video dataset `raw_data/main_video` (it will be copied as-is).
  - Preserves group/dataset attributes and attempts to preserve chunking and filters when
    copying datasets that are not transformed.

This is conservative: it only changes dtype when it can safely do so or when the user
explicitly asked for lower precision for small arrays.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List

import h5py
import numpy as np
from src.utils.analysed_writer import write_analysed_data


def choose_position_dtype(arr: np.ndarray) -> np.dtype:
    """Return a safe reduced dtype for integer coordinate arrays if possible.

    If all values are >=0 and <=65535 -> uint16.
    Else if all values fit in int16 range -> int16.
    Else return original dtype.
    """
    mi = int(arr.min())
    ma = int(arr.max())
    if mi >= 0 and ma <= 65535:
        return np.uint16
    if mi >= -32768 and ma <= 32767:
        return np.int16
    return arr.dtype


def should_convert_template(name: str, dtype: np.dtype) -> bool:
    return name.startswith("analysed_data/xy_tracking/bead_") and name.endswith("_template") and np.dtype(dtype) == np.float32


def should_handle_positions(name: str, dtype: np.dtype) -> bool:
    return name.endswith("/positions") and np.issubdtype(np.dtype(dtype), np.integer)


def should_downcast_float(name: str, dtype: np.dtype) -> bool:
    n = name.lower()
    keywords = ("force", "forces", "rms", "symmetry", "sym")
    return any(k in n for k in keywords) and np.dtype(dtype) == np.float64


def repack_file(src_path: str, dst_path: Optional[str] = None) -> dict:
    src_path = os.path.abspath(src_path)
    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = base + "_repacked" + ext

    summary = {
        "src": src_path,
        "dst": dst_path,
        "converted": [],
        "copied": [],
        "skipped": [],
    }

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # copy root attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # copy top-level groups and their attrs
        for name, obj in src.items():
            if isinstance(obj, h5py.Group):
                g = dst.create_group(name)
                for k, v in obj.attrs.items():
                    g.attrs[k] = v

        # visitor: create groups/datasets in dst
        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                g = dst.require_group(name)
                for k, v in obj.attrs.items():
                    g.attrs[k] = v
                return

            if isinstance(obj, h5py.Dataset):
                dtype = obj.dtype
                action = "copy"
                new_dtype = None

                if should_convert_template(name, dtype):
                    action = "convert"
                    new_dtype = np.float16

                elif should_handle_positions(name, dtype):
                    data = obj[...]
                    chosen = choose_position_dtype(data)
                    if np.dtype(chosen) != np.dtype(data.dtype):
                        action = "convert"
                        new_dtype = chosen
                    else:
                        action = "copy"

                elif should_downcast_float(name, dtype):
                    action = "convert"
                    new_dtype = np.float32

                if action == "copy":
                    try:
                        src.copy(name, dst)
                        summary["copied"].append(name)
                    except Exception:
                        # fallback
                        data = obj[...]
                        dset = dst.create_dataset(name, data=data, dtype=data.dtype,
                                                  chunks=obj.chunks, compression=obj.compression,
                                                  compression_opts=obj.compression_opts)
                        for k, v in obj.attrs.items():
                            dset.attrs[k] = v
                        summary["copied"].append(name)
                    return

                if action == "convert":
                    data = obj[...]
                    cast = data.astype(new_dtype)
                    kwargs = {}
                    if cast.size > 100000:
                        kwargs['chunks'] = obj.chunks if obj.chunks is not None else True
                        kwargs['compression'] = obj.compression or 'gzip'
                        kwargs['compression_opts'] = obj.compression_opts or 4
                    else:
                        kwargs['compression'] = 'gzip'
                        kwargs['compression_opts'] = 4

                    dset = dst.create_dataset(name, data=cast, dtype=cast.dtype, **kwargs)
                    for k, v in obj.attrs.items():
                        dset.attrs[k] = v
                    summary["converted"].append({"name": name, "from": str(dtype), "to": str(cast.dtype), "size": int(cast.size)})
                    return

        src.visititems(visitor)

    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Repack HDF5 with conservative precision reductions")
    p.add_argument("src", help="Source HDF5 file")
    p.add_argument("--dst", help="Destination HDF5 file (optional)")
    p.add_argument("--migrate-analysed", action='store_true', help="Migrate/normalize the analysed_data group into compact layout")
    p.add_argument("--drop-templates", action='store_true', help="Drop templates when migrating analysed_data (save space)")
    args = p.parse_args(argv)

    if not os.path.exists(args.src):
        print("Source file not found:", args.src, file=sys.stderr)
        return 2

    dst = args.dst

    if args.migrate_analysed:
        base, ext = os.path.splitext(os.path.abspath(args.src))
        dst_path = dst or (base + "_migrated" + ext)
        migrate_analysed(args.src, dst_path, drop_templates=bool(args.drop_templates))
        return 0

    summary = repack_file(args.src, dst)

    print("Repack complete")
    print("Source:", summary['src'])
    print("Destination:", summary['dst'])
    print("Converted datasets:")
    for c in summary['converted']:
        print("  - {name}: {from} -> {to} (n={size})".format(**c))
    print("Copied datasets:")
    for c in summary['copied'][:50]:
        print("  -", c)
    return 0


def migrate_analysed(src_path: str, dst_path: str, drop_templates: bool = False) -> None:
    """Create dst_path by copying src_path but normalizing /analysed_data into compact layout.

    This function is aggressive: it will convert positions to smaller integer types when
    safe, downcast floats, and pack templates into a stacked float16 array unless
    drop_templates is True.
    """
    src_path = os.path.abspath(src_path)
    dst_path = os.path.abspath(dst_path)
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # copy everything except analysed_data
        for name, obj in src.items():
            if name == 'analysed_data':
                continue
            try:
                src.copy(name, dst)
            except Exception:
                # fallback: shallow copy attrs and datasets
                if isinstance(obj, h5py.Group):
                    g = dst.create_group(name)
                    for k, v in obj.attrs.items():
                        g.attrs[k] = v
                elif isinstance(obj, h5py.Dataset):
                    data = obj[...]
                    dset = dst.create_dataset(name, data=data, dtype=data.dtype,
                                              chunks=obj.chunks, compression=obj.compression,
                                              compression_opts=obj.compression_opts)
                    for k, v in obj.attrs.items():
                        dset.attrs[k] = v

        # now build analysed_data from source if present
        if 'analysed_data' not in src:
            print('No analysed_data in source; copied file without migration.')
            return

        a = src['analysed_data']

        # 1) try to find positions
        positions = None
        # common paths
        candidate_paths = [
            'analysed_data/xy_tracking/positions',
            'analysed_data/positions',
            'analysed_data/xy_tracking/pos',
        ]
        for p in candidate_paths:
            if p in src:
                positions = src[p][...]
                break

        # 2) templates: collect bead_*_template under analysed_data/xy_tracking
        templates_list = []
        if 'analysed_data/xy_tracking' in src:
            xy = src['analysed_data/xy_tracking']
            # find keys like bead_0_template
            bead_keys = []
            for k in xy.keys():
                if k.endswith('_template'):
                    bead_keys.append(k)
            # sort by bead index if present
            def bead_index(key: str) -> int:
                import re
                m = re.search(r'bead_(\d+)', key)
                return int(m.group(1)) if m else 0

            bead_keys = sorted(bead_keys, key=bead_index)
            for k in bead_keys:
                templates_list.append(xy[k][...])

        templates = None
        if templates_list and not drop_templates:
            templates = np.stack(templates_list, axis=0)

        # 3) find forces/rms/symmetry datasets inside analysed_data
        forces = None
        rms = None
        symmetry = None
        for root, ds in _iter_datasets(a):
            lname = root.lower()
            if 'force' in lname and forces is None:
                forces = ds[...]
            if 'rms' in lname and rms is None:
                rms = ds[...]
            if 'symmetry' in lname and symmetry is None:
                symmetry = ds[...]

        if positions is None:
            print('positions dataset not found in analysed_data; migration will skip analysed_data creation')
            return

        # write analysed_data into dst using helper
        metaattrs = {}
        # copy top-level analysed_data attrs if present
        for k, v in a.attrs.items():
            metaattrs[k] = v

        write_analysed_data(dst, positions, forces=forces, rms=rms, symmetry=symmetry,
                            templates=templates, metaattrs=metaattrs, drop_templates=drop_templates)

    print('Migration complete ->', dst_path)


def _iter_datasets(group) -> List[tuple]:
    """Yield (fullpath, dataset) for datasets under group."""
    out = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, obj))
    group.visititems(visitor)
    return out


def _gather_analysed_from_source(src) -> dict:
    """Collect positions, templates, forces/rms/symmetry from a source h5py.File or group.

    Returns a dict with keys: positions (ndarray or None), templates (list of arrays), forces, rms, symmetry, metaattrs
    """
    out = {'positions': None, 'templates_list': [], 'forces': None, 'rms': None, 'symmetry': None, 'metaattrs': {}}
    if 'analysed_data' not in src:
        return out

    a = src['analysed_data']
    for k, v in a.attrs.items():
        out['metaattrs'][k] = v

    # try candidate positions
    candidate_paths = [
        'analysed_data/xy_tracking/positions',
        'analysed_data/positions',
        'analysed_data/xy_tracking/pos',
    ]
    for p in candidate_paths:
        if p in src:
            out['positions'] = src[p][...]
            break

    # templates
    if 'analysed_data/xy_tracking' in src:
        xy = src['analysed_data/xy_tracking']
        bead_keys = []
        for k in xy.keys():
            if k.endswith('_template'):
                bead_keys.append(k)

        def bead_index(key: str) -> int:
            import re
            m = re.search(r'bead_(\d+)', key)
            return int(m.group(1)) if m else 0

        bead_keys = sorted(bead_keys, key=bead_index)
        for k in bead_keys:
            out['templates_list'].append(xy[k][...])

    # forces/rms/symmetry: search datasets under analysed_data
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            lname = name.lower()
            try:
                if 'force' in lname and out['forces'] is None:
                    out['forces'] = obj[...]
                if 'rms' in lname and out['rms'] is None:
                    out['rms'] = obj[...]
                if 'symmetry' in lname and out['symmetry'] is None:
                    out['symmetry'] = obj[...]
            except Exception:
                pass

    a.visititems(visitor)
    return out


def verify_migration(file_path: str, target_group: str = 'analysed_data_compact') -> dict:
    """Verify numeric integrity between existing analysed_data layout and compact target group.

    Returns a report dict with comparisons and error metrics.
    """
    report = {'file': file_path, 'target_group': target_group, 'positions': None, 'templates': None, 'forces': None, 'rms': None, 'symmetry': None}
    with h5py.File(file_path, 'r') as f:
        src_info = _gather_analysed_from_source(f)

        # positions
        if src_info['positions'] is None:
            report['positions'] = {'status': 'missing'}
        else:
            orig_pos = src_info['positions']
            if target_group in f and 'positions' in f[target_group]:
                new_pos = f[f'{target_group}/positions'][...]
                # cast back to original dtype for comparison
                new_pos_up = new_pos.astype(orig_pos.dtype)
                eq = np.array_equal(orig_pos, new_pos_up)
                report['positions'] = {'status': 'present', 'equal': bool(eq)}
            else:
                report['positions'] = {'status': 'target_missing'}

        # templates
        templates_report = {'status': None, 'beads': []}
        if src_info['templates_list']:
            templates_report['status'] = 'present'
            # if target exists, compare per-bead
            if target_group in f and 'templates' in f[target_group]:
                stacked = f[f'{target_group}/templates'][...].astype(np.float32)
                for i, orig in enumerate(src_info['templates_list']):
                    if i >= stacked.shape[0]:
                        templates_report['beads'].append({'index': i, 'status': 'missing_in_target'})
                        continue
                    t_new = stacked[i].astype(np.float32)
                    t_orig = orig.astype(np.float32)
                    diff = np.abs(t_orig - t_new)
                    max_abs = float(diff.max())
                    mean_abs = float(diff.mean())
                    # relative error (safe): mean(|diff|/ (|orig|+eps))
                    eps = 1e-8
                    rel = (diff / (np.abs(t_orig) + eps))
                    max_rel = float(np.nanmax(rel))
                    templates_report['beads'].append({'index': i, 'max_abs': max_abs, 'mean_abs': mean_abs, 'max_rel': max_rel})
            else:
                templates_report['status'] = 'target_missing'
        else:
            templates_report['status'] = 'none_in_source'

        report['templates'] = templates_report

        # forces, rms, symmetry - compare if present
        for key in ('forces', 'rms', 'symmetry'):
            val = src_info.get(key)
            if val is None:
                report[key] = {'status': 'missing_in_source'}
                continue
            if target_group in f and key in f[target_group]:
                newv = f[f'{target_group}/{key}'][...].astype(np.float64)
                origv = np.array(val).astype(np.float64)
                diff = np.abs(origv - newv)
                max_abs = float(diff.max())
                mean_abs = float(diff.mean())
                # relative
                denom = np.maximum(np.abs(origv), 1e-8)
                rel = diff / denom
                max_rel = float(np.nanmax(rel))
                report[key] = {'status': 'present', 'max_abs': max_abs, 'mean_abs': mean_abs, 'max_rel': max_rel}
            else:
                report[key] = {'status': 'target_missing'}

    return report


def migrate_analysed_inplace(src_path: str, drop_templates: bool = True, target_group: str = 'analysed_data_compact') -> None:
    """Add a compact analysed_data group inside `src_path` without creating a new file.

    The function will collect analysed_data pieces from existing layout and write the
    compact layout into `/{target_group}` to avoid overwriting existing `/analysed_data`.
    """
    src_path = os.path.abspath(src_path)
    with h5py.File(src_path, 'r+') as src:
        gathered = _gather_analysed_from_source(src)
        positions = gathered['positions']
        templates = None
        if gathered['templates_list'] and not drop_templates:
            templates = np.stack(gathered['templates_list'], axis=0)

        forces = gathered['forces']
        rms = gathered['rms']
        symmetry = gathered['symmetry']

        if positions is None:
            print('positions not found; skipping inplace migration for', src_path)
            return

        # write into target_group
        write_analysed_data(src, positions, forces=forces, rms=rms, symmetry=symmetry,
                            templates=templates, metaattrs=gathered['metaattrs'],
                            drop_templates=drop_templates, group_path=target_group)

    print('In-place migration complete ->', src_path, 'group:', target_group)


def describe_compact_schema(file_path: str, group: str = 'analysed_data_compact') -> dict:
    """Return a compact description of the analysed_data compact group in the file.

    Useful for quick inspection and for showing the new format to users.
    """
    out = {'file': file_path, 'group': group, 'exists': False, 'datasets': {}}
    with h5py.File(file_path, 'r') as f:
        if group not in f:
            return out
        out['exists'] = True
        g = f[group]
        # group attrs
        out['attrs'] = {k: v for k, v in g.attrs.items()}
        for name, obj in g.items():
            if isinstance(obj, h5py.Dataset):
                out['datasets'][name] = {'shape': tuple(obj.shape), 'dtype': str(obj.dtype), 'chunks': obj.chunks, 'compression': obj.compression}
    return out


if __name__ == '__main__':
    raise SystemExit(main())
