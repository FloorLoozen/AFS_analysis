import h5py
import sys

def inspect_hdf5(filename):
    """Inspect HDF5 file structure."""
    with h5py.File(filename, 'r') as f:
        print(f"=== HDF5 File Structure for {filename} ===\n")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                if obj.attrs:
                    print(f"  Attributes: {dict(obj.attrs)}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                if obj.attrs:
                    print(f"  Attributes: {dict(obj.attrs)}")
        
        f.visititems(print_structure)
        
        print("\n=== Top Level Keys ===")
        print(list(f.keys()))

if __name__ == "__main__":
    inspect_hdf5("20251023.hdf5")
