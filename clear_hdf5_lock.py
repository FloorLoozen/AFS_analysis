"""Clear HDF5 file locks by attempting to open and close them properly."""

import h5py
import sys
from pathlib import Path

def clear_hdf5_lock(file_path):
    """
    Try to clear HDF5 file lock by opening and closing it.
    
    Args:
        file_path: Path to HDF5 file
    """
    try:
        print(f"Attempting to clear lock on: {file_path}")
        
        # Try opening in read mode first
        try:
            with h5py.File(file_path, 'r') as f:
                print(f"✓ File opened successfully in read mode")
                print(f"  Keys: {list(f.keys())}")
        except Exception as e:
            print(f"✗ Cannot open in read mode: {e}")
            
            # Try append mode to clear write lock
            try:
                with h5py.File(file_path, 'a') as f:
                    print(f"✓ File opened and closed in append mode")
                    print(f"  Lock should now be cleared")
            except Exception as e2:
                print(f"✗ Cannot open in append mode: {e2}")
                print(f"\nThe file may be locked by another process.")
                print(f"Please close any programs that might have the file open.")
                return False
        
        print(f"\n✓ File is now accessible")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clear_hdf5_lock.py <path_to_hdf5_file>")
        print("\nOr drag and drop an HDF5 file onto this script.")
        
        # Interactive mode
        file_path = input("\nEnter path to HDF5 file (or press Enter to exit): ").strip()
        if not file_path:
            sys.exit(0)
    else:
        file_path = sys.argv[1]
    
    # Remove quotes if present
    file_path = file_path.strip('"').strip("'")
    
    if not Path(file_path).exists():
        print(f"✗ File not found: {file_path}")
        sys.exit(1)
    
    if not file_path.endswith('.h5'):
        print(f"⚠ Warning: File doesn't have .h5 extension")
    
    success = clear_hdf5_lock(file_path)
    sys.exit(0 if success else 1)
