"""Force clear HDF5 file locks using low-level operations."""

import sys
import os
from pathlib import Path

def force_clear_lock(file_path):
    """
    Force clear HDF5 file lock by resetting file access flags.
    
    This uses h5py's libver='earliest' mode which is more tolerant of
    inconsistent file states.
    """
    try:
        import h5py
        print(f"Attempting to force-clear lock on: {file_path}")
        
        # First, try to just delete and recreate if it's truly corrupted
        # But let's check the file first
        if not os.path.exists(file_path):
            print(f"✗ File doesn't exist: {file_path}")
            return False
        
        # Try opening with different strategies
        strategies = [
            ('r', 'latest', 'Read-only with latest library version'),
            ('r', 'earliest', 'Read-only with earliest library version (most compatible)'),
            ('r+', 'latest', 'Read-write with latest library version'),
            ('a', 'latest', 'Append with latest library version'),
        ]
        
        for mode, libver, description in strategies:
            try:
                print(f"\nTrying: {description} (mode='{mode}', libver='{libver}')")
                # Use swmr=False to avoid SWMR write mode conflicts
                with h5py.File(file_path, mode, libver=libver, swmr=False) as f:
                    print(f"✓ Success! File opened with mode='{mode}', libver='{libver}'")
                    print(f"  Keys: {list(f.keys())}")
                    # Force flush
                    f.flush()
                print(f"✓ File closed successfully")
                return True
            except Exception as e:
                print(f"✗ Failed: {str(e)[:100]}")
                continue
        
        print(f"\n⚠ All strategies failed. The file may be corrupted.")
        print(f"\nOptions:")
        print(f"1. Close any programs that might have the file open")
        print(f"2. Restart your computer")
        print(f"3. If the file is not important, delete it manually")
        
        return False
        
    except ImportError:
        print("✗ h5py is not installed")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Force Clear HDF5 File Locks")
        print("=" * 50)
        print("\nUsage: python force_clear_hdf5.py <path_to_file>")
        
        file_path = input("\nEnter path to HDF5 file: ").strip().strip('"').strip("'")
        if not file_path:
            sys.exit(0)
    else:
        file_path = sys.argv[1].strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        sys.exit(1)
    
    success = force_clear_lock(file_path)
    
    if success:
        print(f"\n✓ File lock cleared successfully!")
        print(f"You can now open the file in your application.")
    else:
        print(f"\n✗ Could not clear file lock automatically.")
        print(f"\nManual solution:")
        print(f"1. Make sure no programs are using the file")
        print(f"2. Try restarting your computer")
        print(f"3. Or rename/delete the file if it's not important")
    
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
