"""Data export utilities - separate from UI."""

import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List


class DataExporter:
    """Handles exporting analysis results to various formats."""
    
    @staticmethod
    def export_to_csv(data: Dict[str, np.ndarray], file_path: str):
        """
        Export data to CSV file.
        
        Args:
            data: Dictionary of column_name -> data_array
            file_path: Output CSV file path
        """
        import csv
        
        # Get length of data
        lengths = [len(v) for v in data.values()]
        if not lengths:
            raise ValueError("No data to export")
        
        max_length = max(lengths)
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(data.keys())
            
            # Write data rows
            for i in range(max_length):
                row = []
                for key, values in data.items():
                    if i < len(values):
                        row.append(values[i])
                    else:
                        row.append('')
                writer.writerow(row)
    
    @staticmethod
    def export_to_json(data: Dict[str, Any], file_path: str):
        """
        Export data to JSON file.
        
        Args:
            data: Dictionary of data (will convert numpy arrays to lists)
            file_path: Output JSON file path
        """
        # Convert numpy arrays to lists
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    @staticmethod
    def export_to_numpy(data: Dict[str, np.ndarray], file_path: str):
        """
        Export data to NumPy .npz file.
        
        Args:
            data: Dictionary of array_name -> data_array
            file_path: Output .npz file path
        """
        np.savez(file_path, **data)
    
    @staticmethod
    def export_metadata(metadata: Dict[str, Any], file_path: str):
        """
        Export metadata to text file.
        
        Args:
            metadata: Metadata dictionary
            file_path: Output text file path
        """
        with open(file_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
