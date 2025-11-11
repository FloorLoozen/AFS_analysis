"""Simple logging utility for AFS Analysis."""

import sys
from datetime import datetime
from typing import Optional


class Logger:
    """Simple logger for debugging and info messages."""
    
    DEBUG = False  # Set to True for verbose output
    
    @staticmethod
    def debug(message: str, module: Optional[str] = None):
        """Log debug message (only if DEBUG is enabled)."""
        if Logger.DEBUG:
            prefix = f"[{module}]" if module else "[DEBUG]"
            print(f"{prefix} {message}")
    
    @staticmethod
    def info(message: str, module: Optional[str] = None):
        """Log info message."""
        prefix = f"[{module}]" if module else "[INFO]"
        print(f"{prefix} {message}")
    
    @staticmethod
    def warning(message: str, module: Optional[str] = None):
        """Log warning message."""
        prefix = f"[{module}]" if module else "[WARNING]"
        print(f"⚠ {prefix} {message}", file=sys.stderr)
    
    @staticmethod
    def error(message: str, module: Optional[str] = None):
        """Log error message."""
        prefix = f"[{module}]" if module else "[ERROR]"
        print(f"✗ {prefix} {message}", file=sys.stderr)
    
    @staticmethod
    def success(message: str, module: Optional[str] = None):
        """Log success message."""
        prefix = f"[{module}]" if module else "[SUCCESS]"
        print(f"✓ {prefix} {message}")
