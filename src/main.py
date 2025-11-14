"""
Main application entry point for AFS Analysis.

Simple video analysis application with PyQt5 interface.
"""

import sys
from pathlib import Path


def setup_python_path() -> None:
    """Add src directory to Python path."""
    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def create_application():
    """Create and configure the PyQt5 application."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    # Enable high DPI support
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # type: ignore
    except AttributeError:
        pass  # These attributes might not exist in all Qt versions
    
    app = QApplication(sys.argv)
    app.setApplicationName("AFS Analysis")
    app.setApplicationVersion("1.0.0")
    
    return app


def main():
    """Initialize and run the AFS Analysis application."""
    try:
        setup_python_path()
        
        # Create PyQt5 application
        app = create_application()
        
        # Import and create main window
        from ui.main_window import MainWindow
        window = MainWindow()
        window.showMaximized()  # Start maximized by default
        
        # Start the Qt event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
