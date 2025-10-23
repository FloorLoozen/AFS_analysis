"""Main application window for AFS Analysis."""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QAction, 
    QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    """Main application window with two-column layout.
    
    Left column: Video player controls
    Right column: Analysis controls and tools
    """

    def __init__(self):
        """Initialize the main application window."""
        super().__init__()
        
        self.video_widget = None
        self.analysis_widget = None
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AFS Analysis")
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central layout
        self._create_central_layout()
        
        self.statusBar().showMessage("Ready")

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        toggle_action = QAction("Toggle Maximize", self)
        toggle_action.setShortcut("F11")
        toggle_action.triggered.connect(self._toggle_fullscreen)
        file_menu.addAction(toggle_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_central_layout(self):
        """Create main layout: left column (video only) + right column (analysis tabs)."""
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(10)
        
        # Left column - Video player only
        from src.ui.video_widget import VideoWidget
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_widget)
        
        # Right column - Analysis tabs (includes Info tab)
        from src.ui.analysis_widget import AnalysisWidget
        self.analysis_widget = AnalysisWidget()
        self.analysis_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.analysis_widget)
        
        # Set equal stretch for 50/50 horizontal split
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        
        # Connect widgets
        self.analysis_widget.set_video_widget(self.video_widget)
        
        # Connect signals
        self.video_widget.video_loaded.connect(self.analysis_widget.update_video_info)
        
        self.setCentralWidget(central)

    def _open_video(self):
        """Open video file dialog."""
        if self.video_widget:
            self.video_widget.open_video_dialog()
            # Update analysis widget info after loading
            if self.analysis_widget:
                self.analysis_widget.update_video_info()

    def _toggle_fullscreen(self):
        """Toggle between maximized and normal window."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AFS Analysis",
            "AFS Analysis v1.0.0\n\n"
            "Video analysis tool for Atomic Force Spectroscopy data."
        )

    def closeEvent(self, event):
        """Handle window close event."""
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clean up video resources
            if self.video_widget:
                self.video_widget.cleanup()
            event.accept()
        else:
            event.ignore()
