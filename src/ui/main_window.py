"""Main application window for AFS Analysis."""

from typing import Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QAction, 
    QMessageBox, QSizePolicy, QTabWidget, QDialog, QFileDialog,
    QGraphicsOpacityEffect
)
from PyQt5.QtCore import QTimer


class MainWindow(QMainWindow):
    """Main application window with tabbed interface.
    
    Main tabs: Traces, Preview, Analysis
    """

    def __init__(self):
        """Initialize the main application window."""
        super().__init__()
        
        self.video_widget: Optional[object] = None  # Will be VideoWidget
        self.tab_widget: Optional[QTabWidget] = None
        self.info_dialog: Optional[QDialog] = None
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AFS Analysis")
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central layout
        self._create_central_layout()
        
        self.statusBar().showMessage("Ready")  # type: ignore
        
        # Maximize window on startup
        self.showMaximized()

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")  # type: ignore
        
        open_action = QAction("Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)  # type: ignore
        
        file_menu.addSeparator()  # type: ignore
        
        toggle_action = QAction("Toggle Maximize", self)
        toggle_action.setShortcut("F11")
        toggle_action.triggered.connect(self._toggle_fullscreen)
        file_menu.addAction(toggle_action)  # type: ignore
        
        file_menu.addSeparator()  # type: ignore
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)  # type: ignore
        file_menu.addAction(exit_action)  # type: ignore

        # Info menu (standalone)
        info_menu = menubar.addMenu("Info")  # type: ignore
        
        show_info_action = QAction("Show Info...", self)
        show_info_action.setShortcut("Ctrl+I")
        show_info_action.triggered.connect(self._show_info_dialog)
        info_menu.addAction(show_info_action)  # type: ignore

        # Help menu
        help_menu = menubar.addMenu("Help")  # type: ignore
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)  # type: ignore

    def _create_central_layout(self):
        """Create main layout with tabs at top level."""
        central = QWidget(self)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)
        
        # Create main tab widget at top level
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Style tabs to be consistent with the rest of the UI
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #b0b0b0;
                background: #f0f0f0;
                margin-top: 0px;
                padding: 0px;
            }
            QTabBar::tab {
                background: #d4d4d4;
                color: #333333;
                border: 1px solid #b0b0b0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 12px 40px;
                margin-right: 4px;
                margin-top: 4px;
                font-size: 11pt;
                min-width: 120px;
                min-height: 20px;
            }
            QTabBar::tab:selected {
                background: #f0f0f0;
                color: #000000;
                font-weight: bold;
                padding-bottom: 14px;
                margin-top: 2px;
            }
            QTabBar::tab:hover:!selected {
                background: #e0e0e0;
            }
        """)
        
        # Import tab modules
        from src.ui.tabs.traces_with_video_tab import TracesWithVideoTab
        from src.ui.tabs.preview_with_video_tab import PreviewWithVideoTab
        from src.ui.tabs.analysis_tab import AnalysisTab
        
        # Create tabs
        self.traces_with_video_tab = TracesWithVideoTab()
        self.preview_with_video_tab = PreviewWithVideoTab()
        self.analysis_tab = AnalysisTab()
        
        # Add tabs
        self.tab_widget.addTab(self.traces_with_video_tab, "Traces")
        self.tab_widget.addTab(self.preview_with_video_tab, "Preview")
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        
        main_layout.addWidget(self.tab_widget)
        
        # Get shared video widget reference from traces tab
        self.video_widget = self.traces_with_video_tab.video_widget
        
        # Share video widget reference with other tabs
        self.preview_with_video_tab.set_video_widget(self.video_widget)
        self.analysis_tab.set_video_widget(self.video_widget)
        
        # Initially disable/dim content until video is loaded
        self._set_content_enabled(False)
        
        # Connect signals
        assert self.video_widget is not None  # Type checker hint
        self.video_widget.video_loaded.connect(self._on_video_loaded)  # type: ignore
        
        # Connect tab changes to handle video widget reparenting
        assert self.tab_widget is not None  # Type checker hint
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        
        self.setCentralWidget(central)
        
        # Show video open dialog on startup after a delay to let the UI load
        QTimer.singleShot(500, self._show_startup_video_dialog)

    def _show_startup_video_dialog(self):
        """Show video open dialog on application startup."""
        self._open_video()
    
    def _open_video(self):
        """Open video file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 File",
            "",
            "HDF5 Files (*.hdf5 *.h5);;All Files (*.*)"
        )
        
        if file_path and self.video_widget:
            self.video_widget.load_video(file_path)  # type: ignore

    def _on_video_loaded(self):
        """Handle video loaded event."""
        # Enable content now that video is loaded
        self._set_content_enabled(True)
        
        # Notify all tabs
        assert self.tab_widget is not None  # Type checker hint
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, 'on_video_loaded'):
                tab.on_video_loaded()  # type: ignore
        
        # Update info dialog if open
        if self.info_dialog and hasattr(self.info_dialog, 'update_info'):
            self.info_dialog.update_info()  # type: ignore
    
    def _set_content_enabled(self, enabled: bool):
        """Enable or disable content areas based on video load status."""
        # Apply to each tab
        assert self.tab_widget is not None  # Type checker hint
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if tab:
                # Disable interaction when no video
                tab.setEnabled(enabled)
                
                # Apply opacity effect to dim content
                if not enabled:
                    opacity_effect = QGraphicsOpacityEffect()
                    opacity_effect.setOpacity(0.3)
                    tab.setGraphicsEffect(opacity_effect)
                else:
                    tab.setGraphicsEffect(None)  # Remove effect when enabled

    def _on_tab_changed(self, index: int):
        """Handle tab change - reparent video widget."""
        if index == 0:  # Traces tab
            self.traces_with_video_tab.attach_video_widget()  # type: ignore
        elif index == 1:  # Preview tab
            self.preview_with_video_tab.attach_video_widget()  # type: ignore
        # Analysis tab (index 2) doesn't need video

    def _show_info_dialog(self):
        """Show info dialog as popup."""
        from src.ui.tabs.info_tab import InfoTab
        
        if not self.info_dialog:
            self.info_dialog = QDialog(self)
            self.info_dialog.setWindowTitle("Measurement Info")
            self.info_dialog.resize(600, 700)
            
            layout = QVBoxLayout(self.info_dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            
            info_tab = InfoTab()
            info_tab.set_video_widget(self.video_widget)
            info_tab.on_video_loaded()
            
            layout.addWidget(info_tab)
            
            # Store reference for updates
            self.info_dialog.info_tab = info_tab  # type: ignore
            self.info_dialog.update_info = info_tab.update_info  # type: ignore
        
        self.info_dialog.show()
        self.info_dialog.raise_()
        self.info_dialog.activateWindow()

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

    def closeEvent(self, event):  # type: ignore
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
                self.video_widget.cleanup()  # type: ignore
            event.accept()
        else:
            event.ignore()
