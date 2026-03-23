"""PyQt6 GUI for HROVER — heart rate video overlay tool."""

import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QTimer,
    Qt,
    QThread,
    pyqtSignal,
    QUrl,
    QSize,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QImage,
    QPalette,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .gpx_parser import parse_gpx_to_timeline
from .hr_data import HRTimeline, HRZoneConfig
from .overlay import OverlayConfig, draw_hr_overlay
from .sync import compute_offset, frame_to_utc, get_video_creation_time
from .video_processor import (
    ENCODERS,
    QUALITY_LABELS,
    RESOLUTIONS,
    EncoderConfig,
    auto_scale_overlay,
    process_video,
)


# ---------------------------------------------------------------------------
# Video player widget
# ---------------------------------------------------------------------------

class VideoPlayerWidget(QWidget):
    """Video display area with transport controls (play/pause, scrub, time)."""

    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fps = 30.0
        self._total_frames = 0
        self._current_frame = 0
        self._playing = False

        # Video display
        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display.setMinimumSize(320, 240)
        self._display.setStyleSheet("background-color: #1a1a1a;")
        self._display.setText("Drag & drop a video file here\nor use File → Open Video")

        # Transport controls
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(40)
        self._play_btn.clicked.connect(self.toggle_playback)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setFixedWidth(120)

        transport = QHBoxLayout()
        transport.addWidget(self._play_btn)
        transport.addWidget(self._slider)
        transport.addWidget(self._time_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._display, 1)
        layout.addLayout(transport)

        # Playback timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)

    def display_size(self) -> QSize:
        return self._display.size()

    def set_video_info(self, fps: float, total_frames: int):
        self._fps = fps
        self._total_frames = total_frames
        self._slider.setRange(0, max(0, total_frames - 1))
        self._update_time_label()

    def set_frame(self, pixmap: QPixmap):
        self._display.setPixmap(pixmap)

    def toggle_playback(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if self._total_frames == 0:
            return
        self._playing = True
        self._play_btn.setText("⏸")
        # Cap display rate at ~30fps
        interval = max(33, int(1000 / self._fps))
        self._timer.start(interval)

    def pause(self):
        self._playing = False
        self._play_btn.setText("▶")
        self._timer.stop()

    def _on_timer_tick(self):
        # Advance frames to match real-time playback
        step = max(1, int(self._fps / 30))
        next_frame = self._current_frame + step
        if next_frame >= self._total_frames:
            self.pause()
            return
        self._slider.blockSignals(True)
        self._slider.setValue(next_frame)
        self._slider.blockSignals(False)
        self._current_frame = next_frame
        self._update_time_label()
        self.frame_changed.emit(next_frame)

    def _on_slider_changed(self, value: int):
        self._current_frame = value
        self._update_time_label()
        self.frame_changed.emit(value)

    def _update_time_label(self):
        cur = self._format_time(self._current_frame)
        total = self._format_time(self._total_frames)
        self._time_label.setText(f"{cur} / {total}")

    def _format_time(self, frame: int) -> str:
        if self._fps <= 0:
            return "00:00"
        secs = int(frame / self._fps)
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------

class SettingsPanel(QWidget):
    """Right-side settings panel with file loading, sync info, and overlay config."""

    settings_changed = pyqtSignal()
    export_requested = pyqtSignal()
    video_file_selected = pyqtSignal(str)
    gpx_file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # --- Files group ---
        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout(files_group)

        # Video
        vid_row = QHBoxLayout()
        self._video_label = QLabel("No video loaded")
        self._video_label.setWordWrap(True)
        vid_btn = QPushButton("Video...")
        vid_btn.setFixedWidth(70)
        vid_btn.clicked.connect(self._browse_video)
        vid_row.addWidget(self._video_label, 1)
        vid_row.addWidget(vid_btn)
        files_layout.addLayout(vid_row)

        # GPX
        gpx_row = QHBoxLayout()
        self._gpx_label = QLabel("No GPX loaded")
        self._gpx_label.setWordWrap(True)
        gpx_btn = QPushButton("GPX...")
        gpx_btn.setFixedWidth(70)
        gpx_btn.clicked.connect(self._browse_gpx)
        gpx_row.addWidget(self._gpx_label, 1)
        gpx_row.addWidget(gpx_btn)
        files_layout.addLayout(gpx_row)

        layout.addWidget(files_group)

        # --- Sync info group ---
        sync_group = QGroupBox("Sync Info")
        sync_layout = QVBoxLayout(sync_group)
        self._sync_video_label = QLabel("Video start: —")
        self._sync_gpx_label = QLabel("GPX range: —")
        self._sync_offset_label = QLabel("Auto offset: —")
        for lbl in (self._sync_video_label, self._sync_gpx_label, self._sync_offset_label):
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px;")
            sync_layout.addWidget(lbl)
        layout.addWidget(sync_group)

        # --- Overlay settings group ---
        overlay_group = QGroupBox("Overlay Settings")
        ol = QVBoxLayout(overlay_group)

        # Max HR
        row = QHBoxLayout()
        row.addWidget(QLabel("Max HR:"))
        self.max_hr_spin = QSpinBox()
        self.max_hr_spin.setRange(100, 230)
        self.max_hr_spin.setValue(190)
        self.max_hr_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        row.addWidget(self.max_hr_spin)
        ol.addLayout(row)

        # Position
        row = QHBoxLayout()
        row.addWidget(QLabel("Position:"))
        self.position_combo = QComboBox()
        self.position_combo.addItems(["bottom-left", "bottom-right", "top-left", "top-right"])
        self.position_combo.currentTextChanged.connect(lambda: self.settings_changed.emit())
        row.addWidget(self.position_combo)
        ol.addLayout(row)

        # Opacity
        row = QHBoxLayout()
        row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(lambda: self.settings_changed.emit())
        self._opacity_value = QLabel("0.70")
        self._opacity_value.setFixedWidth(35)
        self.opacity_slider.valueChanged.connect(
            lambda v: self._opacity_value.setText(f"{v / 100:.2f}")
        )
        row.addWidget(self.opacity_slider)
        row.addWidget(self._opacity_value)
        ol.addLayout(row)

        # BPM font size
        row = QHBoxLayout()
        row.addWidget(QLabel("BPM font size:"))
        self.bpm_font_scale_spin = QDoubleSpinBox()
        self.bpm_font_scale_spin.setRange(0.5, 5.0)
        self.bpm_font_scale_spin.setSingleStep(0.1)
        self.bpm_font_scale_spin.setDecimals(1)
        self.bpm_font_scale_spin.setValue(1.8)
        self.bpm_font_scale_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        row.addWidget(self.bpm_font_scale_spin)
        ol.addLayout(row)

        # Show graph
        self.show_graph_check = QCheckBox("Show HR graph")
        self.show_graph_check.setChecked(True)
        self.show_graph_check.stateChanged.connect(lambda: self.settings_changed.emit())
        self.show_graph_check.stateChanged.connect(self._on_show_graph_changed)
        ol.addWidget(self.show_graph_check)

        # Graph duration
        self._graph_duration_row = QHBoxLayout()
        self._graph_duration_row.addWidget(QLabel("Graph (s):"))
        self.graph_duration_spin = QSpinBox()
        self.graph_duration_spin.setRange(10, 300)
        self.graph_duration_spin.setValue(60)
        self.graph_duration_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        self._graph_duration_row.addWidget(self.graph_duration_spin)
        ol.addLayout(self._graph_duration_row)

        # Offset adjustment
        row = QHBoxLayout()
        row.addWidget(QLabel("Offset (s):"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-3600.0, 3600.0)
        self.offset_spin.setSingleStep(0.5)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setSuffix(" s")
        self.offset_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        row.addWidget(self.offset_spin)
        ol.addLayout(row)

        # Show map
        self.show_map_check = QCheckBox("Show map")
        self.show_map_check.setChecked(True)
        self.show_map_check.stateChanged.connect(lambda: self.settings_changed.emit())
        ol.addWidget(self.show_map_check)

        layout.addWidget(overlay_group)

        # --- Export group ---
        export_group = QGroupBox("Export")
        el = QVBoxLayout(export_group)

        # Encoder
        row = QHBoxLayout()
        row.addWidget(QLabel("Encoder:"))
        self.encoder_combo = QComboBox()
        for key, info in ENCODERS.items():
            self.encoder_combo.addItem(info["label"], userData=key)
        el.addLayout(row)
        row.addWidget(self.encoder_combo)

        # Quality
        row = QHBoxLayout()
        row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        for key, label in QUALITY_LABELS.items():
            self.quality_combo.addItem(label, userData=key)
        medium_index = self.quality_combo.findData("medium")
        if medium_index != -1:
            self.quality_combo.setCurrentIndex(medium_index)  # default: medium
        row.addWidget(self.quality_combo)
        el.addLayout(row)

        # Resolution
        row = QHBoxLayout()
        row.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        for key, info in RESOLUTIONS.items():
            self.resolution_combo.addItem(info["label"], userData=key)
        row.addWidget(self.resolution_combo)
        el.addLayout(row)

        self._export_btn = QPushButton("Export Video")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(lambda: self.export_requested.emit())
        el.addWidget(self._export_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        el.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        el.addWidget(self._status_label)

        layout.addWidget(export_group)

        layout.addStretch()

    # --- Public methods ---

    def set_video_path(self, path: str):
        self._video_label.setText(Path(path).name)

    def set_gpx_path(self, path: str):
        self._gpx_label.setText(Path(path).name)

    def set_sync_info(self, video_start, gpx_start, gpx_end, offset_seconds: float):
        self._sync_video_label.setText(
            f"Video start: {video_start.strftime('%Y-%m-%d %H:%M:%S') if video_start else '—'}"
        )
        self._sync_gpx_label.setText(
            f"GPX: {gpx_start.strftime('%H:%M:%S')} → {gpx_end.strftime('%H:%M:%S')}"
        )
        self._sync_offset_label.setText(f"Auto offset: {offset_seconds:+.1f}s")

    def set_export_enabled(self, enabled: bool):
        self._export_btn.setEnabled(enabled)

    def set_progress(self, current: int, total: int):
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    def set_status(self, text: str):
        self._status_label.setText(text)
        self._progress_bar.setVisible(False)

    def _on_show_graph_changed(self):
        enabled = self.show_graph_check.isChecked()
        self.graph_duration_spin.setEnabled(enabled)

    def get_overlay_config(self) -> OverlayConfig:
        return OverlayConfig(
            position=self.position_combo.currentText(),
            opacity=self.opacity_slider.value() / 100.0,
            bpm_font_scale=self.bpm_font_scale_spin.value(),
            show_graph=self.show_graph_check.isChecked(),
            graph_duration=float(self.graph_duration_spin.value()),
            show_map=self.show_map_check.isChecked(),
        )

    def get_zone_config(self) -> HRZoneConfig:
        return HRZoneConfig(max_hr=self.max_hr_spin.value())

    def get_manual_offset(self) -> float:
        return self.offset_spin.value()

    def get_encoder_config(self) -> EncoderConfig:
        return EncoderConfig(
            encoder=self.encoder_combo.currentData(),
            quality=self.quality_combo.currentData(),
            resolution=self.resolution_combo.currentData(),
        )

    # --- Private methods ---

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if path:
            self.video_file_selected.emit(path)

    def _browse_gpx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GPX",
            "", "GPX Files (*.gpx);;All Files (*)"
        )
        if path:
            self.gpx_file_selected.emit(path)


# ---------------------------------------------------------------------------
# Processing thread
# ---------------------------------------------------------------------------

class ProcessingThread(QThread):
    """Runs video processing in a background thread."""

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path, output_path, hr_timeline, video_start,
                 offset, overlay_config, zone_config, encoder_config=None):
        super().__init__()
        self._video_path = video_path
        self._output_path = output_path
        self._hr_timeline = hr_timeline
        self._video_start = video_start
        self._offset = offset
        self._overlay_config = replace(overlay_config)  # fresh copy
        self._zone_config = zone_config
        self._encoder_config = encoder_config

    def run(self):
        try:
            result = process_video(
                video_path=self._video_path,
                output_path=self._output_path,
                hr_timeline=self._hr_timeline,
                video_start=self._video_start,
                offset=self._offset,
                overlay_config=self._overlay_config,
                zone_config=self._zone_config,
                progress_callback=self._on_progress,
                encoder_config=self._encoder_config,
            )
            self.finished_ok.emit(str(result))
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, current: int, total: int):
        self.progress.emit(current, total)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class HROverMainWindow(QMainWindow):
    """Main application window."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HROVER — Heart Rate Video Overlay")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        # State
        self._cap = None
        self._fps = 30.0
        self._total_frames = 0
        self._video_width = 0
        self._video_height = 0
        self._video_path = None
        self._gpx_path = None
        self._hr_timeline = None
        self._video_start = None
        self._auto_offset = timedelta(0)
        self._processing_thread = None

        # UI
        self._video_player = VideoPlayerWidget()
        self._settings = SettingsPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._video_player)

        # Wrap settings in a scroll area
        scroll = QScrollArea()
        scroll.setWidget(self._settings)
        scroll.setWidgetResizable(True)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(splitter)

        # Menu bar
        self._setup_menu()

        # Connections
        self._video_player.frame_changed.connect(self._on_frame_changed)
        self._settings.settings_changed.connect(self._refresh_frame)
        self._settings.export_requested.connect(self._start_export)
        self._settings.video_file_selected.connect(self._load_video)
        self._settings.gpx_file_selected.connect(self._load_gpx)

    def _setup_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        open_video = QAction("Open &Video...", self)
        open_video.triggered.connect(
            lambda: self._settings._browse_video()
        )
        file_menu.addAction(open_video)

        open_gpx = QAction("Open &GPX...", self)
        open_gpx.triggered.connect(
            lambda: self._settings._browse_gpx()
        )
        file_menu.addAction(open_gpx)

        file_menu.addSeparator()

        export_action = QAction("&Export...", self)
        export_action.triggered.connect(self._start_export)
        file_menu.addAction(export_action)

    # --- File loading ---

    def _load_video(self, path: str):
        if self._cap is not None:
            self._cap.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", f"Cannot open video:\n{path}")
            return

        self._cap = cap
        self._video_path = path
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._video_player.set_video_info(self._fps, self._total_frames)
        self._settings.set_video_path(path)

        # Try to get creation time for sync
        self._video_start = get_video_creation_time(path)
        self._update_sync_info()
        self._check_export_ready()
        self._refresh_frame()

    def _load_gpx(self, path: str):
        try:
            self._hr_timeline = parse_gpx_to_timeline(path)
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        self._gpx_path = path
        self._settings.set_gpx_path(path)
        self._update_sync_info()
        self._check_export_ready()
        self._refresh_frame()

    def _update_sync_info(self):
        if self._hr_timeline is None:
            return
        if self._video_start is not None:
            self._auto_offset = compute_offset(self._hr_timeline.start_time, self._video_start)
        else:
            # No video timestamp — align GPX start to video start (offset = 0)
            self._auto_offset = timedelta(0)
        self._settings.set_sync_info(
            self._video_start,
            self._hr_timeline.start_time,
            self._hr_timeline.end_time,
            self._auto_offset.total_seconds(),
        )

    def _check_export_ready(self):
        ready = self._video_path is not None and self._hr_timeline is not None
        self._settings.set_export_enabled(ready)

    # --- Frame rendering ---

    def _on_frame_changed(self, frame_idx: int):
        self._refresh_frame_at(frame_idx)

    def _refresh_frame(self):
        """Re-render the current frame with current settings."""
        if self._cap is not None:
            current = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            # If we haven't read any frame yet, start at 0
            frame_idx = max(0, current - 1)
            self._refresh_frame_at(frame_idx)

    def _refresh_frame_at(self, frame_idx: int):
        if self._cap is None:
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return

        # Apply overlay if GPX is loaded
        if self._hr_timeline is not None:
            overlay_config = self._settings.get_overlay_config()
            zone_config = self._settings.get_zone_config()
            auto_scale_overlay(overlay_config, self._video_width, self._video_height)

            manual_offset = self._settings.get_manual_offset()
            total_offset = self._auto_offset + timedelta(seconds=manual_offset)

            video_start = self._video_start or self._hr_timeline.start_time
            video_time = frame_to_utc(frame_idx, self._fps, video_start)
            gpx_time = video_time + total_offset

            bpm = self._hr_timeline.get_bpm_at(gpx_time)
            hr_window = self._hr_timeline.get_window(gpx_time, overlay_config.graph_duration)
            trace_start = video_start + total_offset
            trace_end = trace_start + timedelta(seconds=self._total_frames / self._fps)
            frame = draw_hr_overlay(
                frame, bpm, hr_window, zone_config, overlay_config,
                gps_timeline=self._hr_timeline.gps,
                current_time=gpx_time,
                trace_start=trace_start,
                trace_end=trace_end,
            )

        # BGR → RGB → QImage → QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit display
        display_size = self._video_player.display_size()
        scaled = pixmap.scaled(
            display_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_player.set_frame(scaled)

    # --- Export ---

    def _start_export(self):
        if self._video_path is None or self._hr_timeline is None:
            QMessageBox.information(self, "Not Ready", "Load both a video and GPX file first.")
            return

        # Ask for output path
        default_name = Path(self._video_path).with_stem(
            Path(self._video_path).stem + "_hr"
        ).with_suffix(".mp4")
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video",
            str(default_name),
            "MP4 Video (*.mp4);;All Files (*)"
        )
        if not output_path:
            return

        manual_offset = self._settings.get_manual_offset()
        total_offset = self._auto_offset + timedelta(seconds=manual_offset)
        overlay_config = self._settings.get_overlay_config()
        zone_config = self._settings.get_zone_config()
        encoder_config = self._settings.get_encoder_config()

        self._settings.set_status("Exporting...")
        self._settings.set_export_enabled(False)

        video_start = self._video_start or self._hr_timeline.start_time
        self._processing_thread = ProcessingThread(
            video_path=self._video_path,
            output_path=output_path,
            hr_timeline=self._hr_timeline,
            video_start=video_start,
            offset=total_offset,
            overlay_config=overlay_config,
            zone_config=zone_config,
            encoder_config=encoder_config,
        )
        self._processing_thread.progress.connect(self._settings.set_progress)
        self._processing_thread.finished_ok.connect(self._on_export_done)
        self._processing_thread.error.connect(self._on_export_error)
        self._processing_thread.start()

    def _on_export_done(self, path: str):
        self._settings.set_status(f"Done! Saved to:\n{Path(path).name}")
        self._settings.set_export_enabled(True)

    def _on_export_error(self, msg: str):
        self._settings.set_status(f"Error: {msg}")
        self._settings.set_export_enabled(True)
        QMessageBox.critical(self, "Export Error", msg)

    # --- Drag & Drop ---

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            suffix = Path(path).suffix.lower()
            if suffix == ".gpx":
                self._load_gpx(path)
            elif suffix in self.VIDEO_EXTENSIONS:
                self._load_video(path)

    # --- Cleanup ---

    def closeEvent(self, event):
        if self._cap is not None:
            self._cap.release()
        if self._processing_thread is not None and self._processing_thread.isRunning():
            self._processing_thread.wait(3000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------

def _apply_dark_theme(app: QApplication):
    """Apply a dark Fusion theme for a modern look."""
    app.setStyle("Fusion")
    palette = QPalette()

    dark = QColor(53, 53, 53)
    darker = QColor(35, 35, 35)
    text = QColor(220, 220, 220)
    highlight = QColor(42, 130, 218)
    disabled_text = QColor(128, 128, 128)

    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, darker)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, dark)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, highlight)
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)

    app.setPalette(palette)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch_gui():
    """Launch the HROVER GUI application."""
    app = QApplication(sys.argv)
    _apply_dark_theme(app)

    window = HROverMainWindow()
    window.show()

    sys.exit(app.exec())
