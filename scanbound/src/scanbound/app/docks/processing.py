"""Display (levels, gamma, colormap) and processing chain of the current image."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, QTimer, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget

from ...processing import ProcessingChain
from ..session import Session
from ..widgets.display_controls import DisplayControls, fill_colormap_combo
from ..widgets.spm_processing import ProcessingPanel
from ..theme import mark


class DisplayTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.cmap = QComboBox()
        fill_colormap_combo(self.cmap)
        self.cmap.currentIndexChanged.connect(self._on_cmap)
        self.controls = DisplayControls()
        self.controls.changed.connect(self._on_changed)
        apply_all = QPushButton("Use for all images")
        apply_all.setToolTip("Levels are relative to each image's range, so the same settings give a comparable look")
        apply_all.clicked.connect(self._apply_all)
        row = QHBoxLayout()
        row.addWidget(QLabel("Colormap"))
        row.addWidget(self.cmap, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(row)
        lay.addWidget(self.controls, 1)
        lay.addWidget(apply_all)
        self._loading = False
        session.imageChanged.connect(lambda _i: self.load())
        session.imageDataChanged.connect(lambda i: i == session.current_id and self.load())

    def load(self) -> None:
        e = self.session.entry
        if e is None:
            return
        self._loading = True
        try:
            self.controls.settings = e.display.copy()
            try:
                stats = self.session.display_stats()
                img = self.session.image()
            except Exception:  # noqa: BLE001 - the canvas reports it
                return
            self.controls.set_image(stats, img.unit if img is not None else "")
            k = self.cmap.findData(e.display.cmap)
            if k < 0:
                k = self.cmap.findText(e.display.cmap)
            if k >= 0:
                self.cmap.setCurrentIndex(k)
        finally:
            self._loading = False

    def _cmap_name(self) -> str:
        return self.cmap.currentData() or self.cmap.currentText()

    def _on_cmap(self, *_):
        if self._loading:
            return
        self.controls.set_cmap(self._cmap_name())
        self._on_changed(True)

    def _on_changed(self, final: bool) -> None:
        if self._loading or self.session.entry is None:
            return
        self.session.set_display(self.controls.settings, final)

    def _apply_all(self) -> None:
        ids = [self.session.project.entry(i).id for i in self.session.image_ids()]
        self.session.set_display(self.controls.settings, True, ids)
        self.session.info(f"Display settings applied to {len(ids)} image(s)")


class ProcessingTab(QWidget):
    fourierEditRequested = pyqtSignal(object)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.panel = ProcessingPanel()
        self.panel.changed.connect(self._on_changed)
        self.panel.applyAllRequested.connect(self._apply_all)
        self.panel.fourierEditRequested.connect(self.fourierEditRequested)
        self.keep = QCheckBox("Use these steps for every image I open")
        self.keep.setToolTip("On: switching images applies the current steps to the new image.\n"
                             "Off: every image keeps its own steps.")
        self.keep.setChecked(bool(QSettings("ScanBound", "ScanBound").value("keep_processing", False, type=bool)))
        self.keep.toggled.connect(lambda on: QSettings("ScanBound", "ScanBound").setValue("keep_processing", on))
        self.note = QLabel("Each image keeps its own steps. They are saved with the project and recorded in every "
                           "result, so processed images can be regenerated from the raw file.")
        self.note.setWordWrap(True)
        mark(self.note, "muted")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.panel, 1)
        lay.addWidget(self.keep)
        lay.addWidget(self.note)
        self._loading = False
        session.imageChanged.connect(lambda _i: self.load())

    def chain(self) -> ProcessingChain:
        return ProcessingChain.from_dict(self.panel.to_dict())

    def load(self) -> None:
        e = self.session.entry
        if e is None:
            return
        if self.keep.isChecked() and self.panel.cards:
            new = self.chain()                      # carry the current steps over to this image
            if new.to_dict() != e.chain.to_dict():
                self.session.set_chain(new)
            return
        self._loading = True
        try:
            if e.chain.steps:
                self.panel.set_dict(e.chain.to_dict())
            else:
                self.panel.reset(emit=False)
        finally:
            QTimer.singleShot(80, self._done_loading)       # after the panel's debounce timer

    def _done_loading(self) -> None:
        self._loading = False

    def _on_changed(self) -> None:
        if self._loading or self.session.entry is None:
            return
        new = self.chain()
        if new.to_dict() != self.session.entry.chain.to_dict():
            self.session.set_chain(new)

    def _apply_all(self) -> None:
        ids = [self.session.project.entry(i).id for i in self.session.image_ids()]
        self.session.set_chain(self.chain(), ids)
        self.session.info(f"Processing applied to {len(ids)} image(s): {self.chain().summary() or 'no steps'}")


class ProcessingDock(QTabWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.display = DisplayTab(session)
        self.processing = ProcessingTab(session)
        self.addTab(self.display, "Display")
        self.addTab(self.processing, "Processing")
