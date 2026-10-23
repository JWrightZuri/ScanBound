"""
The app's state: the open project, the current image, the selection, the active label.

The model (Project, AnnotationStore, ProcessingChain ...) knows nothing about Qt.  ``Session`` turns
its changes into Qt signals, so every view (canvas, docks, dialogs) listens to one object and
never to another widget.  All edits go through the store or through ``Session`` methods; views
only render.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from ..annotations import Annotation, StoreEvent
from ..display import DisplaySettings, ImageStats as DisplayStats
from ..model import Image
from ..processing import ProcessingChain
from ..project import DEFAULT_NAME, Project


class Session(QObject):
    projectChanged = pyqtSignal()                 # a different project (or its file list) is loaded
    imageChanged = pyqtSignal(str)                # the current image id ("" = none)
    imageDataChanged = pyqtSignal(str)            # chain / calibration of an image changed (pixels may differ)
    displayChanged = pyqtSignal(str, bool)        # display settings of an image changed (id, final)
    annotationsChanged = pyqtSignal(object)       # StoreEvent
    selectionChanged = pyqtSignal()
    labelsChanged = pyqtSignal()
    currentLabelChanged = pyqtSignal(object)      # str or None
    dirtyChanged = pyqtSignal(bool)
    message = pyqtSignal(str, str)                # (text, level: "info" / "warning" / "error")

    def __init__(self, project: Optional[Project] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.project: Project = project or Project(os.getcwd())
        self.current_id: str = ""
        self.channel: Optional[str] = None        # the channel last chosen; kept while moving between files
        self.selection: list[str] = []
        self.current_label: Optional[str] = None
        self._stats: dict[str, DisplayStats] = {}
        self._unsubscribe = None
        self._was_dirty = False
        self._attach()

    # ------------------------------------------------------------------ project
    def _attach(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
        self._unsubscribe = self.project.annotations.subscribe(self._on_store)
        from .features import REVIEW
        store = self.project.annotations
        store.auto_accept = not REVIEW
        if not REVIEW:                                  # proposals from earlier sessions: accepted as they are
            pending = [a.id for a in store if a.status == "proposed"]
            if pending:
                store.accept(pending)
                store.clear_history()
        self._stats.clear()
        self._was_dirty = self.project.dirty

    def set_project(self, project: Project) -> None:
        self.project = project
        self.current_id = ""
        self.selection = []
        self.project.labels.merge(a.label for a in project.annotations)
        self._attach()
        self.projectChanged.emit()
        self.labelsChanged.emit()
        self._check_dirty()
        ids = self.image_ids()
        self.set_current(ids[0] if ids else "")

    def open_folder(self, folder: str, recursive: bool = False, project: Optional[Project] = None) -> None:
        """Open a data folder (``project``: already scanned, e.g. in a background thread)."""
        p = project or Project.open_folder(folder, recursive or bool(self._saved_recursive(folder)))
        if recursive:
            p.settings["recursive"] = True
        self.set_project(p)
        n = len(self.project.files)
        saved = os.path.isfile(os.path.join(folder, DEFAULT_NAME))
        self.info(f"Opened {folder}: {n} data file(s)" + (" (with its saved project)" if saved else ""))

    @staticmethod
    def saved_recursive(folder: str) -> bool:
        return Session._saved_recursive(folder)

    @staticmethod
    def _saved_recursive(folder: str) -> bool:
        path = os.path.join(folder, DEFAULT_NAME)
        if not os.path.isfile(path):
            return False
        try:
            return bool(Project.load(path).settings.get("recursive"))
        except Exception:  # noqa: BLE001 - a broken project file is reported when it is opened
            return False

    def open_project(self, path: str, project: Optional[Project] = None) -> None:
        p = project
        if p is None:
            p = Project.load(path)
            p.scan(bool(p.settings.get("recursive")))
        self.set_project(p)
        self.info(f"Opened project {path}")

    def save(self, path: Optional[str] = None) -> str:
        out = self.project.save(path)
        self._check_dirty()
        self.info(f"Saved {out}")
        return out

    def rescan(self) -> list[str]:
        new = self.project.scan()
        if new:
            self.projectChanged.emit()
            self.info(f"{len(new)} new file(s)")
        return new

    def image_ids(self) -> list[str]:
        """One id per data file (its default channel); channels are switched in the viewer."""
        return [rel + "::" for rel, f in self.project.files.items() if self.is_image_file(rel)]

    def is_image_file(self, rel: str) -> bool:
        f = self.project.files.get(rel)
        if f is None:
            return False
        if f.kind:
            return f.kind in ("scan", "image")
        return not rel.lower().endswith(".dat")

    # ------------------------------------------------------------------ messages
    def info(self, text: str) -> None:
        self.message.emit(text, "info")

    def warn(self, text: str) -> None:
        self.message.emit(text, "warning")

    def error(self, text: str) -> None:
        self.message.emit(text, "error")

    def _check_dirty(self) -> None:
        d = self.project.dirty
        if d != self._was_dirty:
            self._was_dirty = d
            self.dirtyChanged.emit(d)

    # ------------------------------------------------------------------ current image
    def set_current(self, image_id: str) -> None:
        """``image_id`` may be "file::" (the channel last chosen if the file has it, else its default channel)
        or "file::channel"."""
        if image_id:
            rel, ch = self.project.split_image_id(image_id)
            if not ch and self.channel and "::" in image_id:
                try:
                    if self.channel in self.channels(rel + "::"):
                        image_id = self.project.make_image_id(rel, self.channel)
                except Exception:  # noqa: BLE001 - reported below when the file is opened
                    pass
            try:
                image_id = self.project.entry(image_id).id
            except Exception as exc:  # noqa: BLE001 - unreadable file
                self.error(f"Cannot open {image_id}: {exc}")
                return
        if image_id == self.current_id:
            return
        self.current_id = image_id
        self.selection = []
        self.imageChanged.emit(image_id)
        self.selectionChanged.emit()

    @property
    def entry(self):
        return self.project.entry(self.current_id) if self.current_id else None

    def image(self, image_id: Optional[str] = None) -> Optional[Image]:
        """The processed image (raw data through its chain)."""
        iid = image_id or self.current_id
        if not iid:
            return None
        return self.project.processed_image(iid)

    def raw_image(self, image_id: Optional[str] = None) -> Optional[Image]:
        iid = image_id or self.current_id
        return self.project.raw_image(iid) if iid else None

    def display_stats(self, image_id: Optional[str] = None) -> Optional[DisplayStats]:
        img = self.image(image_id)
        if img is None:
            return None
        key = img.digest
        if key not in self._stats:
            if len(self._stats) > 64:
                self._stats.clear()
            self._stats[key] = DisplayStats(img.data)
        return self._stats[key]

    def channels(self, image_id: Optional[str] = None) -> list[str]:
        iid = image_id or self.current_id
        if not iid:
            return []
        rel, _ = self.project.split_image_id(iid)
        return [c.key for c in self.project.data_file(rel).channels]

    def set_channel(self, channel: str) -> None:
        self.channel = channel or None
        if self.current_id:
            rel, _ = self.project.split_image_id(self.current_id)
            self.set_current(self.project.make_image_id(rel, channel))

    def set_chain(self, chain: ProcessingChain, image_ids: Optional[Iterable[str]] = None) -> None:
        ids = list(image_ids) if image_ids is not None else [self.current_id]
        for i in ids:
            if i:
                self.project.set_chain(i, chain.copy())
                self.imageDataChanged.emit(i)
        self._check_dirty()

    def set_display(self, display: DisplaySettings, final: bool = True, image_ids: Optional[Iterable[str]] = None) -> None:
        ids = list(image_ids) if image_ids is not None else [self.current_id]
        for i in ids:
            if i:
                self.project.set_display(i, display.copy())
                self.displayChanged.emit(i, final)
        self._check_dirty()

    def set_calibration(self, dx: Optional[float], dy: Optional[float] = None, rel_uncertainty: float = 0.0) -> None:
        if not self.current_id:
            return
        self.project.set_calibration(self.current_id, dx, dy, rel_uncertainty)
        self.imageDataChanged.emit(self.current_id)
        self._check_dirty()

    def set_reviewed(self, image_id: str, reviewed: bool = True) -> None:
        self.project.entry(image_id).reviewed = reviewed
        self.project.dirty = True
        self.imageDataChanged.emit(image_id)
        self._check_dirty()

    # ------------------------------------------------------------------ annotations
    @property
    def store(self):
        return self.project.annotations

    def annotations(self, image_id: Optional[str] = None) -> list[Annotation]:
        iid = image_id or self.current_id
        return self.store.for_image(iid) if iid else []

    def _on_store(self, event: StoreEvent) -> None:
        new_labels = [c.after.label for c in event.changes if c.after is not None and c.after.label
                      and c.after.label not in self.project.labels]
        if new_labels:
            self.project.labels.merge(new_labels)
            self.labelsChanged.emit()
        removed = set(event.ids("removed"))
        if removed & set(self.selection):
            self.selection = [i for i in self.selection if i not in removed]
            self.selectionChanged.emit()
        self.annotationsChanged.emit(event)
        self._check_dirty()

    def select(self, ids: Iterable[str], add: bool = False, toggle: bool = False) -> None:
        ids = [i for i in ids if i in self.store]
        if toggle:
            cur = list(self.selection)
            for i in ids:
                cur.remove(i) if i in cur else cur.append(i)
            new = cur
        elif add:
            new = list(dict.fromkeys(self.selection + ids))
        else:
            new = list(dict.fromkeys(ids))
        if new != self.selection:
            self.selection = new
            self.selectionChanged.emit()

    def selected(self) -> list[Annotation]:
        return [self.store[i] for i in self.selection if i in self.store]

    def set_current_label(self, label: Optional[str]) -> None:
        if label is not None and label not in self.project.labels:
            self.project.labels.add(label)
            self.labelsChanged.emit()
        if label != self.current_label:
            self.current_label = label
            self.currentLabelChanged.emit(label)

    def label_color(self, label: Optional[str]) -> str:
        return self.project.labels.color(label)

    def labels_edited(self) -> None:
        """Call after changing ``project.labels`` directly."""
        self.project.dirty = True
        self.labelsChanged.emit()
        self._check_dirty()


class _Entry:
    def __init__(self, image_id: str, display: DisplaySettings):
        self.id = image_id
        self.display = display
        self.chain = ProcessingChain()
        self.reviewed = False


class StandaloneSession(Session):
    """A session around one in-memory image (no project files), e.g. for drawing regions on a
    Fourier transform.  Annotations live in the session's own store."""

    def __init__(self, image: Image, display: Optional[DisplaySettings] = None, parent: Optional[QObject] = None):
        super().__init__(Project(os.getcwd()), parent)
        self._image = image
        self._entry = _Entry(image.id, display or DisplaySettings())
        self.current_id = image.id

    def set_current(self, image_id: str) -> None:
        pass

    @property
    def entry(self):
        return self._entry

    def image(self, image_id: Optional[str] = None) -> Optional[Image]:
        return self._image

    def raw_image(self, image_id: Optional[str] = None) -> Optional[Image]:
        return self._image

    def set_image(self, image: Image) -> None:
        self._image = image
        self.imageDataChanged.emit(self.current_id)

    def set_display(self, display: DisplaySettings, final: bool = True, image_ids=None) -> None:  # noqa: ANN001
        self._entry.display = display.copy()
        self.displayChanged.emit(self.current_id, final)
