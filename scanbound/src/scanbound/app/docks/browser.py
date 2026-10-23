"""Dataset browser: the project's data files as a folder tree, each with a thumbnail, size, quality and review state."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPoint, QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QToolButton,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...display import DisplaySettings, ImageStats as DisplayStats, render
from ...io import open_data
from ..features import REVIEW
from ..qtutil import run_job, to_qimage
from ..session import Session

THUMB = 56
COLS = ["File"]       # each entry: the name over its size, quality and annotation counts


def make_thumbnail(path: str) -> dict:
    """Runs in a worker thread: open the file, render a small preview, collect quality flags."""
    f = open_data(path)
    out: dict = {"kind": f.kind, "format": f.format, "warnings": list(f.warnings)}
    if f.kind == "spectrum":
        sp = f.spectrum()
        out["text"] = f"{len(sp.axis.values)} pts"
        return out
    img = f.image()
    h, w = img.shape
    step = max(1, int(max(h, w) // (THUMB * 2)))
    small = np.asarray(img.data[::step, ::step], float)
    rgb = render(small, DisplayStats(small), DisplaySettings())
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    out["thumb"] = np.ascontiguousarray(rgb)
    out["text"] = f"{w} × {h}"
    out["quality"] = [(q.level, q.code, q.message) for q in img.quality()]
    out["channels"] = len(f.channels)
    return out


FOLDER_ROLE = Qt.ItemDataRole.UserRole + 4      # folder items: their relative path
FAV_ROLE = Qt.ItemDataRole.UserRole + 5         # file items: starred
SORTS = (("name", "Name"), ("date", "Date modified"), ("type", "Type"), ("size", "File size"),
         ("annotated", "Annotations")) + ((("review", "To review"),) if REVIEW else ())
MAX_THUMB_JOBS = 3                              # thumbnails rendered at the same time
EXPAND_ALL_BELOW = 400                          # with more files, folders start collapsed


class BrowserDock(QWidget):
    """Files of the project as a folder tree.  Thumbnails load in the background, the visible rows
    first, a few at a time, so a folder with thousands of files stays responsive."""
    spectrumActivated = pyqtSignal(str)          # relative path of a spectroscopy file

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter files…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        self.only_todo = QCheckBox("Needs review")
        self.only_todo.setToolTip("Only files with proposals that are not reviewed yet")
        self.only_todo.toggled.connect(self._apply_filter)
        self.only_todo.setVisible(REVIEW)
        self.only_fav = QCheckBox("★")
        self.only_fav.setToolTip("Only favourite files (right-click a file, or press F2, to star it)")
        self.only_fav.toggled.connect(self._apply_filter)
        settings = QSettings("ScanBound", "ScanBound")
        self.sort = QComboBox()
        for key, text in SORTS:
            self.sort.addItem(text, key)
        self.sort.setToolTip("Sort the files in every folder")
        k = self.sort.findData(settings.value("browser_sort", "name"))
        self.sort.setCurrentIndex(max(k, 0))
        self.sort_desc = QToolButton()
        self.sort_desc.setCheckable(True)
        self.sort_desc.setChecked(settings.value("browser_desc", False, type=bool))
        self.sort_desc.setText("↓" if self.sort_desc.isChecked() else "↑")
        self.sort_desc.setToolTip("Ascending / descending")
        self.sort.currentIndexChanged.connect(self._resort)
        self.sort_desc.toggled.connect(self._resort)
        self._cache: dict[str, dict] = {}
        self._cache_root = ""
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(COLS)
        self.tree.setRootIsDecorated(True)
        self.tree.setIconSize(QSize(THUMB, THUMB))
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(14)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._on_current)
        self.tree.itemActivated.connect(lambda it, _c: self._activate(it))
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        star = QShortcut(QKeySequence("F2"), self.tree, context=Qt.ShortcutContext.WidgetShortcut)
        star.activated.connect(self.toggle_favourite)
        self.tree.itemExpanded.connect(lambda _it: self._schedule_thumbs())
        self.tree.verticalScrollBar().valueChanged.connect(lambda _v: self._schedule_thumbs())
        expand = QToolButton()
        expand.setText("⊞")
        expand.setToolTip("Expand all folders")
        expand.clicked.connect(self.tree.expandAll)
        collapse = QToolButton()
        collapse.setText("⊟")
        collapse.setToolTip("Collapse all folders")
        collapse.clicked.connect(self.tree.collapseAll)
        top = QHBoxLayout()
        top.addWidget(self.filter, 1)
        top.addWidget(self.only_fav)
        top.addWidget(self.only_todo)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Sort"))
        row2.addWidget(self.sort, 1)
        row2.addWidget(self.sort_desc)
        row2.addWidget(expand)
        row2.addWidget(collapse)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addLayout(row2)
        lay.addWidget(self.tree, 1)
        self.items: dict[str, QTreeWidgetItem] = {}       # file rel -> item, in tree order
        self.folders: dict[str, QTreeWidgetItem] = {}
        self._loaded: set[str] = set()                     # thumbnails requested
        self._running = 0
        self._generation = 0
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(60)
        self._thumb_timer.timeout.connect(self._start_thumbs)
        session.projectChanged.connect(lambda: self.rebuild())
        session.imageChanged.connect(self._on_image_changed)
        session.annotationsChanged.connect(lambda ev: self._update_counts({i.rpartition("::")[0] for i in ev.image_ids}))
        session.imageDataChanged.connect(lambda i: self._update_counts({i.rpartition("::")[0]}))

    # ------------------------------------------------------------------ building
    def rebuild(self, keep_thumbnails: bool = False) -> None:
        """Rebuild the tree; ``keep_thumbnails`` reuses the ones already rendered (re-sorting)."""
        self._generation += 1
        self._running = 0
        if not keep_thumbnails or self._cache_root != self.session.project.root:
            self._cache = {}
            self._cache_root = self.session.project.root
        self._loaded = set(self._cache)
        current = self.session.current_id.rpartition("::")[0]
        self.tree.clear()
        self.items = {}
        self.folders = {}
        p = self.session.project
        favourites = self.favourites()
        counts = self._counts(set(p.files))
        for rel in self._ordered(list(p.files), counts):
            parent = self._folder(rel.rpartition("/")[0])
            it = QTreeWidgetItem([rel.rpartition("/")[2]])
            it.setSizeHint(0, QSize(THUMB, THUMB + 4))
            it.setData(0, Qt.ItemDataRole.UserRole, rel)
            it.setData(0, FAV_ROLE, rel in favourites)
            e = p.files[rel]
            if parent is None:
                self.tree.addTopLevelItem(it)
            else:
                parent.addChild(it)
            self.items[rel] = it
            if rel in self._cache:
                self._apply_thumb(it, self._cache[rel])
            elif e.error:
                self._set_detail(it, "unreadable", e.error, Qt.GlobalColor.red)
            else:
                self._set_detail(it, "")
        per_folder: dict[str, int] = {}
        for rel in self.items:
            d = rel.rpartition("/")[0]
            while d:
                per_folder[d] = per_folder.get(d, 0) + 1
                d = d.rpartition("/")[0]
        for path, f in self.folders.items():
            f.setText(0, f"📁 {path.rpartition('/')[2]}   ({per_folder.get(path, 0)})")
        if len(self.items) < EXPAND_ALL_BELOW:
            self.tree.expandAll()
        self._update_counts(set(self.items), counts)
        if current in self.items:
            self._on_image_changed(self.session.current_id)
        self._schedule_thumbs()

    # ------------------------------------------------------------------ sorting and favourites
    def _ordered(self, rels: list[str], counts: dict) -> list[str]:
        """Files in tree order: at every level the subfolders (by name) first, then the files sorted by
        the chosen key; folders always stay in name order."""
        mode = self.sort.currentData()
        desc = self.sort_desc.isChecked()
        files = self.session.project.files

        def key(rel: str):  # noqa: ANN202
            name = rel.rpartition("/")[2].lower()
            e = files[rel]
            gt, prop = counts.get(rel, (0, 0))
            return {"name": (name,), "date": (e.mtime or 0.0, name), "size": (e.size or 0, name),
                    "type": (os.path.splitext(name)[1], name), "annotated": (gt, name),
                    "review": (prop, name)}.get(mode, (name,))

        by_dir: dict[str, list[str]] = {}
        subdirs: dict[str, set[str]] = {}
        for rel in rels:
            d = rel.rpartition("/")[0]
            by_dir.setdefault(d, []).append(rel)
            while d:
                parent = d.rpartition("/")[0]
                subdirs.setdefault(parent, set()).add(d)
                d = parent
        out: list[str] = []

        def walk(d: str) -> None:
            for sub in sorted(subdirs.get(d, ()), key=str.lower):
                walk(sub)
            out.extend(sorted(by_dir.get(d, []), key=key, reverse=desc))
        walk("")
        return out

    def _resort(self, *_) -> None:
        QSettings("ScanBound", "ScanBound").setValue("browser_sort", self.sort.currentData())
        QSettings("ScanBound", "ScanBound").setValue("browser_desc", self.sort_desc.isChecked())
        self.sort_desc.setText("↓" if self.sort_desc.isChecked() else "↑")
        if self.items:
            self.rebuild(keep_thumbnails=True)

    def favourites(self) -> set[str]:
        return set(self.session.project.settings.get("favourites", []))

    def set_favourite(self, rels, on: bool) -> None:  # noqa: ANN001
        """Star / unstar files (saved with the project)."""
        fav = self.favourites()
        fav = (fav | set(rels)) if on else (fav - set(rels))
        self.session.project.settings["favourites"] = sorted(fav)
        self.session.project.dirty = True
        self.session._check_dirty()
        for rel in rels:
            it = self.items.get(rel)
            if it is not None:
                it.setData(0, FAV_ROLE, on)
                self._set_detail(it)
        self._apply_filter()

    def toggle_favourite(self) -> None:
        rels = [self._rel(it) for it in self.tree.selectedItems() if self._rel(it)]
        if rels:
            fav = self.favourites()
            self.set_favourite(rels, not all(r in fav for r in rels))

    def _context_menu(self, pos) -> None:  # noqa: ANN001
        it = self.tree.itemAt(pos)
        rel = self._rel(it)
        menu = QMenu(self)
        if rel:
            if it not in self.tree.selectedItems():
                self.tree.setCurrentItem(it)
            starred = bool(it.data(0, FAV_ROLE))
            menu.addAction("Remove from favourites" if starred else "Add to favourites  ★", self.toggle_favourite)
            menu.addAction("Show in folder", lambda: self._reveal(rel))
            menu.addSeparator()
        menu.addAction("Expand all folders", self.tree.expandAll)
        menu.addAction("Collapse all folders", self.tree.collapseAll)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _reveal(self, rel: str) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(os.path.join(self.session.project.root, rel))))

    def _folder(self, path: str) -> Optional[QTreeWidgetItem]:
        if not path:
            return None
        f = self.folders.get(path)
        if f is None:
            parent = self._folder(path.rpartition("/")[0])
            f = QTreeWidgetItem([path.rpartition("/")[2]])
            f.setData(0, FOLDER_ROLE, path)
            f.setSizeHint(0, QSize(THUMB, 24))         # folders: one compact row, no thumbnail-sized icon
            font = f.font(0)
            font.setBold(True)
            f.setFont(0, font)
            f.setToolTip(0, os.path.join(self.session.project.root, path))
            f.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if parent is None:
                self.tree.addTopLevelItem(f)
            else:
                parent.addChild(f)
            self.folders[path] = f
        return f

    # ------------------------------------------------------------------ thumbnails (visible rows first)
    def _schedule_thumbs(self) -> None:
        self._thumb_timer.start()

    def _visible_unloaded(self) -> list[str]:
        out = []
        vp = self.tree.viewport().rect()
        it = self.tree.itemAt(vp.topLeft() + QPoint(1, 1))
        while it is not None and self.tree.visualItemRect(it).top() <= vp.bottom():
            rel = self._rel(it)
            if rel and rel not in self._loaded:
                out.append(rel)
            it = self.tree.itemBelow(it)
        return out

    def _start_thumbs(self) -> None:
        if self._running >= MAX_THUMB_JOBS or not self.items:
            return
        todo = self._visible_unloaded()
        if not todo:          # then the rest, in the background, so scrolling finds them ready
            todo = [r for r, it in self.items.items() if r not in self._loaded and not it.isHidden()][:MAX_THUMB_JOBS]
        gen = self._generation
        root = self.session.project.root
        for rel in todo[: MAX_THUMB_JOBS - self._running]:
            self._loaded.add(rel)
            self._running += 1
            run_job(make_thumbnail, os.path.join(root, rel),
                    done=lambda out, r=rel: self._thumb_done(gen, r, out),
                    failed=lambda err, r=rel: self._thumb_failed(gen, r, err))

    def _job_finished(self, gen: int) -> bool:
        if gen != self._generation:
            return False
        self._running = max(0, self._running - 1)
        self._schedule_thumbs()
        return True

    def _thumb_done(self, gen: int, rel: str, out: dict) -> None:
        if not self._job_finished(gen) or rel not in self.items:
            return
        e = self.session.project.files.get(rel)
        if e is not None and not e.kind:
            e.kind, e.format = out["kind"], out["format"]
        self._cache[rel] = out
        self._apply_thumb(self.items[rel], out)

    def _apply_thumb(self, it: QTreeWidgetItem, out: dict) -> None:
        if "thumb" in out:
            it.setIcon(0, QIcon(QPixmap.fromImage(to_qimage(out["thumb"])).scaled(
                THUMB, THUMB, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
        q = out.get("quality", [])
        warn = [m for lvl, _c, m in q if lvl in ("warning", "error")] + out.get("warnings", [])
        codes = [c for _l, c, _m in q if c not in ("8bit",)]
        detail = " · ".join(t for t in [out.get("text", "")] + codes if t)
        tip = "\n".join([m for _l, _c, m in q] + out.get("warnings", []))
        if out["kind"] == "spectrum":
            tip = "spectroscopy - double-click to plot" + ("\n" + tip if tip else "")
        self._set_detail(it, detail, tip, Qt.GlobalColor.darkYellow if warn else None)

    def _set_detail(self, it: QTreeWidgetItem, detail: Optional[str] = None, tip: Optional[str] = None,
                    color=None) -> None:  # noqa: ANN001
        """The second line of an entry: size and quality codes ('unreadable' in red), then the counts."""
        R = Qt.ItemDataRole.UserRole
        if detail is not None:
            it.setData(0, R + 1, detail)
        if tip is not None:
            it.setData(0, R + 2, tip)
        rel = self._rel(it)
        gt, prop, reviewed = it.data(0, R + 3) or (0, 0, False)
        star = "★ " if it.data(0, FAV_ROLE) else ""
        parts = [it.data(0, R + 1) or ""]
        if gt:
            parts.append(f"{gt} annotated" + (" ✓" if reviewed else ""))
        elif reviewed:
            parts.append("reviewed ✓")
        if prop:
            parts.append(f"{prop} to review")
        line = " · ".join(p for p in parts if p)
        name = rel.rpartition("/")[2]
        it.setText(0, f"{star}{name}\n{line}" if line else star + name)
        tip = it.data(0, R + 2) or ""
        it.setToolTip(0, os.path.join(self.session.project.root, rel) + (f"\n\n{tip}" if tip else ""))
        if color is not None:
            it.setForeground(0, color)

    def _thumb_failed(self, gen: int, rel: str, err: str) -> None:
        if not self._job_finished(gen) or rel not in self.items:
            return
        it = self.items[rel]
        last = err.strip().splitlines()[-1] if err.strip() else "error"
        self._set_detail(it, "unreadable", last, Qt.GlobalColor.red)
        e = self.session.project.files.get(rel)
        if e is not None:
            e.error = last

    def _counts(self, rels: set) -> dict[str, list[int]]:
        """{file: [ground truth, pending proposals]} in one pass over the annotations."""
        counts: dict[str, list[int]] = {}
        for a in self.session.store:
            if a.kind not in ("feature", "centre_line"):
                continue
            rel = a.image_id.rpartition("::")[0]
            if rel not in rels:
                continue
            c = counts.setdefault(rel, [0, 0])
            if a.status in ("manual", "accepted"):
                c[0] += 1
            elif a.status == "proposed":
                c[1] += 1
        return counts

    def _update_counts(self, rels: set, counts: Optional[dict] = None) -> None:
        rels = {r for r in rels if r in self.items}
        if not rels:
            return
        counts = self._counts(rels) if counts is None else counts
        reviewed = {e.file for e in self.session.project.images.values() if e.reviewed} if REVIEW else set()
        for rel in rels:
            gt, prop = counts.get(rel, (0, 0))
            it = self.items[rel]
            it.setData(0, Qt.ItemDataRole.UserRole + 3, (gt, prop, rel in reviewed))
            self._set_detail(it)
        self._apply_filter()

    def _apply_filter(self, *_):
        text = self.filter.text().lower().strip()
        todo = self.only_todo.isChecked()
        fav_only = self.only_fav.isChecked()
        shown: set[str] = set()
        for rel, it in self.items.items():
            hide = bool(text) and text not in rel.lower()
            if todo and not (it.data(0, Qt.ItemDataRole.UserRole + 3) or (0, 0))[1]:
                hide = True
            if fav_only and not it.data(0, FAV_ROLE):
                hide = True
            it.setHidden(hide)
            if not hide:
                d = rel.rpartition("/")[0]
                while d and d not in shown:
                    shown.add(d)
                    d = d.rpartition("/")[0]
        for path, f in self.folders.items():
            f.setHidden(path not in shown)
            if (text or todo or fav_only) and path in shown:
                f.setExpanded(True)
        self._schedule_thumbs()

    # ------------------------------------------------------------------ navigation
    def _rel(self, it: Optional[QTreeWidgetItem]) -> str:
        return (it.data(0, Qt.ItemDataRole.UserRole) or "") if it is not None else ""

    def _on_current(self, it, _prev) -> None:  # noqa: ANN001
        rel = self._rel(it)
        if rel and self.session.is_image_file(rel):
            self.session.set_current(rel + "::")

    def _activate(self, it) -> None:  # noqa: ANN001
        rel = self._rel(it)
        if not rel:
            return
        if self.session.is_image_file(rel):
            self.session.set_current(rel + "::")
        else:
            self.spectrumActivated.emit(rel)

    def _on_image_changed(self, image_id: str) -> None:
        rel = image_id.rpartition("::")[0]
        it = self.items.get(rel)
        if it is not None and self.tree.currentItem() is not it:
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(it)          # also opens its folder
            self.tree.blockSignals(False)
            self.tree.scrollToItem(it)

    def step(self, delta: int) -> None:
        """Next / previous visible image file, in tree order (into collapsed folders too)."""
        visible = [r for r, it in self.items.items() if not it.isHidden() and self.session.is_image_file(r)]
        if not visible:
            return
        cur = self.session.current_id.rpartition("::")[0]
        k = visible.index(cur) if cur in visible else -1
        self.session.set_current(visible[(k + delta) % len(visible)] + "::")
