"""
3D model builder window: molecules from .xyz files placed on the molecule annotations of the current
image, on a substrate slab from the lattice fit and the aligned structure model.  Adjust heights,
angles, tilt, position and handedness, check for clashes, export .xyz.  The model logic is in
``scanbound.surface.builder``; this window only edits a Scene and draws it.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...surface.builder import (BuildSettings, MoleculeTemplate, Scene, bonds, build_scene, color, load_molecules,
                                load_state, radius, save_state)
from ...surface.store import SurfaceStore
from ..session import Session
from ..theme import mark
from ..widgets.glview import gl_view_class
from ..qtutil import save_file_name

HIGHLIGHT = "#00E5FF"


def _rgba(hexcol: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    c = QColor(hexcol)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


# --------------------------------------------------------------------------------------- viewers
class _GLViewer(QWidget):
    """pyqtgraph OpenGL view: drag to rotate, wheel to zoom, middle / Ctrl drag to pan."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        import pyqtgraph.opengl as gl
        self.gl = gl
        self.view = gl_view_class()()
        self.view.setBackgroundColor("#1E1E22")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.sub_item = None
        self.mol_item = None
        self.bond_item = None
        self._framed = False

    def show_atoms(self, sub: tuple, mol: tuple, bond_segments: np.ndarray, selected_mask: np.ndarray) -> None:
        gl = self.gl
        for it in (self.sub_item, self.mol_item, self.bond_item):
            if it is not None:
                self.view.removeItem(it)
        self.sub_item = self.mol_item = self.bond_item = None
        s_syms, s_pos, s_layer = sub
        if len(s_pos):
            col = np.array([_rgba(color(s), 1.0) for s in s_syms])
            shade = np.clip(1.0 + 0.18 * (s_pos[:, 2] / max(1e-9, abs(s_pos[:, 2]).max() or 1)), 0.45, 1.0)
            col[:, :3] *= shade[:, None]
            size = np.array([1.35 * radius(s) for s in s_syms])
            self.sub_item = gl.GLScatterPlotItem(pos=s_pos, color=col, size=size, pxMode=False)
            self.sub_item.setGLOptions("opaque")
            self.view.addItem(self.sub_item)
        m_syms, m_pos = mol
        if len(m_pos):
            col = np.array([_rgba(color(s)) for s in m_syms])
            col[selected_mask] = _rgba(HIGHLIGHT)
            size = np.array([0.9 * radius(s) + 0.25 for s in m_syms])
            self.mol_item = gl.GLScatterPlotItem(pos=m_pos, color=col, size=size, pxMode=False)
            self.mol_item.setGLOptions("opaque")
            self.view.addItem(self.mol_item)
        if len(bond_segments):
            self.bond_item = gl.GLLinePlotItem(pos=bond_segments, mode="lines", color=(0.85, 0.85, 0.85, 1.0),
                                               width=2.0, antialias=True)
            self.view.addItem(self.bond_item)
        if not self._framed:
            allp = np.vstack([p for p in (s_pos, m_pos) if len(p)]) if (len(s_pos) or len(m_pos)) else np.zeros((1, 3))
            c = allp.mean(0)
            span = float(np.ptp(allp[:, :2], axis=0).max()) if len(allp) > 1 else 20.0
            from PyQt6.QtGui import QVector3D
            self.view.opts["center"] = QVector3D(float(c[0]), float(c[1]), float(c[2]))
            self.view.setCameraPosition(distance=max(20.0, 1.3 * span), elevation=40, azimuth=-90)
            self._framed = True

    def top_view(self) -> None:
        self.view.setCameraPosition(elevation=90, azimuth=-90)

    def side_view(self) -> None:
        self.view.setCameraPosition(elevation=0, azimuth=-90)

    def grab_image(self):  # noqa: ANN201
        return self.view.grabFramebuffer()


class _MplViewer(QWidget):
    """Fallback when OpenGL is not available: matplotlib 3D (slower, same controls by mouse)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        note = QLabel("OpenGL is not available: simplified view (install pyqtgraph and PyOpenGL for the full one).")
        note.setWordWrap(True)
        mark(note, "muted")
        lay.addWidget(note)
        lay.addWidget(self.canvas, 1)
        self._elev, self._azim = 40, -90

    def show_atoms(self, sub: tuple, mol: tuple, bond_segments: np.ndarray, selected_mask: np.ndarray) -> None:
        ax = self.ax
        try:
            self._elev, self._azim = ax.elev, ax.azim
        except AttributeError:
            pass
        ax.clear()
        s_syms, s_pos, _ = sub
        if len(s_pos):
            top = s_pos[:, 2] > s_pos[:, 2].max() - 0.5
            ax.scatter(*s_pos[top].T, c=[color(s) for s, t in zip(s_syms, top) if t], s=30, depthshade=True)
            ax.scatter(*s_pos[~top].T, c=[color(s) for s, t in zip(s_syms, top) if not t], s=12, alpha=0.5)
        m_syms, m_pos = mol
        if len(m_pos):
            cols = [HIGHLIGHT if sel else color(s) for s, sel in zip(m_syms, selected_mask)]
            ax.scatter(*m_pos.T, c=cols, s=[18 if s == "H" else 40 for s in m_syms], edgecolors="k", linewidths=0.3)
        for a, b in bond_segments.reshape(-1, 2, 3):
            ax.plot(*np.array([a, b]).T, color="#888888", lw=1)
        allp = np.vstack([p for p in (s_pos, m_pos) if len(p)]) if (len(s_pos) or len(m_pos)) else np.zeros((1, 3))
        c = allp.mean(0)
        r = max(5.0, float(np.ptp(allp[:, :2], axis=0).max()) / 2)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(allp[:, 2].min() - 1, allp[:, 2].min() - 1 + 2 * r)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        ax.view_init(self._elev, self._azim)
        self.canvas.draw_idle()

    def top_view(self) -> None:
        self.ax.view_init(90, -90)
        self.canvas.draw_idle()

    def side_view(self) -> None:
        self.ax.view_init(0, -90)
        self.canvas.draw_idle()

    def grab_image(self):  # noqa: ANN201
        return self.canvas.grab().toImage()


def make_viewer(parent: Optional[QWidget] = None) -> QWidget:
    if os.environ.get("SCANBOUND_NO_GL") or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return _MplViewer(parent)
    try:
        return _GLViewer(parent)
    except Exception:  # noqa: BLE001 - no pyqtgraph / OpenGL: fall back
        return _MplViewer(parent)


# ---------------------------------------------------------------------------------------- window
COLS = ["", "Label", "Molecule", "Height (Å)", "Angle (°)", "Tilt (°)", "Mirrored", "Moved (Å)", "Site", "Check"]


class Builder3DWindow(QDialog):
    def __init__(self, session: Session, image_id: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.image_id = image_id
        self.store = SurfaceStore(session.project)
        self.setWindowTitle(f"3D model — {session.project.file_name(image_id)}")
        self.setModal(False)
        self.resize(1400, 860)
        self.templates, self.label_map, self.settings, self._saved = load_state(self.store, image_id)
        self.on_molecules_changed: Optional[Callable[[], None]] = None    # set by the Surface panel
        self.scene: Optional[Scene] = None
        self._checks: dict[str, dict] = {}
        self._updating = False
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(60)
        self._redraw_timer.timeout.connect(self.redraw)

        self.viewer = make_viewer(self)
        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(4, 4, 4, 4)

        # ---- molecules (templates)
        g = QGroupBox("Molecule files")
        gl_ = QVBoxLayout(g)
        self.tlist = QListWidget()
        self.tlist.setMaximumHeight(90)
        self.tlist.currentRowChanged.connect(self._show_template)
        row = QHBoxLayout()
        add = QPushButton("Add .xyz…")
        add.setToolTip("Molecule geometry in XYZ or extended XYZ (Å). Several files: assign them to labels below.")
        add.clicked.connect(self.add_templates)
        rem = QPushButton("Remove")
        rem.clicked.connect(self.remove_template)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        tf = QFormLayout()
        self.t_flatten = QCheckBox("Lay flat (long axis along the annotated angle)")
        self.t_offset = QDoubleSpinBox()
        self.t_offset.setRange(-360, 360)
        self.t_offset.setDecimals(1)
        self.t_offset.setSuffix(" °")
        self.t_offset.setToolTip("Added to every molecule of this kind: matches the file's axis to the way the "
                                 "orientation was annotated")
        self.t_flip = QCheckBox("Upside down (other face to the surface)")
        self.t_hand = QComboBox()
        self.t_hand.addItems(["not chiral / unknown", "L", "R"])
        self.t_hand.setToolTip("Handedness of the geometry in the file: annotations of the other handedness get "
                               "its mirror image")
        self.t_info = QLabel("")
        mark(self.t_info, "muted")
        for w in (self.t_flatten, self.t_flip):
            w.toggled.connect(self._template_edited)
        self.t_offset.valueChanged.connect(self._template_edited)
        self.t_hand.currentIndexChanged.connect(self._template_edited)
        tf.addRow(self.t_flatten)
        tf.addRow("Angle offset", self.t_offset)
        tf.addRow(self.t_flip)
        tf.addRow("File is", self.t_hand)
        tf.addRow(self.t_info)
        gl_.addWidget(self.tlist)
        gl_.addLayout(row)
        gl_.addLayout(tf)
        cl.addWidget(g)

        # ---- labels -> molecules
        g = QGroupBox("Annotation label → molecule")
        self.map_form = QFormLayout(g)
        cl.addWidget(g)

        # ---- substrate and defaults
        g = QGroupBox("Substrate and defaults")
        f = QFormLayout(g)
        self.s_sub = QCheckBox("Include the substrate")
        self.s_layers = QSpinBox()
        self.s_layers.setRange(1, 12)
        self.s_margin = QDoubleSpinBox()
        self.s_margin.setRange(0, 200)
        self.s_margin.setSuffix(" Å")
        self.s_extent = QComboBox()
        self.s_extent.addItem("Around the molecules", "molecules")
        self.s_extent.addItem("The whole image", "image")
        self.s_species = QLineEdit()
        self.s_species.setToolTip("Substrate element when no structure model is aligned (one atom per image maximum)")
        self.s_height = QDoubleSpinBox()
        self.s_height.setRange(0.0, 30.0)
        self.s_height.setDecimals(2)
        self.s_height.setSingleStep(0.1)
        self.s_height.setSuffix(" Å")
        self.s_height.setToolTip("Height of new molecules: from the outermost substrate layer to the molecule's "
                                 "lowest atom")
        self.s_snap = QCheckBox("Snap molecules to their adsorption site")
        self.s_snap.setToolTip("Move each molecule onto the most likely site of the aligned model (Adsorption "
                               "sites analysis) instead of its marked position")
        self.s_prop = QCheckBox("Include unreviewed proposals")
        rebuild = QPushButton("Rebuild")
        rebuild.setToolTip("Rebuild from the annotations with these settings; manual adjustments are kept")
        rebuild.clicked.connect(self.rebuild)
        f.addRow(self.s_sub)
        f.addRow("Layers", self.s_layers)
        f.addRow("Extent", self.s_extent)
        f.addRow("Margin", self.s_margin)
        f.addRow("Element (no model)", self.s_species)
        f.addRow("Default height", self.s_height)
        f.addRow(self.s_snap)
        f.addRow(self.s_prop)
        from ..features import REVIEW
        f.setRowVisible(self.s_prop, REVIEW)            # without review there are no proposals
        f.addRow(rebuild)
        cl.addWidget(g)
        cl.addStretch(1)

        # ---- placements table + editor (below the view)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemChanged.connect(self._enabled_changed)
        ed = QGroupBox("Selected molecules")
        ef = QHBoxLayout(ed)
        self.e_height = self._spin(0, 30, 0.1, " Å", "Height above the outermost layer (lowest atom)")
        self.e_angle = self._spin(-360, 360, 5, " °", "Rotation about the surface normal, counter-clockwise from above")
        self.e_tilt = self._spin(-90, 90, 5, " °", "Tilt about the molecule's long axis")
        self.e_dx = self._spin(-100, 100, 0.25, " Å", "Shift from the marked position along x (image right)")
        self.e_dy = self._spin(-100, 100, 0.25, " Å", "Shift from the marked position along y (image up)")
        self.e_mirror = QCheckBox("Mirrored")
        self.e_mirror.setToolTip("Mirror image in the surface plane (the other enantiomer)")
        self.e_mirror.toggled.connect(lambda v: self._edit("mirrored", v))
        for name, w in (("height", self.e_height), ("angle", self.e_angle), ("tilt", self.e_tilt),
                        ("dx", self.e_dx), ("dy", self.e_dy)):
            w.valueChanged.connect(lambda v, n=name: self._edit(n, v))
        for text, w in (("Height", self.e_height), ("Angle", self.e_angle), ("Tilt", self.e_tilt),
                        ("Δx", self.e_dx), ("Δy", self.e_dy)):
            ef.addWidget(QLabel(text))
            ef.addWidget(w)
        ef.addWidget(self.e_mirror)
        reset = QPushButton("Reset")
        reset.setToolTip("Back to the marked position and orientation, default height, no tilt")
        reset.clicked.connect(self.reset_selected)
        ef.addWidget(reset)
        ef.addStretch(1)
        allrow = QHBoxLayout()
        self.all_height = self._spin(0, 30, 0.1, " Å", "")
        allb = QPushButton("Set height of all")
        allb.clicked.connect(lambda: self._set_all("height", self.all_height.value()))
        rot = QPushButton("Rotate all by…")
        rot.clicked.connect(self._rotate_all)
        selall = QPushButton("Select all")
        selall.clicked.connect(self.table.selectAll)
        allrow.addWidget(QLabel("All molecules:"))
        allrow.addWidget(self.all_height)
        allrow.addWidget(allb)
        allrow.addWidget(rot)
        allrow.addWidget(selall)
        allrow.addStretch(1)

        # ---- bottom bar
        self.status = QLabel("")
        self.status.setWordWrap(True)
        top = QPushButton("Top view")
        top.clicked.connect(self.viewer.top_view)
        side = QPushButton("Side view")
        side.clicked.connect(self.viewer.side_view)
        self.show_sub = QCheckBox("Show substrate")
        self.show_sub.setChecked(True)
        self.show_sub.toggled.connect(lambda _v: self.redraw())
        self.export_sub = QCheckBox("Export substrate")
        self.export_sub.setChecked(True)
        export = QPushButton("Export .xyz…")
        export.setToolTip("Extended XYZ (readable by ASE) with a molecule index per atom, plus a _placements.json "
                          "with every setting and the clash check")
        export.clicked.connect(self.export)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bar = QHBoxLayout()
        bar.addWidget(top)
        bar.addWidget(side)
        bar.addWidget(self.show_sub)
        bar.addStretch(1)
        bar.addWidget(self.export_sub)
        bar.addWidget(export)
        bar.addWidget(close)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(self.viewer)
        lower = QWidget()
        lw = QVBoxLayout(lower)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.addWidget(self.table, 1)
        lw.addWidget(ed)
        lw.addLayout(allrow)
        vsplit.addWidget(lower)
        vsplit.setSizes([560, 260])
        ll.addWidget(vsplit, 1)
        ll.addWidget(self.status)
        ll.addLayout(bar)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(380)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(scroll)
        split.setSizes([1050, 350])
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        self._load_settings()
        self._fill_templates()
        self._fill_label_map()
        session.selectionChanged.connect(self._follow_canvas_selection)
        if self.templates:
            self.rebuild()
        else:
            self.status.setText("Add a molecule .xyz file to start. The image needs a lattice fit and molecule "
                                "annotations.")

    # ------------------------------------------------------------------ small helpers
    def _spin(self, lo: float, hi: float, step: float, suffix: str, tip: str) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(2)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        if tip:
            w.setToolTip(tip)
        return w

    def _load_settings(self) -> None:
        s = self.settings
        self.s_sub.setChecked(s.substrate)
        self.s_layers.setValue(s.layers)
        self.s_margin.setValue(s.margin)
        self.s_extent.setCurrentIndex(max(0, self.s_extent.findData(s.extent)))
        self.s_species.setText(s.species)
        self.s_height.setValue(s.height)
        self.all_height.setValue(s.height)
        self.s_snap.setChecked(s.snap)
        self.s_prop.setChecked(s.include_proposed)

    def _read_settings(self) -> BuildSettings:
        return BuildSettings(self.s_sub.isChecked(), self.s_layers.value(), self.s_margin.value(),
                             self.s_extent.currentData(), self.s_height.value(), self.s_snap.isChecked(),
                             self.s_species.text().strip() or "Au", self.s_prop.isChecked())

    # ------------------------------------------------------------------ templates
    def _fill_templates(self) -> None:
        cur = self.tlist.currentRow()
        self.tlist.blockSignals(True)
        self.tlist.clear()
        for t in self.templates.values():
            self.tlist.addItem(QListWidgetItem(f"{t.name}  ({t.formula()}, {len(t.symbols)} atoms)"))
        self.tlist.blockSignals(False)
        if self.templates:
            self.tlist.setCurrentRow(min(max(cur, 0), len(self.templates) - 1))
        self._show_template(self.tlist.currentRow())

    def current_template(self) -> Optional[MoleculeTemplate]:
        k = self.tlist.currentRow()
        names = list(self.templates)
        return self.templates[names[k]] if 0 <= k < len(names) else None

    def _show_template(self, _row: int) -> None:
        t = self.current_template()
        self._updating = True
        for w in (self.t_flatten, self.t_offset, self.t_flip, self.t_hand):
            w.setEnabled(t is not None)
        if t is not None:
            self.t_flatten.setChecked(t.flatten)
            self.t_offset.setValue(t.angle_offset)
            self.t_flip.setChecked(t.flip)
            self.t_hand.setCurrentIndex({"L": 1, "R": 2}.get(t.handedness or "", 0))
            g = t.geometry()
            self.t_info.setText(f"{np.ptp(g[:, 0]):.1f} × {np.ptp(g[:, 1]):.1f} × {np.ptp(g[:, 2]):.1f} Å as placed"
                                + (f"; {t.source}" if t.source else ""))
        self._updating = False

    def _template_edited(self) -> None:
        t = self.current_template()
        if t is None or self._updating:
            return
        t.flatten = self.t_flatten.isChecked()
        t.angle_offset = self.t_offset.value()
        t.flip = self.t_flip.isChecked()
        t.handedness = {1: "L", 2: "R"}.get(self.t_hand.currentIndex())
        self._show_template(self.tlist.currentRow())
        if self.scene is not None:
            self.scene.templates[t.name] = t
            self.rebuild()                       # handedness can change which molecules are mirrored

    def add_templates(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Molecule geometry", "", "XYZ files (*.xyz *.extxyz);;All files (*)")
        for p in paths:
            try:
                t = MoleculeTemplate.from_xyz(p)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "3D model", f"Could not read {p}: {exc}")
                continue
            name, k = t.name, 2
            while name in self.templates:
                name, k = f"{t.name} {k}", k + 1
            t.name = name
            self.templates[name] = t
        if paths:
            self._fill_templates()
            self._fill_label_map()
            self.rebuild()

    def add_template(self, t: MoleculeTemplate) -> None:
        """For scripts and tests."""
        self.templates[t.name] = t
        self._fill_templates()
        self._fill_label_map()

    def remove_template(self) -> None:
        t = self.current_template()
        if t is None:
            return
        del self.templates[t.name]
        self.label_map = {k: v for k, v in self.label_map.items() if v != t.name}
        self._fill_templates()
        self._fill_label_map()
        self.rebuild()

    def _fill_label_map(self) -> None:
        while self.map_form.rowCount():
            self.map_form.removeRow(0)
        labels = self.store.molecule_labels([self.image_id])
        names = list(self.templates)
        if not labels:
            self.map_form.addRow(QLabel("No molecule labels on this image."))
        for lab in labels:
            cb = QComboBox()
            cb.addItem("(none)", "")
            for n in names:
                cb.addItem(n, n)
            want = self.label_map.get(lab) or (names[0] if len(names) == 1 else "")
            cb.setCurrentIndex(max(0, cb.findData(want)))
            self.label_map[lab] = cb.currentData()
            cb.currentIndexChanged.connect(lambda _i, lab=lab, cb=cb: self._map_changed(lab, cb.currentData()))
            self.map_form.addRow(lab, cb)

    def _map_changed(self, label: str, name: str) -> None:
        self.label_map[label] = name
        self.rebuild()

    # ------------------------------------------------------------------ building
    def rebuild(self) -> None:
        self.settings = self._read_settings()
        if not self.templates:
            self.scene = None
            self._fill_table()
            self.redraw()
            return
        prev = self.scene.placements if self.scene is not None else self._saved
        try:
            self.scene = build_scene(self.store, self.image_id, self.templates, self.label_map, self.settings, prev)
        except Exception as exc:  # noqa: BLE001 - shown to the user, the window stays open
            self.status.setText(str(exc))
            mark(self.status, "error")
            return
        mark(self.status, None)
        self._fill_table()
        self.redraw()
        self.save()

    def save(self) -> None:
        save_state(self.store, self.image_id, self.scene, self.templates, self.label_map, self.settings)
        self.session._check_dirty()
        if self.on_molecules_changed is not None:
            self.on_molecules_changed()

    def reload_molecules(self) -> None:
        """Molecule files or label assignments were changed elsewhere (the Surface panel): take them over."""
        self.templates, self.label_map = load_molecules(self.store)
        self._fill_templates()
        self._fill_label_map()
        self.rebuild()

    def _fill_table(self) -> None:
        sel = set(self.selected_ids())
        self._updating = True
        self.table.setRowCount(0)
        places = self.scene.placements if self.scene is not None else []
        self._checks = {r["id"]: r for r in self.scene.check()} if self.scene is not None else {}
        self.table.setRowCount(len(places))
        for i, p in enumerate(places):
            on = QTableWidgetItem()
            on.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            on.setCheckState(Qt.CheckState.Checked if p.enabled else Qt.CheckState.Unchecked)
            on.setData(Qt.ItemDataRole.UserRole, p.id)
            on.setToolTip("Include this molecule")
            self.table.setItem(i, 0, on)
            self._fill_row(i, p)
        self._updating = False
        if sel:
            self.select_ids(sel)

    def _fill_row(self, i: int, p: Any) -> None:
        c = self._checks.get(p.id, {})
        moved = math.hypot(p.x - p.x0, p.y - p.y0)
        check = "clash" if c.get("clash") else ("ok" if c else "")
        if c and math.isfinite(c.get("closest_substrate", math.nan)):
            check += f" (substrate {c['closest_substrate']:.2f} Å"
            if math.isfinite(c.get("closest_molecule", math.nan)):
                check += f", molecule {c['closest_molecule']:.2f} Å"
            check += ")"
        vals = [p.label or "", p.template, f"{p.height:.2f}", f"{(p.angle % 360):.1f}", f"{p.tilt:.1f}",
                "yes" if p.mirrored else "", f"{moved:.2f}" if moved > 1e-6 else "", p.site, check]
        for j, v in enumerate(vals, start=1):
            it = QTableWidgetItem(v)
            if j == 9 and c.get("clash"):
                it.setForeground(QColor("#E53935"))
            self.table.setItem(i, j, it)

    def selected_ids(self) -> list[str]:
        rows = sorted({ix.row() for ix in self.table.selectedIndexes()})
        out = []
        for r in rows:
            it = self.table.item(r, 0)
            if it is not None:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def select_ids(self, ids: Any) -> None:
        ids = set(ids)
        self.table.blockSignals(True)
        self.table.clearSelection()
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) in ids:
                self.table.selectRow(r)
        self.table.setSelectionMode(mode)
        self.table.blockSignals(False)
        self._selection_changed()

    def _selected_placements(self) -> list[Any]:
        if self.scene is None:
            return []
        ids = set(self.selected_ids())
        return [p for p in self.scene.placements if p.id in ids]

    def _selection_changed(self) -> None:
        ps = self._selected_placements()
        self._updating = True
        if ps:
            p = ps[0]
            self.e_height.setValue(p.height)
            self.e_angle.setValue(p.angle)
            self.e_tilt.setValue(p.tilt)
            self.e_dx.setValue(p.x - p.x0)
            self.e_dy.setValue(p.y - p.y0)
            self.e_mirror.setChecked(p.mirrored)
        self._updating = False
        self._redraw_timer.start()

    def _follow_canvas_selection(self) -> None:
        if not self.isVisible() or self.scene is None:
            return
        ids = {p.id for p in self.scene.placements} & set(self.session.selection)
        if ids:
            self.select_ids(ids)

    def _enabled_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 0 or self.scene is None:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        for p in self.scene.placements:
            if p.id == pid:
                p.enabled = item.checkState() == Qt.CheckState.Checked
        self._changed()

    def _edit(self, field_: str, value: Any) -> None:
        if self._updating:
            return
        for p in self._selected_placements():
            if field_ == "dx":
                p.x = p.x0 + float(value)
            elif field_ == "dy":
                p.y = p.y0 + float(value)
            else:
                setattr(p, field_, value)
        self._changed()

    def _set_all(self, field_: str, value: Any) -> None:
        if self.scene is None:
            return
        for p in self.scene.placements:
            setattr(p, field_, value)
        self._changed()

    def _rotate_all(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        if self.scene is None:
            return
        v, ok = QInputDialog.getDouble(self, "Rotate all", "Add this angle to every molecule (°):", 60.0, -360, 360, 1)
        if ok:
            for p in self.scene.placements:
                p.angle += v
            self._changed()

    def reset_selected(self) -> None:
        for p in self._selected_placements():
            p.x, p.y, p.tilt = p.x0, p.y0, 0.0
            p.height = self.settings.height
        if self.scene is not None:
            self.rebuild()

    def _changed(self) -> None:
        if self.scene is None:
            return
        self._checks = {r["id"]: r for r in self.scene.check()}
        self._updating = True
        for i, p in enumerate(self.scene.placements):
            self._fill_row(i, p)
        self._updating = False
        self._redraw_timer.start()
        self.save()

    # ------------------------------------------------------------------ drawing and export
    def redraw(self) -> None:
        sc = self.scene
        if sc is None:
            self.viewer.show_atoms(([], np.zeros((0, 3)), np.zeros(0)), ([], np.zeros((0, 3))), np.zeros((0, 3)),
                                   np.zeros(0, bool))
            return
        sel = set(self.selected_ids())
        m_syms: list[str] = []
        m_pos: list[np.ndarray] = []
        m_sel: list[np.ndarray] = []
        segs: list[np.ndarray] = []
        for p in sc.placements:
            if not p.enabled:
                continue
            s, g = sc.molecule(p)
            b = bonds(s, g)
            if len(b):
                segs.append(np.stack([g[b[:, 0]], g[b[:, 1]]], axis=1).reshape(-1, 3))
            m_syms += s
            m_pos.append(g)
            m_sel.append(np.full(len(g), p.id in sel))
        sub = (sc.substrate_symbols, sc.substrate_positions, sc.substrate_layers) if self.show_sub.isChecked() \
            else ([], np.zeros((0, 3)), np.zeros(0))
        self.viewer.show_atoms(sub, (m_syms, np.vstack(m_pos) if m_pos else np.zeros((0, 3))),
                               np.vstack(segs) if segs else np.zeros((0, 3)),
                               np.concatenate(m_sel) if m_sel else np.zeros(0, bool))
        n_sub = len(sc.substrate_positions)
        clashes = sum(1 for c in self._checks.values() if c.get("clash"))
        txt = (f"{sum(p.enabled for p in sc.placements)} molecule(s), {len(m_syms)} molecule atoms, {n_sub} substrate atoms "
               f"({sc.provenance.get('model', '')}). ")
        txt += f"{clashes} clash(es): closer than the sum of radii." if clashes else "No clashes."
        if sc.notes:
            txt += " " + " ".join(sc.notes)
        self.status.setText(txt)

    def export(self) -> None:
        if self.scene is None:
            return
        base = os.path.splitext(self.session.project.file_name(self.image_id))[0]
        path = save_file_name(self, "Export 3D model", f"{base}_model.xyz",
                              "Extended XYZ (*.xyz);;Plain XYZ (*.xyz)")
        if not path:
            return
        n = self.export_to(path)
        self.session.info(f"Wrote {n} atoms to {path} (and {os.path.basename(os.path.splitext(path)[0])}_placements.json)")

    def export_to(self, path: str, extended: bool = True) -> int:
        assert self.scene is not None
        return self.scene.write_xyz(path, extended=extended, substrate=self.export_sub.isChecked())

    def closeEvent(self, event) -> None:  # noqa: ANN001
        try:
            self.session.selectionChanged.disconnect(self._follow_canvas_selection)
        except TypeError:
            pass
        if self.scene is not None:
            self.save()
        super().closeEvent(event)
