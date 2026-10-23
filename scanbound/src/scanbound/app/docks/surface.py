"""
Surface panel: fit the substrate lattice, align a structure model, choose what the overlay shows,
set how molecules are read from the annotations, and start the analyses.  The analyses themselves
live in ``scanbound.surface``; this panel only collects options and shows results.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QScrollArea,
                             QSlider, QSpinBox, QVBoxLayout, QWidget)

from ...surface.builder import load_molecules, save_molecules
from ...surface.store import SurfaceStore
from ..qtutil import run_job, save_file_name
from ..session import Session
from ..surface_overlay import SurfaceOverlay
from ..theme import mark

MAPS = (("", "No map"), ("dilatation", "Strain: dilatation"), ("rotation", "Strain: rotation"),
        ("shear", "Strain: shear"), ("domains", "Domains"))


class SurfaceDock(QScrollArea):
    reportReady = pyqtSignal(object)          # a workflow Report to show

    def __init__(self, session: Session, overlay: SurfaceOverlay, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self.overlay = overlay
        self.setWidgetResizable(True)
        self._busy = False
        self._maps: dict[tuple[str, str], Any] = {}        # (image id, kind) -> computed map
        self._alignments: list[Any] = []
        self._builders: dict[str, Any] = {}               # image id -> open 3D model window
        self._fit_notes: dict[str, tuple[str, Optional[str]]] = {}   # image id -> result of the last fit here
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(4, 4, 4, 4)

        # ---------------------------------------------------------------- lattice
        g = QGroupBox("Substrate lattice")
        f = QFormLayout(g)
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.0, 1000.0)
        self.spacing.setDecimals(3)
        self.spacing.setSingleStep(0.01)
        self.spacing.setSuffix(" nm")
        self.spacing.setSpecialValueText("automatic")
        self.spacing.setToolTip("Approximate nearest-neighbour spacing of the lattice (0 = from the Fourier "
                                "transform). Arrows step 0.01; type any value, Page Up/Down steps 0.1.")
        self.spacing.valueChanged.connect(lambda _v: self._preview_inputs_changed())
        est = QPushButton("Estimate")
        est.setToolTip("Fill in the spacing found in the Fourier transform of this image")
        est.clicked.connect(self.estimate_spacing)
        sprow = QHBoxLayout()
        sprow.addWidget(self.spacing, 1)
        sprow.addWidget(est)
        self.contrast = QComboBox()
        self.contrast.addItems(["Bright sites (maxima)", "Dark sites (minima)"])
        self.contrast.currentIndexChanged.connect(lambda _i: self._preview_inputs_changed())
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 20.0)
        self.threshold.setDecimals(2)
        self.threshold.setValue(1.5)
        self.threshold.setSingleStep(0.05)
        self.threshold.setToolTip("Minimum site contrast in robust standard deviations of the filtered image: "
                                  "lower finds weaker sites (and more noise)")
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(0, 1000)                  # 0 … 10 in steps of 0.01
        self.thr_slider.setValue(150)
        self.thr_slider.setToolTip(self.threshold.toolTip())
        self.thr_slider.valueChanged.connect(lambda v: self.threshold.setValue(v / 100.0))
        self.threshold.valueChanged.connect(self._threshold_changed)
        throw = QHBoxLayout()
        throw.addWidget(self.thr_slider, 1)
        throw.addWidget(self.threshold)
        self.preview = QCheckBox("Preview sites")
        self.preview.setToolTip("Show the sites the fit would use, live, while you set the spacing and threshold "
                                "(green: used; grey: below the threshold)")
        self.preview.toggled.connect(self._preview_toggled)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        mark(self.preview_label, "muted")
        self._candidates: Any = None                       # (image id, inputs, SiteSet) of the last preview
        self._preview_busy = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._run_preview)
        self.fit_btn = QPushButton("Fit lattice")
        self.fit_btn.setToolTip("Find the lattice sites, index them and fit origin and vectors with uncertainties. "
                                "Areas drawn as Region 'exclude' are left out.")
        self.fit_btn.clicked.connect(self.fit_lattice)
        clear = QPushButton("Remove")
        clear.clicked.connect(self.remove_fit)
        self.result = QLabel("No lattice fit on this image.")
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cal = QPushButton("Calibrate…")
        cal.setToolTip("Set the pixel size from a known lattice constant (e.g. Au(111), Cu(111))")
        cal.clicked.connect(self.calibrate)
        vec_btn = QPushButton("Lattice vectors…")
        vec_btn.setToolTip("Type in a1 and a2 when you know them (lengths, angles, origin); they are drawn as the "
                           "lattice vectors of this image")
        vec_btn.clicked.connect(lambda: self.edit_lattice_vectors())
        self.from_vectors = QCheckBox("Start from the lattice vectors")
        self.from_vectors.setChecked(True)
        self.from_vectors.setToolTip("When this image has a pair of lattice vectors (drawn or typed in), the fit starts "
                                     "from them and takes its spacing from them instead of guessing the lattice")
        row = QHBoxLayout()
        row.addWidget(self.fit_btn, 1)
        row.addWidget(vec_btn)
        row.addWidget(cal)
        row.addWidget(clear)
        f.addRow("Spacing", sprow)
        f.addRow("Sites are", self.contrast)
        f.addRow("Threshold", throw)
        f.addRow(self.preview)
        f.addRow(self.preview_label)
        f.addRow(self.from_vectors)
        f.addRow(row)
        f.addRow(self.result)
        lay.addWidget(g)

        # ---------------------------------------------------------------- overlay
        g = QGroupBox("Show on the image")
        grid = QGridLayout(g)
        self.chk: dict[str, QCheckBox] = {}
        for k, (key, text, on, tip) in enumerate((
                ("show_sites", "Fitted sites", True, "Circles: indexed sites, green → red with distance from the "
                                                     "ideal lattice; magenta crosses: sites off the lattice"),
                ("show_ideal", "Ideal lattice", False, "Yellow crosses where the fitted lattice puts every site"),
                ("show_cell", "Unit cell", True, "a1 and a2 of the fitted lattice"),
                ("show_model", "Model atoms", False, "Atoms of the aligned structure model (filled: top layer)"),
                ("show_model_sites", "Model sites", False, "Adsorption sites of the model (red top, blue bridge, "
                                                           "green hollow)"))):
            c = QCheckBox(text)
            c.setChecked(on)
            c.setToolTip(tip)
            c.toggled.connect(lambda v, key=key: (setattr(self.overlay, key, v), self.overlay.update()))
            grid.addWidget(c, k // 2, k % 2)
            self.chk[key] = c
        self.map = QComboBox()
        for key, text in MAPS:
            self.map.addItem(text, key)
        self.map.setToolTip("A colour map from the local lattice distortion (strain relative to the mean "
                            "lattice; drift and creep show here too) or the domains")
        self.map.currentIndexChanged.connect(self.update_map)
        grid.addWidget(self.map, 3, 0, 1, 2)
        lay.addWidget(g)

        # ---------------------------------------------------------------- model
        g = QGroupBox("Structure model")
        f = QFormLayout(g)
        self.model = QComboBox()
        self.model.setToolTip("Structure models of this project (saved in the project file)")
        self.model.currentIndexChanged.connect(self._model_changed)
        mrow = QHBoxLayout()
        for text, slot, tip in (("Preset…", self.add_preset, "fcc(111), fcc(100), graphene …"),
                                ("Import…", self.import_model, "ASE extended XYZ (top layers) or a ScanBound model JSON"),
                                ("Edit…", self.edit_model, "Edit atoms and sites as JSON (e.g. a PdGa termination)"),
                                ("Save…", self.save_model, "Save the model as JSON to use in other projects")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            mrow.addWidget(b)
        self.align = QComboBox()
        self.align.setToolTip("Ways the model cell fits the fitted lattice, best first. Equivalent choices the "
                              "image cannot tell apart are listed too: pick the one your other data support.")
        self.origin = QComboBox()
        self.origin.setToolTip("Which model position sits on the image maxima (the fitted sites). STM maxima are "
                               "not always atoms: try alternatives and compare with the molecules.")
        arow = QHBoxLayout()
        use = QPushButton("Use")
        use.clicked.connect(self.use_alignment)
        unuse = QPushButton("None")
        unuse.setToolTip("Analyse without a model: sites are relative to the image maxima")
        unuse.clicked.connect(self.clear_alignment)
        arow.addWidget(use)
        arow.addWidget(unuse)
        self.align_label = QLabel("")
        self.align_label.setWordWrap(True)
        mark(self.align_label, "muted")
        f.addRow("Model", self.model)
        f.addRow(mrow)
        f.addRow("Alignment", self.align)
        f.addRow("On maxima", self.origin)
        f.addRow(arow)
        f.addRow(self.align_label)
        lay.addWidget(g)

        # ---------------------------------------------------------------- settings
        g = QGroupBox("Molecules")
        f = QFormLayout(g)
        self.labels = QLineEdit()
        self.labels.setPlaceholderText("all feature labels except defects")
        self.labels.setToolTip("Comma-separated labels of the feature annotations treated as molecules")
        self.defects = QLineEdit()
        self.defects.setToolTip("Labels of feature annotations that are defects (vacancies, adatoms …)")
        self.chirality = QLineEdit()
        self.chirality.setPlaceholderText("e.g. mol-L=L, mol-R=R")
        self.chirality.setToolTip("Which labels are left- and right-handed. Single annotations can be set with "
                                  "right-click → Handedness, which takes precedence.")
        self.fold = QSpinBox()
        self.fold.setRange(1, 12)
        self.fold.setToolTip("Rotational symmetry of the molecule: orientations are reduced modulo 360°/n")
        self.proposed = QCheckBox("Include unreviewed proposals")
        self.proposed.setToolTip("Off: only manual and accepted annotations enter the statistics")
        self.tol = QDoubleSpinBox()
        self.tol.setRange(0.01, 0.5)
        self.tol.setSingleStep(0.01)
        self.tol.setDecimals(2)
        self.tol.setToolTip("Spread of molecule positions around a site that is still 'on' it, as a fraction "
                            "of the lattice constant (adds to the measured position uncertainty)")
        self.mirror = QCheckBox("Mirror-equivalent orientations")
        self.mirror.setToolTip("Treat θ and −θ relative to the lattice as the same (achiral analysis)")
        for w in (self.labels, self.defects, self.chirality):
            w.editingFinished.connect(self._save_settings)
        for w in (self.fold, self.tol):
            w.valueChanged.connect(lambda _v: self._save_settings())
        for w in (self.proposed, self.mirror):
            w.toggled.connect(lambda _v: self._save_settings())
        f.addRow("Molecules", self.labels)
        f.addRow("Defects", self.defects)
        f.addRow("Handedness", self.chirality)
        f.addRow("Symmetry", self.fold)
        f.addRow("Site tolerance", self.tol)
        f.addRow(self.proposed)
        from ..features import REVIEW
        f.setRowVisible(self.proposed, REVIEW)          # without review there are no proposals
        f.addRow(self.mirror)
        lay.addWidget(g)

        # ---------------------------------------------------------------- analyses
        g = QGroupBox("Analyse this image")
        grid = QGridLayout(g)
        from ...surface import workflow as wf
        self.buttons: list[QPushButton] = [self.fit_btn]
        for k, (text, fn, tip) in enumerate((
                ("Structure", wf.structure_explorer, "Lattice, symmetry, periodicities, strain, domains"),
                ("Adsorption sites", wf.adsorption_map, "Sites, positions in the cell, orientations, handedness, "
                                                        "nearness to defects and steps"),
                ("Assembly", wf.assembly_analysis, "Neighbours, g(r), order, chains, mixing, against random"),
                ("Defects", wf.defect_map, "Vacancies, adatoms and their clusters"),
                ("Step edges", wf.step_edges, "Detect step edges"),
                ("Terraces", wf.terrace_map, "Terraces between the step edges"),
                ("Scan quality", wf.quality_check, "Line jumps, streaks, tip changes, double tip, noise"))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, fn=fn: self.run(fn))
            grid.addWidget(b, k // 2, k % 2)
            self.buttons.append(b)
        b = QPushButton("Explain selected")
        b.setToolTip("Rank candidate explanations of the selected feature with the evidence for each")
        b.clicked.connect(self.explain)
        grid.addWidget(b, 3, 1)
        self.buttons.append(b)
        lay.addWidget(g)

        # ---------------------------------------------------------------- 3D model
        g = QGroupBox("3D model (.xyz)")
        gl_ = QVBoxLayout(g)
        self.mol_list = QListWidget()
        self.mol_list.setMaximumHeight(80)
        self.mol_list.setToolTip("Molecule geometries of this project (saved in the project file). Lay-flat, "
                                 "angle offset and handedness of each are set in the 3D model window.")
        mrow = QHBoxLayout()
        b = QPushButton("Add .xyz…")
        b.setToolTip("Load one or more molecule geometries (XYZ or extended XYZ, Å)")
        b.clicked.connect(lambda: self.add_molecules())
        mrow.addWidget(b)
        b = QPushButton("Remove")
        b.clicked.connect(self.remove_molecule)
        mrow.addWidget(b)
        mrow.addStretch(1)
        self.mol_map = QFormLayout()
        b = QPushButton("Build 3D model (.xyz)…")
        b.setToolTip("Place the molecules on the molecule annotations of this image, on a substrate from the "
                     "lattice fit and the aligned model; adjust and export")
        b.clicked.connect(self.build_model)
        self.buttons.append(b)
        gl_.addWidget(QLabel("Molecule files"))
        gl_.addWidget(self.mol_list)
        gl_.addLayout(mrow)
        gl_.addWidget(QLabel("Label → molecule"))
        gl_.addLayout(self.mol_map)
        gl_.addWidget(b)
        self._mol_sig: Any = None
        lay.addWidget(g)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        mark(self.status, "muted")
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.setWidget(body)

        session.imageChanged.connect(lambda _i: self.refresh())
        self._labels_timer = QTimer(self)                  # new labels can mean new molecule kinds
        self._labels_timer.setSingleShot(True)
        self._labels_timer.setInterval(400)
        self._labels_timer.timeout.connect(self.refresh_molecules)
        session.annotationsChanged.connect(lambda _e: self._labels_timer.start())
        session.labelsChanged.connect(self._labels_timer.start)
        session.imageDataChanged.connect(lambda _i: self.refresh())
        session.projectChanged.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ helpers
    @property
    def store(self) -> SurfaceStore:
        return SurfaceStore(self.session.project)

    def _image(self) -> Any:
        return self.session.image()

    def refresh(self) -> None:
        st = self.store
        s = st.settings
        for w, v in ((self.labels, ", ".join(s.get("molecule_labels") or [])),
                     (self.defects, ", ".join(s.get("defect_labels") or [])),
                     (self.chirality, ", ".join(f"{k}={v}" for k, v in (s.get("chirality") or {}).items()))):
            w.blockSignals(True)
            w.setText(v)
            w.blockSignals(False)
        for w, v in ((self.fold, int(s.get("mol_fold", 1))), (self.tol, float(s.get("site_tolerance", 0.12)))):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        for w, v in ((self.proposed, bool(s.get("include_proposed"))), (self.mirror, bool(s.get("mirror")))):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)
        self._fill_models()
        self.update_overlay()
        self._update_spacing_unit()
        self._mol_sig = None
        self.refresh_molecules()
        if self.preview.isChecked():
            self.overlay.set_preview(None)
            self._preview_inputs_changed()

    def _save_settings(self) -> None:
        st = self.store
        s = st.settings
        labs = [x.strip() for x in self.labels.text().split(",") if x.strip()]
        s["molecule_labels"] = labs or None
        s["defect_labels"] = [x.strip() for x in self.defects.text().split(",") if x.strip()]
        ch = {}
        for part in self.chirality.text().split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                v = v.strip().upper()[:1]
                if k.strip() and v in ("L", "R"):
                    ch[k.strip()] = v
        s["chirality"] = ch
        s["mol_fold"] = self.fold.value()
        s["site_tolerance"] = self.tol.value()
        s["include_proposed"] = self.proposed.isChecked()
        s["mirror"] = self.mirror.isChecked()
        st.changed()
        self.session._check_dirty()

    def update_overlay(self) -> None:
        iid = self.session.current_id
        img = None
        try:
            img = self._image() if iid else None
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            self.overlay.set_data(None, None, (1, 1))
            self.overlay.set_map(None)
            self.result.setText("No image.")
            return
        st = self.store
        fit = st.fit(iid)
        al = st.alignment(iid, fit) if fit is not None else None
        self.overlay.set_data(fit, al, img.shape)
        if fit is None:
            self._feedback(self.result, "No lattice fit on this image.", None)
        else:
            text, tone = self._fit_notes.get(iid, (fit.describe(), None))
            self._feedback(self.result, text, tone)
        self._fill_alignments()
        if al is not None:
            from ...surface.structure import alignment_warnings
            warn = alignment_warnings(al, fit.unit == "m")
            self._feedback(self.align_label, "In use: " + al.describe() + ("\n\n⚠ " + "\n⚠ ".join(warn) if warn else ""),
                           "warn" if warn else "ok")
        self.update_map()

    @staticmethod
    def _feedback(label: QLabel, text: str, tone: Optional[str] = None) -> None:
        label.setText(text)
        mark(label, tone)

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self._busy = busy
        for b in self.buttons:
            b.setEnabled(not busy)
        self.status.setText(text)

    # ------------------------------------------------------------------ lattice
    def fit_lattice(self) -> None:
        from ...surface.workflow import run_lattice_fit
        iid = self.session.current_id
        if self._busy:
            return
        img = self._image()
        if img is None:
            self._feedback(self.result, "Open an image first.", "error")
            return
        kw: dict[str, Any] = {"bright": self.contrast.currentIndex() == 0, "threshold": self.threshold.value()}
        sp = self._spacing_physical(img)
        if sp is not None:
            kw["spacing"] = sp
        start = self._start_vectors(img, iid)
        if start is not None:
            kw["a1"], kw["a2"] = start[0], start[1]
            kw.pop("spacing", None)                        # the vectors give the spacing
        st = self.store
        self._set_busy(True, "Fitting the lattice …")

        def done(fit: Any) -> None:
            from ...surface.workflow import fit_warnings
            self._set_busy(False, "")
            self.preview.setChecked(False)                 # the fitted sites replace the preview
            self._maps = {k: v for k, v in self._maps.items() if k[0] != iid}
            self.session._check_dirty()
            self.update_overlay()
            warn = fit_warnings(fit, kw.get("spacing"))
            desc = fit.describe()
            if start is not None:
                desc += f"\nStarted from the lattice vectors of {start[2]}."
            if warn:
                self._fit_notes[iid] = (desc + "\n\n⚠ " + "\n⚠ ".join(warn), "warn")
                self.session.info("Lattice fitted, with warnings: " + " ".join(warn))
            else:
                self._fit_notes[iid] = ("✓ " + desc, "ok")
                self.session.info(fit.describe())
            self.update_overlay()

        def failed(tb: str) -> None:
            from ...surface.workflow import lattice_fit_advice
            text = lattice_fit_advice(tb, img)
            self._set_busy(False, "")
            self._feedback(self.result, text.split("\n\n")[0], "error")
            QMessageBox.warning(self, "Lattice fit", text)
        run_job(run_lattice_fit, st, iid, done=done, failed=failed, **kw)

    def _start_vectors(self, img: Any, iid: str) -> Any:
        """(a1, a2, lattice name) physical starting vectors from the image's lattice-vector pair, if wanted."""
        from ...annotations.lattice_vectors import physical_pair
        from ...surface.lattice import image_scale
        if not self.from_vectors.isChecked() or not iid:
            return None
        return physical_pair(self.session.store.for_image(iid, kinds=("lattice_vector",)), image_scale(img)[0])

    def edit_lattice_vectors(self, lattice: Optional[str] = None) -> None:
        """Type in the lattice vectors of the current image."""
        from ..dialogs.surface_tools import LatticeVectorsDialog
        iid = self.session.current_id
        if not iid:
            self._feedback(self.result, "Open an image first.", "error")
            return
        dlg = LatticeVectorsDialog(self.session, iid, self.store.fit(iid), lattice, self)
        if dlg.exec():
            self.session.info(f"Lattice vectors of {dlg.lattice_name()} set: {dlg.summary()}")

    # spacing is shown in nm on calibrated images and in pixels otherwise
    @staticmethod
    def _in_metres(img: Any) -> bool:
        return bool(img is not None and img.calibrated and img.x.unit == "m")

    def _spacing_physical(self, img: Any) -> Optional[float]:
        v = self.spacing.value()
        if v <= 0 or img is None:
            return None
        if self._in_metres(img):
            return v * 1e-9
        return v * (img.pixel_size[0] if img.calibrated else 1.0)

    def _update_spacing_unit(self) -> None:
        img = self._image()
        nm = img is None or self._in_metres(img)
        self.spacing.setSuffix(" nm" if nm else " px")
        self.spacing.setDecimals(3 if nm else 1)
        self.spacing.setSingleStep(0.01 if nm else 0.1)

    def estimate_spacing(self) -> None:
        from ...surface.lattice import estimate_spacing, image_scale
        img = self._image()
        if img is None:
            return
        sp = estimate_spacing(img, image_scale(img)[0])
        if sp is None:
            self.status.setText("No clear periodicity in the Fourier transform: enter the spacing by hand.")
            return
        self.spacing.setValue(sp / 1e-9 if self._in_metres(img) else sp / (img.pixel_size[0] if img.calibrated else 1.0))

    # ------------------------------------------------------------------ live site preview
    def _preview_key(self) -> tuple:
        return (self.session.current_id, round(self.spacing.value(), 6), self.contrast.currentIndex())

    def _preview_toggled(self, on: bool) -> None:
        if on:
            self._preview_inputs_changed()
        else:
            self._preview_timer.stop()
            self.overlay.set_preview(None)
            self.preview_label.setText("")

    def _preview_inputs_changed(self) -> None:
        if self.preview.isChecked():
            self.preview_label.setText("finding sites …")
            self._preview_timer.start()

    def _threshold_changed(self, v: float) -> None:
        self.thr_slider.blockSignals(True)
        self.thr_slider.setValue(int(round(min(v, 10.0) * 100)))
        self.thr_slider.blockSignals(False)
        self._apply_preview()

    def _run_preview(self) -> None:
        from ...surface.workflow import preview_sites
        img = self._image()
        if not self.preview.isChecked() or img is None:
            return
        if self._preview_busy:                             # a run is going: try again when it is done
            self._preview_timer.start()
            return
        key = self._preview_key()
        if self._candidates is not None and self._candidates[0] == key:
            self._apply_preview()
            return
        self._preview_busy = True
        kw: dict[str, Any] = {"bright": self.contrast.currentIndex() == 0, "spacing": self._spacing_physical(img)}

        def done(res: Any) -> None:
            self._preview_busy = False
            self._candidates = (key, res[0], img.shape)
            self._apply_preview()

        def failed(tb: str) -> None:
            self._preview_busy = False
            self.preview_label.setText("preview failed: " + tb.strip().splitlines()[-1])
        run_job(preview_sites, self.store, key[0], done=done, failed=failed, **kw)

    def _apply_preview(self) -> None:
        c = self._candidates
        if not self.preview.isChecked() or c is None:
            return
        if c[0] != self._preview_key():
            self._preview_timer.start()
            return
        sites = c[1]
        ok = sites.score >= self.threshold.value()
        self.overlay.set_preview(sites.xy, ok, c[2])
        self.preview_label.setText(f"{int(ok.sum())} sites used, {int((~ok).sum())} below the threshold")

    def remove_fit(self) -> None:
        iid = self.session.current_id
        if iid:
            self.store.remove_fit(iid)
            self._fit_notes.pop(iid, None)
            self._maps = {k: v for k, v in self._maps.items() if k[0] != iid}
            self.session._check_dirty()
            self.update_overlay()

    def update_map(self) -> None:
        key = self.map.currentData()
        iid = self.session.current_id
        fit = self.store.fit(iid) if iid else None
        if not key or fit is None:
            self.overlay.set_map(None)
            return
        img = self._image()
        try:
            if (iid, key) not in self._maps:
                if key == "domains":
                    from ...surface.domains import find_domains
                    dr = find_domains(fit.sites, img, scale=tuple(fit.scale), shape=img.shape)
                    m = dr.label_map.astype(float)
                    m[dr.label_map < 0] = np.nan
                    self._maps[(iid, key)] = (m, True)
                else:
                    from ...surface.distortion import local_distortion
                    dm = self._maps.get((iid, "_distortion")) or local_distortion(fit)
                    self._maps[(iid, "_distortion")] = dm
                    self._maps[(iid, key)] = (dm.map(key, img.shape, min_confidence=0.5), False)
            m, cat = self._maps[(iid, key)]
            self.overlay.set_map(m, categorical=cat)
        except Exception as exc:  # noqa: BLE001
            self.overlay.set_map(None)
            self.session.warn(f"Map failed: {exc}")

    def calibrate(self) -> None:
        from ...surface.lattice import calibrate_from_lattice
        from ..dialogs.surface_tools import CalibrateDialog
        iid = self.session.current_id
        fit = self.store.fit(iid) if iid else None
        if fit is None:
            self.session.warn("Fit the lattice first.")
            return
        dlg = CalibrateDialog(fit, self)
        if not dlg.exec():
            return
        a, b, g, unc = dlg.values()
        r = calibrate_from_lattice(fit, a, b, g, unc)
        self.session.set_calibration(r["dx"], r["dy"], r["rel_uncertainty"])
        self.session.info(f"Pixel size {r['dx'] * 1e9:.5g} × {r['dy'] * 1e9:.5g} nm (± {100 * r['rel_uncertainty']:.2g} %); "
                          f"{r['note']}. Refit the lattice to update it.")

    # ------------------------------------------------------------------ model
    def _fill_models(self) -> None:
        cur = self.model.currentText()
        self.model.blockSignals(True)
        self.model.clear()
        names = [m.name for m in self.store.models()]
        self.model.addItems(names)
        iid = self.session.current_id
        d = self.store.data["images"].get(iid, {}).get("alignment") if iid else None
        want = d["model"] if d else cur
        if want in names:
            self.model.setCurrentIndex(names.index(want))
        self.model.blockSignals(False)
        self._fill_origin()

    def _model_changed(self) -> None:
        self._fill_alignments()
        self._fill_origin()

    def current_model(self) -> Any:
        name = self.model.currentText()
        return self.store.model(name) if name else None

    def _fill_alignments(self) -> None:
        from ...surface.structure import align
        self.align.clear()
        self._alignments = []
        iid = self.session.current_id
        fit = self.store.fit(iid) if iid else None
        m = self.current_model()
        if m is None:
            self._feedback(self.align_label, "No structure model yet: add one with Preset…, Import… or Edit….", "muted")
            return
        if fit is None:
            self._feedback(self.align_label, f"Fit the lattice of this image first: the model {m.name} is aligned "
                                             "to the fitted lattice.", "warn")
            return
        try:
            self._alignments = align(m, fit)
        except Exception as exc:  # noqa: BLE001
            self._feedback(self.align_label, f"Cannot align {m.name}: {exc}", "error")
            return
        for a in self._alignments:
            self.align.addItem(f"a along {a.rotation:.1f}°{' mirrored' if a.mirrored else ''}, "
                               f"{abs(np.linalg.det(a.T)):.0f} cell(s), scale {a.scale_ratio:.3f}, "
                               f"Δγ {a.angle_error:.1f}°")
        if not self._alignments:
            from ...surface.structure import explain_no_alignment
            self.align.addItem("(the model does not fit this lattice)")
            self._feedback(self.align_label, explain_no_alignment(m, fit), "error")
        elif self.store.alignment(iid, fit) is None:
            self._feedback(self.align_label, f"{len(self._alignments)} way(s) to align {m.name}: choose one and the "
                                             "position on the maxima, then press Use.", "muted")

    def _fill_origin(self) -> None:
        self.origin.clear()
        m = self.current_model()
        if m is None:
            return
        seen = set()
        for a in m.atoms:
            if a.layer == 0 and (a.u, a.v) not in seen:
                seen.add((a.u, a.v))
                self.origin.addItem(f"{a.species} atom ({a.u:.2f}, {a.v:.2f})", (a.u, a.v))
        for s in m.sites:
            if (s.u, s.v) not in seen:
                seen.add((s.u, s.v))
                self.origin.addItem(f"{s.name} ({s.u:.2f}, {s.v:.2f})", (s.u, s.v))

    def use_alignment(self) -> None:
        iid = self.session.current_id
        k = self.align.currentIndex()
        reason = None
        if not iid:
            reason = "Open an image first."
        elif self.current_model() is None:
            reason = "There is no structure model yet: add one with Preset…, Import… or Edit…."
        elif self.store.fit(iid) is None:
            reason = "Fit the lattice of this image first (Fit lattice): the model is aligned to the fitted lattice."
        elif not self._alignments:
            reason = self.align_label.text() or "The model does not fit this lattice."
        elif k < 0 or k >= len(self._alignments):
            reason = "Choose an alignment in the list first."
        if reason:
            self._feedback(self.align_label, reason, "error")
            QMessageBox.information(self, "Structure model", reason)
            return
        a = self._alignments[k]
        uv = self.origin.currentData() or (0.0, 0.0)
        self.store.set_alignment(iid, a.model.name, a.T, (-uv[0], -uv[1]), a.mirrored)
        self.session._check_dirty()
        self.chk["show_model"].setChecked(True)
        self.update_overlay()

    def clear_alignment(self) -> None:
        iid = self.session.current_id
        if iid:
            self.store.clear_alignment(iid)
            self.session._check_dirty()
            self.update_overlay()

    def _add_model(self, m: Any) -> None:
        self.store.set_model(m)
        self.session._check_dirty()
        self._fill_models()
        self.model.setCurrentText(m.name)
        self._model_changed()

    def add_preset(self) -> None:
        from ..dialogs.surface_tools import PresetDialog
        dlg = PresetDialog(self)
        if dlg.exec():
            try:
                self._add_model(dlg.model())
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Structure model", f"Could not make this model: {exc}")

    def import_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import structure model", "",
                                              "Models (*.json *.xyz *.extxyz);;All files (*)")
        if not path:
            return
        from ...surface.structure import SurfaceModel, from_extxyz
        try:
            m = SurfaceModel.load(path) if path.lower().endswith(".json") else from_extxyz(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Structure model",
                                f"Could not read {path}:\n{exc}\n\nA model is a ScanBound model JSON (Save… writes one) "
                                "or an ASE extended XYZ of the top layers with the surface cell in its Lattice= header.")
            return
        self._add_model(m)

    def edit_model(self) -> None:
        from ..dialogs.surface_tools import ModelEditor
        dlg = ModelEditor(self.current_model(), self)
        if dlg.exec() and dlg.result_model is not None:
            self._add_model(dlg.result_model)

    def save_model(self) -> None:
        m = self.current_model()
        if m is None:
            QMessageBox.information(self, "Structure model", "There is no model to save: add one with Preset…, "
                                                             "Import… or Edit… first.")
            return
        path = save_file_name(self, "Save structure model", f"{m.name}.json", "JSON (*.json)")
        if path:
            m.save(path)

    # ------------------------------------------------------------------ analyses
    def run(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> None:
        iid = self.session.current_id
        if not iid or self._busy:
            return
        self._image()                                  # processed image cached on this thread
        st = self.store
        self._set_busy(True, "Working …")

        def done(rep: Any) -> None:
            self._set_busy(False, "")
            self.session._check_dirty()
            self.reportReady.emit(rep)

        def failed(tb: str) -> None:
            msg = tb.strip().splitlines()[-1]
            self._set_busy(False, msg.split(": ", 1)[-1] if "SurfaceError" in msg else msg)
            if "SurfaceError" not in msg:
                print(tb)
        run_job(fn, st, *(args or (iid,)), done=done, failed=failed, **kw)

    def build_model(self) -> None:
        """Open the 3D model builder for the current image, after checking what it needs."""
        from ..dialogs.builder3d import Builder3DWindow
        iid = self.session.current_id
        missing = []
        if iid is None:
            missing.append("open an image")
        else:
            st = self.store
            fit = st.fit(iid)
            img = self._image()
            if fit is None:
                missing.append("fit the lattice (Fit lattice)")
            elif fit.unit != "m":
                missing.append("calibrate the pixel size (Calibrate…), then fit the lattice again")
            if not st.objects(iid, img):
                missing.append("add molecule annotations (feature points with a label that is not a defect label)")
        if missing:
            QMessageBox.information(self, "3D model", "Before building a 3D model:\n\n• " + "\n• ".join(missing))
            return
        if not load_molecules(self.store)[0]:
            self.add_molecules()
            if not load_molecules(self.store)[0]:
                return
        w = self._builders.get(iid)
        if w is None or not w.isVisible():
            w = Builder3DWindow(self.session, iid, self.window())
            w.on_molecules_changed = self.refresh_molecules
            self._builders[iid] = w
        w.show()
        w.raise_()

    # ------------------------------------------------------------------ molecule files
    def refresh_molecules(self) -> None:
        """Molecule list and label -> molecule choices from the project (only rebuilt when they change)."""
        if self.session.project is None:
            return
        temps, label_map = load_molecules(self.store)
        labels = self.store.molecule_labels()
        sig = (tuple((k, len(t.symbols)) for k, t in temps.items()), tuple(sorted(label_map.items())), tuple(labels))
        if sig == self._mol_sig:
            return
        self._mol_sig = sig
        self.mol_list.clear()
        for t in temps.values():
            self.mol_list.addItem(f"{t.name}  ({t.formula()}, {len(t.symbols)} atoms)")
        while self.mol_map.rowCount():
            self.mol_map.removeRow(0)
        if not labels:
            self.mol_map.addRow(QLabel("No molecule labels in this project yet."))
        names = list(temps)
        for lab in labels:
            cb = QComboBox()
            cb.addItem("(none)", "")
            for n in names:
                cb.addItem(n, n)
            want = label_map.get(lab) or (names[0] if len(names) == 1 else "")
            cb.setCurrentIndex(max(0, cb.findData(want)))
            cb.setToolTip("Molecule file placed on the annotations with this label")
            cb.currentIndexChanged.connect(lambda _i, lab=lab, cb=cb: self._assign(lab, cb.currentData()))
            self.mol_map.addRow(lab, cb)

    def _save_molecules(self, temps: dict, label_map: dict) -> None:
        save_molecules(self.store, temps, label_map)
        self.session._check_dirty()
        self.refresh_molecules()
        for w in list(self._builders.values()):
            if w.isVisible():
                w.reload_molecules()

    def add_molecules(self, paths: Optional[list[str]] = None) -> None:
        from ...surface.builder import MoleculeTemplate
        if self.session.project is None:
            return
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Molecule geometry", "",
                                                    "XYZ files (*.xyz *.extxyz);;All files (*)")
        temps, label_map = load_molecules(self.store)
        added = []
        for p in paths:
            try:
                t = MoleculeTemplate.from_xyz(p)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "3D model", f"Could not read {p}: {exc}")
                continue
            name, k = t.name, 2
            while name in temps:
                name, k = f"{t.name} {k}", k + 1
            t.name = name
            temps[name] = t
            added.append(name)
        if not added:
            return
        labels = self.store.molecule_labels()
        # unassigned labels whose name matches a file name get it (e.g. label "PTCDA", file PTCDA.xyz)
        for lab in labels:
            if not label_map.get(lab):
                hit = next((n for n in added if n.lower() == lab.lower()), None)
                if hit:
                    label_map[lab] = hit
        if len(temps) == 1:
            label_map = {lab: added[0] for lab in labels}
        self._save_molecules(temps, label_map)

    def remove_molecule(self) -> None:
        temps, label_map = load_molecules(self.store)
        k = self.mol_list.currentRow()
        names = list(temps)
        if not 0 <= k < len(names):
            return
        name = names[k]
        del temps[name]
        self._save_molecules(temps, {lab: n for lab, n in label_map.items() if n != name})

    def _assign(self, label: str, name: str) -> None:
        temps, label_map = load_molecules(self.store)
        label_map[label] = name
        self._save_molecules(temps, label_map)

    def explain(self) -> None:
        from ...surface.workflow import explain_object
        sel = [a for a in self.session.selected() if a.kind == "feature"]
        if len(sel) != 1:
            self.status.setText("Select one feature annotation to explain.")
            return
        self.run(explain_object, self.session.current_id, sel[0].id)
