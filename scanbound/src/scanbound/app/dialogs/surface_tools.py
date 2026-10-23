"""Small dialogs of the surface analyses: choosing images, structure models, calibration, model editing."""

from __future__ import annotations

import json
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ..session import Session
from ..theme import mark


class ImageChooser(QDialog):
    """Tick images (and put them in order, e.g. time order)."""

    def __init__(self, session: Session, title: str, hint: str, ordered: bool = False, minimum: int = 2,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.minimum = minimum
        self.list = QListWidget()
        p = session.project
        cur = session.current_id
        ids = session.image_ids()
        ids = [cur] + [i for i in ids if i != cur] if cur in ids else ids
        for i in ids:
            rel, ch = p.split_image_id(i)
            it = QListWidgetItem(f"{rel}  [{ch}]" if ch else rel)
            it.setData(Qt.ItemDataRole.UserRole, i)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if i == cur else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        if ordered:
            self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        lab = QLabel(hint + (" Drag to put them in order; the first is the reference." if ordered else
                             " The first ticked image is the reference."))
        lab.setWordWrap(True)
        mark(lab, "muted")
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(lab)
        lay.addWidget(self.list, 1)
        lay.addWidget(bb)
        self.resize(520, 420)

    def chosen(self) -> list[str]:
        out = []
        for k in range(self.list.count()):
            it = self.list.item(k)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def _ok(self) -> None:
        if len(self.chosen()) >= self.minimum:
            self.accept()


class RegisterDialog(QDialog):
    def __init__(self, session: Session, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Register images")
        p = session.project
        self.other = QComboBox()
        for i in session.image_ids():
            if i != session.current_id:
                rel, ch = p.split_image_id(i)
                self.other.addItem(f"{rel}  [{ch}]" if ch else rel, i)
        self.method = QComboBox()
        for key, text in (("auto", "Automatic (lattices if both are fitted, else image correlation)"),
                          ("translation", "Translation (phase correlation)"),
                          ("rigid", "Rotation + scale + translation (log-polar)"),
                          ("lattice", "From the two lattice fits")):
            self.method.addItem(text, key)
        hint = QLabel("Finds the transformation from the current image to the other one, so molecules can be "
                      "followed or measured in both. Channels of one file need no registration.")
        hint.setWordWrap(True)
        mark(hint, "muted")
        form = QFormLayout()
        form.addRow("Other image", self.other)
        form.addRow("Method", self.method)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addLayout(form)
        lay.addWidget(bb)


class PresetDialog(QDialog):
    """A structure model from a built-in surface (fcc(111), graphene ...)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        from ...surface.structure import BULK_BCC, BULK_FCC, PRESETS
        self.setWindowTitle("Structure model from a preset")
        self.kind = QComboBox()
        self.kind.addItems(list(PRESETS))
        self.element = QComboBox()
        self.element.setEditable(True)
        self.element.addItems([""] + sorted(set(BULK_FCC) | set(BULK_BCC)))
        self.a = QDoubleSpinBox()
        self.a.setRange(0.0, 100.0)
        self.a.setDecimals(4)
        self.a.setSuffix(" Å")
        self.a.setSpecialValueText("from the element")
        self.a.setToolTip("Bulk lattice constant (fcc/bcc) or surface lattice constant (2D presets); 0 = tabulated")
        hint = QLabel("Element and lattice constant set the atom names and the expected cell. Surface "
                      "terminations of alloys (e.g. PdGa) are entered as their own model: Import or Edit.")
        hint.setWordWrap(True)
        mark(hint, "muted")
        form = QFormLayout()
        form.addRow("Surface", self.kind)
        form.addRow("Element", self.element)
        form.addRow("Lattice constant", self.a)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(hint)
        lay.addWidget(bb)

    def model(self) -> Any:
        from ...surface.structure import preset
        a = self.a.value() * 1e-10 if self.a.value() > 0 else None
        return preset(self.kind.currentText(), self.element.currentText().strip(), a)


MODEL_HELP = """A surface model as JSON. Lengths in metres, angles in degrees, positions as fractions of the cell:
{"name": "PdGa(-1-1-1) Pd3", "a": 6.95e-10, "b": 6.95e-10, "gamma": 120,
 "atoms": [{"species": "Pd", "u": 0.0, "v": 0.0, "layer": 0, "height": 0.0}, ...],
 "sites": [{"name": "Pd3 centre", "u": 0.33, "v": 0.67, "kind": "hollow"}, ...],
 "mirror_allowed": false, "notes": "termination from ..."}
Leave "sites" empty to generate tops, bridges and hollows from the atoms.  layer 0 is the outermost."""


class ModelEditor(QDialog):
    def __init__(self, model: Any = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit structure model")
        self.edit = QPlainTextEdit()
        self.edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        d = model.to_dict() if model is not None else {"name": "new model", "a": 2.8e-10, "b": 2.8e-10, "gamma": 120.0,
                                                      "atoms": [{"species": "X", "u": 0, "v": 0, "layer": 0}],
                                                      "sites": [], "mirror_allowed": True, "notes": ""}
        self.edit.setPlainText(json.dumps(d, indent=1))
        help_ = QLabel(MODEL_HELP)
        help_.setWordWrap(True)
        help_.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mark(help_, "muted")
        self.error = QLabel("")
        mark(self.error, "error")
        regen = QPushButton("Regenerate sites")
        regen.setToolTip("Replace the sites with tops, bridges and hollows computed from the atoms")
        regen.clicked.connect(self._regen)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(regen)
        row.addWidget(self.error, 1)
        row.addWidget(bb)
        lay = QVBoxLayout(self)
        lay.addWidget(help_)
        lay.addWidget(self.edit, 1)
        lay.addLayout(row)
        self.resize(720, 620)
        self.result_model: Any = None

    def _parse(self) -> Any:
        from ...surface.structure import SurfaceModel
        return SurfaceModel.from_dict(json.loads(self.edit.toPlainText()))

    def _regen(self) -> None:
        from ...surface.structure import generate_sites
        try:
            m = self._parse()
            m.sites = generate_sites(m)
            self.edit.setPlainText(json.dumps(m.to_dict(), indent=1))
            self.error.setText("")
        except Exception as exc:  # noqa: BLE001
            self.error.setText(str(exc))

    def _ok(self) -> None:
        try:
            self.result_model = self._parse()
        except Exception as exc:  # noqa: BLE001
            self.error.setText(f"Not a valid model: {exc}")
            return
        self.accept()


class CalibrateDialog(QDialog):
    def __init__(self, fit: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate from a known lattice")
        ms = fit.measurements()
        self.a = QDoubleSpinBox()
        self.a.setRange(0.01, 1000)
        self.a.setDecimals(4)
        self.a.setSuffix(" Å")
        self.a.setValue(ms["a1"].value * 1e10 if fit.unit == "m" else 2.5)
        self.b = QDoubleSpinBox()
        self.b.setRange(0.0, 1000)
        self.b.setDecimals(4)
        self.b.setSuffix(" Å")
        self.b.setSpecialValueText("same as a")
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.0, 180)
        self.gamma.setDecimals(2)
        self.gamma.setSuffix(" °")
        self.gamma.setSpecialValueText("not used")
        self.unc = QDoubleSpinBox()
        self.unc.setRange(0, 50)
        self.unc.setDecimals(2)
        self.unc.setSuffix(" %")
        lab = QLabel(f"Fitted: {fit.describe()}\nEnter the known surface lattice (e.g. Au(111): 2.884 Å, 120°). "
                     "The pixel size is set so the fitted lattice matches it; a leftover shear is reported, "
                     "since drift cannot be removed by a pixel size.")
        lab.setWordWrap(True)
        mark(lab, "muted")
        form = QFormLayout()
        form.addRow("Known a", self.a)
        form.addRow("Known b", self.b)
        form.addRow("Known γ", self.gamma)
        form.addRow("Uncertainty of a", self.unc)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(lab)
        lay.addLayout(form)
        lay.addWidget(bb)

    def values(self) -> tuple[float, Optional[float], Optional[float], float]:
        return (self.a.value() * 1e-10, self.b.value() * 1e-10 if self.b.value() > 0 else None,
                self.gamma.value() if self.gamma.value() > 0 else None, self.unc.value() / 100)


def text_value(parent: QWidget, title: str, label: str, value: str) -> Optional[str]:
    from PyQt6.QtWidgets import QInputDialog
    v, ok = QInputDialog.getText(parent, title, label, QLineEdit.EchoMode.Normal, value)
    return v if ok else None


class LatticeVectorsDialog(QDialog):
    """Type in a1 and a2 of a lattice (lengths, angles, origin); writes the two lattice-vector annotations."""

    NEW = "New lattice"

    def __init__(self, session: Session, image_id: str, fit: Any = None, lattice: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        from ...annotations import lattice_vectors as lv
        from ...surface.lattice import image_scale
        from ..tools import lattice_groups
        super().__init__(parent)
        self.setWindowTitle("Lattice vectors")
        self.session, self.image_id, self.fit = session, image_id, fit
        import numpy as np
        img = session.image(image_id)
        self.shape = tuple(np.shape(getattr(img, "data", img))[:2]) if img is not None else (0, 0)
        scale, unit, _rel, _notes = image_scale(img)
        self.scale = scale
        self.unit = unit
        self.factor, self.shown = (1e9, "nm") if unit == "m" else (1.0, unit)
        self.groups = lattice_groups(session.store.for_image(image_id, kinds=("lattice_vector",)))

        self.lattice = QComboBox()
        self.lattice.addItems(list(self.groups) + [self.NEW])
        self.lattice.setToolTip("The lattice to set: one drawn on this image, or a new pair")

        def spin(lo: float, hi: float, dec: int, suffix: str, step: float) -> QDoubleSpinBox:
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSuffix(suffix)
            w.setSingleStep(step)
            w.setKeyboardTracking(False)
            return w
        nm = self.shown == "nm"
        dec, step = (4, 0.001) if nm else (2, 0.1)
        self.l1 = spin(0.0, 1e6, dec, f" {self.shown}", step)
        self.l2 = spin(0.0, 1e6, dec, f" {self.shown}", step)
        self.t1 = spin(-360.0, 360.0, 2, " °", 1.0)
        self.t1.setToolTip("Direction of a1, counter-clockwise from the image x axis (0° right, 90° up)")
        self.gamma = spin(-180.0, 180.0, 2, " °", 1.0)
        self.gamma.setToolTip("Angle from a1 to a2, counter-clockwise positive (e.g. 60° or 120° for hexagonal, "
                              "90° for square)")
        self.ox = spin(-1e6, 1e6, 1, " px", 1.0)
        self.oy = spin(-1e6, 1e6, 1, " px", 1.0)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("x"))
        orow.addWidget(self.ox, 1)
        orow.addWidget(QLabel("y"))
        orow.addWidget(self.oy, 1)

        from_fit = QPushButton("From the lattice fit")
        from_fit.setEnabled(fit is not None)
        from_fit.setToolTip("Fill in a1, a2 and the origin of this image's lattice fit" if fit is not None
                            else "This image has no lattice fit")
        from_fit.clicked.connect(lambda: self._fill(lv.values_from_fit(fit)))
        swap = QPushButton("Swap a1 ↔ a2")
        swap.clicked.connect(lambda: self._fill(self.values().swapped()))
        brow = QHBoxLayout()
        brow.addWidget(from_fit)
        brow.addWidget(swap)
        brow.addStretch(1)

        cal = "" if nm or unit != "px" else " The image is not calibrated, so lengths are in pixels."
        lab = QLabel("Enter the lattice vectors if you know them (e.g. from LEED, the literature or another image). "
                     "Angles are counter-clockwise from the image x axis; γ is the angle from a1 to a2. a1 of the "
                     "first lattice is the reference direction for orientations, and the vectors are the starting "
                     "guess of Fit lattice (tick 'Start from the lattice vectors' in the Surface panel)." + cal)
        lab.setWordWrap(True)
        mark(lab, "muted")
        self.error = QLabel("")
        self.error.setWordWrap(True)
        mark(self.error, "error")
        form = QFormLayout()
        form.addRow("Lattice", self.lattice)
        form.addRow("|a1|", self.l1)
        form.addRow("a1 angle", self.t1)
        form.addRow("|a2|", self.l2)
        form.addRow("γ (a1 → a2)", self.gamma)
        form.addRow("Origin", orow)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(lab)
        lay.addLayout(form)
        lay.addLayout(brow)
        lay.addWidget(self.error)
        lay.addWidget(bb)

        self.lattice.currentTextChanged.connect(self._lattice_changed)
        if lattice in self.groups:
            self.lattice.setCurrentText(lattice)
        self._lattice_changed(self.lattice.currentText())

    # ---- values ------------------------------------------------------------------------------
    def _default(self) -> Any:
        from ...annotations import lattice_vectors as lv
        if self.fit is not None:
            return lv.values_from_fit(self.fit)
        h, w = self.shape if self.shape != (0, 0) else (100, 100)
        length = 1e-9 if self.unit == "m" else 20.0 * self.scale[0]
        return lv.LatticeValues(length, 0.0, length, 60.0, (w / 2.0, h / 2.0), self.unit)

    def _lattice_changed(self, name: str) -> None:
        from ...annotations import lattice_vectors as lv
        vecs = self.groups.get(name, [])
        v = lv.values_from_pair(vecs, self.scale, self.unit) if len(vecs) >= 2 else None
        if v is None:
            d = self._default()
            if len(vecs) == 1:                               # a1 drawn, a2 missing: keep a1, guess a2
                one = lv.values_from_pair([vecs[0], vecs[0]], self.scale, self.unit)
                d = one.with_gamma(d.gamma if abs(d.gamma) > 5 else 60.0)
            v = d
        self._fill(v)

    def _fill(self, v: Any) -> None:
        self.l1.setValue(v.a1_length * self.factor)
        self.l2.setValue(v.a2_length * self.factor)
        t = v.a1_angle % 360.0
        self.t1.setValue(t - 360.0 if t > 180.0 else t)
        self.gamma.setValue(v.gamma)
        self.ox.setValue(v.origin[0])
        self.oy.setValue(v.origin[1])
        self.error.setText("")

    def values(self) -> Any:
        from ...annotations import lattice_vectors as lv
        base = lv.LatticeValues(self.l1.value() / self.factor, self.t1.value() % 360.0, self.l2.value() / self.factor,
                                0.0, (self.ox.value(), self.oy.value()), self.unit)
        return base.with_gamma(self.gamma.value())

    def lattice_name(self) -> str:
        name = self.lattice.currentText()
        if name != self.NEW:
            return name
        k = 1
        while f"lattice {k}" in self.groups:
            k += 1
        return f"lattice {k}"

    def summary(self) -> str:
        v = self.values()
        return (f"|a1| = {v.a1_length * self.factor:.4g} {self.shown} at {self.t1.value():.1f}°, "
                f"|a2| = {v.a2_length * self.factor:.4g} {self.shown}, γ = {v.gamma:.1f}°")

    def apply(self) -> list[str]:
        """Write the annotations; the problems that stopped it (empty when written)."""
        from ...annotations import lattice_vectors as lv
        v = self.values()
        problems = v.problems()
        if problems:
            return problems
        name = self.lattice_name()
        new, updated = lv.annotations_for(v, self.image_id, self.scale, name, self.groups.get(name, []))
        store = self.session.store
        with store.transaction(f"Lattice vectors ({name})"):
            for a in updated:
                store.replace(a)
            if new:
                store.add(new)
        return []

    def _ok(self) -> None:
        problems = self.apply()
        if problems:
            self.error.setText(" ".join(problems))
            return
        self.accept()
