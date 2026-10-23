"""Image information: calibration, quality warnings, processing history, source file and metadata."""

from __future__ import annotations

import json

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from ...model.units import format_value
from ..session import Session


class InfoDock(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Property", "Value"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setWordWrap(True)
        copy = QPushButton("Copy")
        copy.clicked.connect(self.copy)
        self.calibrate = QPushButton("Set pixel size…")
        row = QHBoxLayout()
        row.addWidget(self.calibrate)
        row.addStretch(1)
        row.addWidget(copy)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.tree, 1)
        lay.addLayout(row)
        session.imageChanged.connect(lambda _i: self.refresh())
        session.imageDataChanged.connect(lambda i: i == session.current_id and self.refresh())

    def _section(self, title: str, rows) -> None:  # noqa: ANN001
        top = QTreeWidgetItem([title, ""])
        for k, v in rows:
            it = QTreeWidgetItem([str(k), str(v)])
            it.setToolTip(1, str(v))
            top.addChild(it)
        self.tree.addTopLevelItem(top)
        top.setExpanded(True)

    def refresh(self) -> None:
        self.tree.clear()
        s = self.session
        try:
            img = s.image()
        except Exception as exc:  # noqa: BLE001
            self._section("Error", [("", str(exc))])
            return
        if img is None:
            return
        h, w = img.shape
        dx, dy = img.pixel_size
        ew, eh = img.extent
        st = img.stats
        rows = [("Channel", f"{img.quantity} ({img.unit or 'a.u.'})"), ("Size", f"{w} × {h} px")]
        if img.calibrated:
            rows += [("Pixel size", f"{format_value(dx, img.x.unit)} × {format_value(dy, img.y.unit)}"
                      + ("" if img.isotropic else "  (not square)")),
                     ("Scan size", f"{format_value(ew, img.x.unit)} × {format_value(eh, img.y.unit)}")]
        rows += [("Calibration x", img.x.calibration.describe()), ("Calibration y", img.y.calibration.describe()),
                 ("Range", f"{format_value(st.minimum, img.unit)} … {format_value(st.maximum, img.unit)}"),
                 ("Valid pixels", f"{100 * st.n / max(st.n + st.n_invalid, 1):.1f} %"), ("Stored as", str(img.raw_dtype or "float"))]
        if img.frame is not None:
            fr = img.frame
            rows.append(("Scan frame", f"centre ({format_value(fr.center[0], 'm')}, {format_value(fr.center[1], 'm')}), "
                                       f"angle {fr.angle:g}°"))
        self._section("Image", rows)
        q = img.quality()
        self._section("Quality", [(i.level, i.message) for i in q] or [("ok", "no issues found")])
        self._section("Processing (reproducible)", [(r.op, _params(r.params) + (f" — {r.note}" if r.note else ""))
                                                    for r in img.history] or [("raw", "no processing")])
        src = img.source
        self._section("Source", [("File", src.path), ("Format", src.format), ("Channel", src.channel),
                                 ("Direction", src.direction or "-"), ("SHA-256", src.sha256 or "(computed on export)")])
        md = img.metadata or {}
        self._section("File header", [(k, v) for k, v in list(md.items())[:400]])
        self.tree.topLevelItem(self.tree.topLevelItemCount() - 1).setExpanded(False)

    def copy(self) -> None:
        lines = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            lines.append(f"[{top.text(0)}]")
            for j in range(top.childCount()):
                c = top.child(j)
                lines.append(f"{c.text(0)}\t{c.text(1)}")
        QGuiApplication.clipboard().setText("\n".join(lines))


def _params(p: dict) -> str:
    try:
        text = json.dumps(p, default=str)
    except TypeError:
        text = str(p)
    return text if len(text) < 300 else text[:297] + "…"
