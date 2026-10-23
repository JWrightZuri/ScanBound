"""
Building a 3D atomic model (.xyz) of what an image shows: the substrate slab from the fitted lattice
and its aligned structure model, and a molecule from a template .xyz on every molecule annotation,
at its position and orientation in the image.

    tpl = MoleculeTemplate.from_xyz("pentacene.xyz")          # centred, flattened, long axis along +x
    scene = build_scene(store, image_id, {"mol": tpl}, BuildSettings(layers=3, height=3.0))
    scene.placements[0].height = 2.6                           # Å between the surface and the lowest atom
    scene.placements[0].angle += 60                            # degrees, counter-clockwise seen from above
    scene.write_xyz("model.xyz")                               # extended XYZ with provenance in the comment line

Coordinates (Å): x to the right of the image, y up the image, z out of the surface; z = 0 is the
outermost substrate layer.  A molecule's angle is the annotation's orientation (counter-clockwise
from the image x axis) plus the template's own angle offset; a molecule whose handedness differs
from its template's is mirrored in the surface plane.  The result is a starting geometry for
calculations or simulated images, not a relaxed structure: heights are what you set.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

ANG = 1e-10

# covalent radii (Å, Cordero 2008) and CPK-like colours for drawing and bond / clash checks
COVALENT = {"H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
            "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
            "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
            "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20,
            "Br": 1.20, "Rb": 2.20, "Sr": 1.95, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45,
            "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
            "Ba": 2.15, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
            "Pb": 1.46, "Bi": 1.48}
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85,
       "I": 1.98, "B": 1.92, "Si": 2.10}
COLORS = {"H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "S": "#FFFF30",
          "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094", "P": "#FF8000", "B": "#FFB5B5", "Si": "#F0C8A0",
          "Cu": "#C88033", "Ag": "#C0C0C0", "Au": "#FFD123", "Pt": "#D0D0E0", "Pd": "#006985", "Ni": "#50D050",
          "Ga": "#C28F8F", "Al": "#BFA6A6", "Fe": "#E06633", "Co": "#F090A0", "Ir": "#175487", "Rh": "#0A7D8C",
          "W": "#2194D6", "Mo": "#54B5B5", "Ru": "#248F8F", "Zn": "#7D80B0", "Sn": "#668080", "In": "#A67573"}


def element(sym: str) -> str:
    """Normalise 'C1', 'ca', 'Au' to an element symbol (first letter upper, second lower)."""
    m = re.match(r"([A-Za-z]{1,2})", sym.strip())
    if not m:
        return "X"
    s = m.group(1)
    s = s[0].upper() + s[1:].lower()
    if s not in COVALENT and s[0] in COVALENT:
        return s[0]
    return s


def radius(sym: str) -> float:
    return COVALENT.get(sym, 1.5)


def color(sym: str) -> str:
    return COLORS.get(sym, "#FF69B4")


# --------------------------------------------------------------------------------------- xyz files
def read_xyz(path_or_text: str) -> tuple[list[str], np.ndarray, str]:
    """(symbols, positions (N, 3) Å, comment) of the first frame of an XYZ / extended-XYZ file
    (a path, or the file's text)."""
    if "\n" not in path_or_text and os.path.isfile(path_or_text):
        with open(path_or_text, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = path_or_text
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise ValueError("empty file")
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError("not an XYZ file: the first line must be the number of atoms") from None
    if len(lines) < 2 + n:
        raise ValueError(f"the file announces {n} atoms but has only {max(0, len(lines) - 2)} atom lines")
    comment = lines[1]
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"bad atom line: {ln!r}")
        syms.append(element(parts[0]))
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return syms, np.array(pos, float).reshape(-1, 3), comment


def write_xyz(path: str, symbols: Sequence[str], positions: np.ndarray, comment: str = "",
              lattice: Optional[np.ndarray] = None, groups: Optional[Sequence[int]] = None) -> None:
    """Plain XYZ, or extended XYZ when ``lattice`` (3 x 3, Å, rows = vectors) or ``groups`` is given."""
    P = np.asarray(positions, float).reshape(-1, 3)
    ext = lattice is not None or groups is not None
    if ext:
        head = []
        if lattice is not None:
            head.append('Lattice="' + " ".join(f"{v:.6f}" for v in np.asarray(lattice, float).ravel()) + '"')
        head.append("Properties=species:S:1:pos:R:3" + (":molecule:I:1" if groups is not None else ""))
        head.append('pbc="F F F"')
        if comment:                                  # free text as one quoted value, so ASE can parse the line
            head.append('comment="' + comment.replace('"', "'") + '"')
        comment = " ".join(head)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{len(P)}\n{comment.replace(chr(10), ' ')}\n")
        for k, (s, p) in enumerate(zip(symbols, P)):
            g = f" {int(groups[k])}" if groups is not None else ""
            fh.write(f"{s:<3s} {p[0]:14.6f} {p[1]:14.6f} {p[2]:14.6f}{g}\n")


def bonds(symbols: Sequence[str], positions: np.ndarray, tol: float = 1.2) -> np.ndarray:
    """(M, 2) index pairs closer than ``tol`` × the sum of covalent radii."""
    from scipy.spatial import cKDTree
    P = np.asarray(positions, float).reshape(-1, 3)
    if len(P) < 2:
        return np.zeros((0, 2), int)
    r = np.array([radius(s) for s in symbols])
    pairs = cKDTree(P).query_pairs(2 * tol * float(r.max()), output_type="ndarray")
    if not len(pairs):
        return np.zeros((0, 2), int)
    d = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
    keep = (d < tol * (r[pairs[:, 0]] + r[pairs[:, 1]])) & (d > 0.1)
    return pairs[keep]


# ---------------------------------------------------------------------------------------- templates
@dataclass
class MoleculeTemplate:
    """A molecule to place on the annotations.  ``positions``: Å, as read; the prepared geometry
    (centred, optionally flattened into the xy plane, long axis along +x) comes from ``geometry()``."""
    name: str
    symbols: list[str]
    positions: np.ndarray
    source: str = ""
    flatten: bool = True                  # rotate the molecular plane into the surface plane
    angle_offset: float = 0.0             # degrees added to every placement of this molecule
    flip: bool = False                    # turn the molecule upside down (its other face down)
    handedness: Optional[str] = None      # "L" / "R": the handedness of the geometry in the file

    @classmethod
    def from_xyz(cls, path_or_text: str, name: str = "", **kw: Any) -> "MoleculeTemplate":
        syms, P, _ = read_xyz(path_or_text)
        if not name:
            name = os.path.splitext(os.path.basename(path_or_text))[0] if "\n" not in path_or_text else "molecule"
        return cls(name, syms, P, path_or_text if "\n" not in path_or_text else "", **kw)

    def geometry(self) -> np.ndarray:
        """Prepared positions: centroid at the origin in x, y; lowest atom at z = 0."""
        P = np.asarray(self.positions, float).reshape(-1, 3)
        P = P - P.mean(axis=0)
        if self.flatten and len(P) >= 3:
            _, _, Vt = np.linalg.svd(P, full_matrices=False)
            ax, ay, n = Vt[0], Vt[1], Vt[2]
            # deterministic signs: the long axis points to the atom farthest along it; the normal keeps
            # the file's z direction when it has one
            far = P[np.argmax(np.abs(P @ ax))]
            if far @ ax < 0:
                ax = -ax
            if n[2] < 0 or (abs(n[2]) < 1e-9 and np.cross(ax, ay) @ n < 0):
                n = -n
            ay = np.cross(n, ax)
            P = P @ np.column_stack([ax, ay, n])
        if self.flip:
            P = P * np.array([1.0, -1.0, -1.0])          # 180° about the long axis: the other face down
        P[:, 2] -= P[:, 2].min()
        return P

    def footprint(self) -> float:
        """Radius of the molecule in the surface plane (Å, atom centres)."""
        g = self.geometry()
        return float(np.hypot(g[:, 0], g[:, 1]).max()) if len(g) else 0.0

    def formula(self) -> str:
        from collections import Counter
        c = Counter(self.symbols)
        order = [s for s in ("C", "H") if s in c] + sorted(s for s in c if s not in ("C", "H"))
        return "".join(f"{s}{c[s] if c[s] > 1 else ''}" for s in order)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "symbols": list(self.symbols), "positions": np.round(self.positions, 6).tolist(),
                "source": self.source, "flatten": self.flatten, "angle_offset": self.angle_offset,
                "flip": self.flip, "handedness": self.handedness}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MoleculeTemplate":
        return cls(d["name"], list(d["symbols"]), np.asarray(d["positions"], float).reshape(-1, 3), d.get("source", ""),
                   bool(d.get("flatten", True)), float(d.get("angle_offset", 0.0)), bool(d.get("flip", False)),
                   d.get("handedness"))


# ---------------------------------------------------------------------------------------- scene
@dataclass
class Placement:
    id: str                               # annotation id
    label: Optional[str]
    template: str                         # template name
    x: float                              # Å
    y: float                              # Å (up the image)
    angle: float = 0.0                    # degrees, counter-clockwise seen from above
    height: float = 3.0                   # Å from the outermost substrate layer to the lowest atom
    tilt: float = 0.0                     # degrees about the molecule's long axis
    mirrored: bool = False                # mirror image in the surface plane
    enabled: bool = True
    x0: float = 0.0                       # position from the image (before any move)
    y0: float = 0.0
    site: str = ""                        # assigned adsorption site, when snapped

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Placement":
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class BuildSettings:
    substrate: bool = True
    layers: int = 3                       # substrate layers (the model's stack, repeated downward)
    margin: float = 6.0                   # Å of substrate around the molecules
    extent: str = "molecules"             # "molecules" (their bounding box + margin) or "image"
    height: float = 3.0                   # default height of new placements, Å
    snap: bool = False                    # put molecules exactly on their assigned adsorption site
    species: str = "Au"                   # substrate element when no structure model is aligned
    include_proposed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "BuildSettings":
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in keys})


@dataclass
class Scene:
    templates: dict[str, MoleculeTemplate]
    placements: list[Placement]
    substrate_symbols: list[str] = field(default_factory=list)
    substrate_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    substrate_layers: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    settings: BuildSettings = field(default_factory=BuildSettings)
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ---- molecules
    def molecule(self, p: Placement) -> tuple[list[str], np.ndarray]:
        t = self.templates[p.template]
        g = t.geometry().copy()
        if p.tilt:
            a = math.radians(p.tilt)
            Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            g = g @ Rx.T
            g[:, 2] -= g[:, 2].min()
        if p.mirrored:
            g[:, 1] = -g[:, 1]
        th = math.radians(p.angle + t.angle_offset)
        R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
        g = g @ R.T
        g += np.array([p.x, p.y, p.height])
        return list(t.symbols), g

    def atoms(self, substrate: bool = True) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(symbols, positions Å, group): group -1 = substrate, k = the k-th placement."""
        syms: list[str] = []
        pos: list[np.ndarray] = []
        grp: list[np.ndarray] = []
        if substrate and len(self.substrate_positions):
            syms += list(self.substrate_symbols)
            pos.append(self.substrate_positions)
            grp.append(np.full(len(self.substrate_positions), -1))
        for k, p in enumerate(self.placements):
            if not p.enabled:
                continue
            s, g = self.molecule(p)
            syms += s
            pos.append(g)
            grp.append(np.full(len(g), k))
        if not pos:
            return [], np.zeros((0, 3)), np.zeros(0, int)
        return syms, np.vstack(pos), np.concatenate(grp)

    # ---- checks
    def check(self) -> list[dict[str, Any]]:
        """Per molecule: closest approach to the substrate and to other molecules, against the sum
        of van der Waals (molecule) / covalent (substrate) radii."""
        from scipy.spatial import cKDTree
        syms, P, G = self.atoms(True)
        out = []
        if not len(P):
            return out
        tree = cKDTree(P)
        for k, p in enumerate(self.placements):
            if not p.enabled:
                continue
            idx = np.nonzero(G == k)[0]
            d, j = tree.query(P[idx], k=min(len(P), 40))
            d = np.atleast_2d(d)
            j = np.atleast_2d(j)
            best = {"substrate": (math.inf, None), "molecule": (math.inf, None)}
            for a, (dd, jj) in enumerate(zip(d, j)):
                for dist, other in zip(dd, jj):
                    if other >= len(G) or G[other] == k:
                        continue
                    key = "substrate" if G[other] < 0 else "molecule"
                    sa, sb = syms[idx[a]], syms[other]
                    lim = (VDW.get(sa, 1.7) + VDW.get(sb, 1.7)) * 0.75 if key == "molecule" else \
                        (radius(sa) + radius(sb)) * 1.0
                    if dist / lim < best[key][0]:
                        best[key] = (dist / lim, (float(dist), int(G[other])))
            row = {"id": p.id, "label": p.label, "template": p.template, "height": p.height}
            for key in ("substrate", "molecule"):
                r, info = best[key]
                row[f"closest_{key}"] = info[0] if info else math.nan
                row[f"{key}_ratio"] = r if math.isfinite(r) else math.nan
                if key == "molecule" and info:
                    row["nearest_molecule"] = self.placements[info[1]].id
            row["clash"] = bool((row["substrate_ratio"] < 1.0) or (row["molecule_ratio"] < 1.0))
            out.append(row)
        return out

    # ---- output
    def cell(self) -> np.ndarray:
        """A box around everything (Å, rows = vectors) for the extended XYZ header; not periodic."""
        _, P, _ = self.atoms(True)
        if not len(P):
            return np.eye(3)
        span = P.max(axis=0) - P.min(axis=0) + np.array([0.0, 0.0, 10.0])
        return np.diag(np.maximum(span, 1.0))

    def write_xyz(self, path: str, extended: bool = True, substrate: bool = True) -> int:
        syms, P, G = self.atoms(substrate)
        info = {"generator": "ScanBound surface builder", **{k: v for k, v in self.provenance.items()
                                                             if isinstance(v, (str, int, float))}}
        comment = "; ".join(f"{k}={v}" for k, v in info.items())
        if extended:
            write_xyz(path, syms, P, comment, lattice=self.cell(), groups=G)
        else:
            write_xyz(path, syms, P, comment)
        side = os.path.splitext(path)[0] + "_placements.json"
        with open(side, "w", encoding="utf-8") as fh:
            json.dump({"placements": [p.to_dict() for p in self.placements], "settings": self.settings.to_dict(),
                       "templates": {k: {kk: vv for kk, vv in t.to_dict().items() if kk != "positions"}
                                     for k, t in self.templates.items()},
                       "provenance": self.provenance, "checks": _clean(self.check()), "notes": self.notes},
                      fh, indent=1)
        return len(P)


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ------------------------------------------------------------------------------------- building
def _to_xyz(xy_px: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Image pixels -> Å with y up the image."""
    P = np.asarray(xy_px, float).reshape(-1, 2) * np.asarray(scale, float) / ANG
    return np.column_stack([P[:, 0], -P[:, 1]])


def substrate_atoms(alignment: Any, box_px: tuple[float, float, float, float], layers: int = 3
                    ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Atoms of the aligned model inside a pixel box (x0, y0, x1, y1), its layer stack repeated down
    to ``layers`` layers: (symbols, positions Å with z = 0 at the top layer, layer index)."""
    fit = alignment.fit
    model = alignment.model
    x0, y0, x1, y1 = box_px
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)
    f = fit.to_fractional(corners)
    Tinv = np.linalg.inv(alignment.T)
    mc = (f @ Tinv.T) - alignment.offset
    lo, hi = np.floor(mc.min(0)) - 1, np.ceil(mc.max(0)) + 1
    if np.prod(hi - lo) > 4e5:
        raise ValueError("the substrate region is too large: reduce the margin or use the molecules' extent")
    m, n = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    cells = np.column_stack([m.ravel(), n.ravel()])
    stack = sorted({a.layer for a in model.atoms})
    heights = {L: float(np.mean([a.height for a in model.atoms if a.layer == L])) / ANG for L in stack}
    nL = len(stack)
    spacing = (heights[stack[0]] - heights[stack[-1]]) / (nL - 1) if nL > 1 else 0.0
    syms, pos, lay = [], [], []
    for k in range(max(1, int(layers))):
        if k >= nL and spacing <= 0:
            break
        base = stack[k % nL]
        dz = -(k // nL) * nL * spacing
        for atom in (a for a in model.atoms if a.layer == base):
            uv = cells + np.array([atom.u, atom.v])
            px = fit.to_pixel(alignment.model_to_fit(uv))
            ok = (px[:, 0] >= x0) & (px[:, 0] <= x1) & (px[:, 1] >= y0) & (px[:, 1] <= y1)
            xy = _to_xyz(px[ok], fit.scale)
            z = np.full(len(xy), atom.height / ANG + dz)
            syms += [element(atom.species)] * len(xy)
            pos.append(np.column_stack([xy, z]))
            lay.append(np.full(len(xy), k))
    if not pos:
        return [], np.zeros((0, 3)), np.zeros(0, int)
    P = np.vstack(pos)
    P[:, 2] -= heights[stack[0]]                 # top layer at z = 0
    return syms, P, np.concatenate(lay)


def placements_from_objects(objects: Iterable[Any], templates: dict[str, MoleculeTemplate],
                            label_map: dict[str, str], scale: tuple[float, float], height: float = 3.0
                            ) -> tuple[list[Placement], list[str]]:
    """One placement per object whose label has a template (``label_map``: label -> template name;
    with a single template every object uses it)."""
    out, skipped = [], []
    only = next(iter(templates)) if len(templates) == 1 else None
    for o in objects:
        name = label_map.get(o.label or "", "") or only
        if not name or name not in templates:
            skipped.append(o.id)
            continue
        t = templates[name]
        x, y = _to_xyz(o.xy[None], scale)[0]
        mirrored = bool(t.handedness and o.handedness and o.handedness != t.handedness)
        out.append(Placement(o.id, o.label, name, float(x), float(y), float(o.orientation or 0.0), float(height),
                             mirrored=mirrored, x0=float(x), y0=float(y)))
    return out, skipped


def build_scene(store: Any, image_id: str, templates: dict[str, MoleculeTemplate],
                label_map: Optional[dict[str, str]] = None, settings: Optional[BuildSettings] = None,
                previous: Optional[Sequence[Placement]] = None) -> Scene:
    """The scene of one image: substrate from the lattice fit and aligned model (or a one-atom
    model of ``settings.species`` on the image maxima), molecules on the molecule annotations.
    ``previous`` placements (same annotation ids) keep their manual adjustments."""
    from .adsorption import adsorption_registry
    from .structure import Alignment, Atom, SurfaceModel
    settings = settings or BuildSettings()
    if not templates:
        raise ValueError("Add at least one molecule .xyz file.")
    fit = store.fit(image_id)
    if fit is None:
        raise ValueError("This image has no lattice fit yet: fit the lattice first (Surface panel → Fit lattice).")
    if fit.unit != "m":
        raise ValueError("The image is not calibrated in metres: set the pixel size (or calibrate from the lattice) "
                         "and refit the lattice.")
    img = store.project.processed_image(image_id)
    st_props = store.settings.get("include_proposed")
    store.settings["include_proposed"] = bool(settings.include_proposed or st_props)
    try:
        objs = store.objects(image_id, img)
    finally:
        store.settings["include_proposed"] = st_props
    if not objs:
        raise ValueError("No molecule annotations on this image (reviewed feature annotations with a molecule label).")
    al = store.alignment(image_id, fit)
    notes = []
    if al is None:
        el = element(settings.species or "Au")
        a = float(np.linalg.norm(fit.a1))
        b = float(np.linalg.norm(fit.a2))
        from .lattice import gamma_deg
        model = SurfaceModel(f"{el} on the image maxima", a, b, gamma_deg(fit.a1, fit.a2), [Atom(el, 0.0, 0.0)])
        al = Alignment(model, fit, np.eye(2))
        notes.append(f"No structure model aligned: one {el} atom per image maximum, one layer.")
    labels = label_map or {}
    places, skipped = placements_from_objects(objs, templates, labels, tuple(fit.scale), settings.height)
    if skipped:
        notes.append(f"{len(skipped)} annotation(s) have no molecule assigned to their label and were left out.")
    if not places:
        raise ValueError("No annotation label is assigned to a molecule file.")
    if settings.snap:
        res = adsorption_registry([o for o in objs if o.id in {p.id for p in places}], fit, al,
                                  store.context(image_id, img), site_tolerance=store.settings.get("site_tolerance", 0.12))
        rows = {r["id"]: r for r in res.rows}
        for p in places:
            r = rows.get(p.id)
            if r and r["site"] != "none" and np.isfinite(r["site_offset_x"]):
                p.x -= r["site_offset_x"] / ANG
                p.y -= r["site_offset_y"] / ANG
                p.x0, p.y0, p.site = p.x, p.y, r["site"]
    prev = {p.id: p for p in (previous or [])}
    for k, p in enumerate(places):
        q = prev.get(p.id)
        if q is not None and q.template == p.template:
            dx, dy = q.x - q.x0, q.y - q.y0              # keep manual moves relative to the image position
            places[k] = Placement(**{**q.to_dict(), "x": p.x + dx, "y": p.y + dy, "x0": p.x, "y0": p.y,
                                     "site": p.site, "label": p.label})
    # substrate box
    syms, P, L = [], np.zeros((0, 3)), np.zeros(0, int)
    if settings.substrate:
        rows, cols = img.shape
        if settings.extent == "image":
            box = (0.0, 0.0, float(cols), float(rows))
        else:
            xy = np.array([[p.x, p.y] for p in places])
            reach = max(t.footprint() for t in templates.values()) + settings.margin
            lo = xy.min(0) - reach
            hi = xy.max(0) + reach
            sc = np.asarray(fit.scale) / ANG
            box = (lo[0] / sc[0], -hi[1] / sc[1], hi[0] / sc[0], -lo[1] / sc[1])
        syms, P, L = substrate_atoms(al, box, settings.layers)
    rel, ch = store.project.split_image_id(image_id)
    prov = {"image": rel, "channel": ch, "lattice": fit.describe(), "model": al.model.name,
            "alignment": al.describe(), "molecules": len(places),
            "templates": ", ".join(f"{t.name} ({t.formula()})" for t in templates.values())}
    try:
        prov["sha1"] = store.project.hash_file(rel)
    except Exception:  # noqa: BLE001
        pass
    return Scene(dict(templates), places, syms, P, L, settings, prov, notes)


# --------------------------------------------------------------------------------- project storage
def save_state(store: Any, image_id: str, scene: Optional[Scene] = None, templates: Optional[dict] = None,
               label_map: Optional[dict[str, str]] = None, settings: Optional[BuildSettings] = None) -> None:
    b = store.data.setdefault("builder", {})
    if templates is not None:
        b["molecules"] = {k: t.to_dict() for k, t in templates.items()}
    img = b.setdefault("images", {}).setdefault(image_id, {})
    if label_map is not None:                    # labels are project-wide, so is the label -> molecule map
        b["label_map"] = dict(label_map)
        img.pop("label_map", None)
    if settings is not None:
        img["settings"] = settings.to_dict()
    if scene is not None:
        img["placements"] = [p.to_dict() for p in scene.placements]
    store.changed()


def load_molecules(store: Any) -> tuple[dict[str, MoleculeTemplate], dict[str, str]]:
    """The project's molecule files and label -> molecule map (shared by every image)."""
    temps, label_map, _, _ = load_state(store, "")
    return temps, label_map


def save_molecules(store: Any, templates: dict[str, MoleculeTemplate], label_map: dict[str, str]) -> None:
    b = store.data.setdefault("builder", {})
    b["molecules"] = {k: t.to_dict() for k, t in templates.items()}
    b["label_map"] = dict(label_map)
    store.changed()


def load_state(store: Any, image_id: str) -> tuple[dict[str, MoleculeTemplate], dict[str, str], BuildSettings,
                                                    list[Placement]]:
    b = store.data.get("builder", {})
    temps = {}
    for k, d in (b.get("molecules") or {}).items():
        try:
            temps[k] = MoleculeTemplate.from_dict(d)
        except Exception:  # noqa: BLE001
            continue
    img = (b.get("images") or {}).get(image_id, {})
    label_map = dict(b.get("label_map") or img.get("label_map") or {})     # per-image maps of earlier versions
    return (temps, label_map, BuildSettings.from_dict(img.get("settings")),
            [Placement.from_dict(d) for d in img.get("placements") or []])
