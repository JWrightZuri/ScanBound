"""
Surface structure models: a 2D cell with atoms and adsorption sites, aligned to a fitted lattice.

    model = preset("fcc(111)", element="Cu")          # or SurfaceModel.load("PdGa_111_Pd3.json")
    model.sites                                       # top / bridge / hollow sites in the cell (generated)
    al = align(model, fit)[0]                         # best of the symmetry-equivalent alignments
    al.atom_pixels(image.shape)                       # where the model puts its atoms in the image
    al.site_fractional()                              # the model's sites in the fitted lattice's coordinates
    al.scale_ratio                                    # fitted / model lattice constant (a calibration check)

Models are plain JSON (``SurfaceModel.save`` / ``load``) so a surface termination worked out
elsewhere (DFT, a structure database, ASE) can be typed in once and reused.  ``from_extxyz`` reads
the top layers of a slab written by ASE (``ase.io.write("slab.xyz", slab)``).

Alignment and its ambiguity
---------------------------
An image of one terrace fixes the lattice but not, in general, which of the symmetry-equivalent
orientations of a lower-symmetry termination it shows, nor whether a bright protrusion is an atom,
a hollow or something else.  :func:`align` therefore returns *every* alignment consistent with the
lattice (rotations by the lattice symmetry, and the mirror image when allowed), ranked only by how
well the cell shape matches; the first one is not more true than the others when they tie.  The
user picks one (and the origin, ``offset``) from other evidence: step directions, a known
adsorbate, a second technique.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .lattice import LatticeFit, gamma_deg


@dataclass
class Atom:
    species: str
    u: float
    v: float
    layer: int = 0                      # 0 = outermost, 1 = the layer below ...
    height: float = 0.0                 # m, relative to the outermost layer (negative below)

    def to_dict(self) -> dict[str, Any]:
        return {"species": self.species, "u": self.u, "v": self.v, "layer": self.layer, "height": self.height}


@dataclass
class SiteDef:
    name: str                           # "top", "bridge", "hollow (fcc)" ...
    u: float
    v: float
    kind: str = "custom"                # "top" / "bridge" / "hollow" / "custom"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "u": self.u, "v": self.v, "kind": self.kind}


@dataclass
class SurfaceModel:
    name: str
    a: float                            # m
    b: float                            # m
    gamma: float                        # degrees, angle between a and b
    atoms: list[Atom] = field(default_factory=list)
    sites: list[SiteDef] = field(default_factory=list)
    notes: str = ""
    mirror_allowed: bool = True         # False for chiral terminations whose handedness is known

    def __post_init__(self) -> None:
        if not self.sites and self.atoms:
            self.sites = generate_sites(self)

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell vectors in an x-right / y-down frame with a along +x and b counter-clockwise on screen."""
        g = math.radians(self.gamma)
        return np.array([self.a, 0.0]), np.array([self.b * math.cos(g), -self.b * math.sin(g)])

    def cartesian(self, uv: np.ndarray) -> np.ndarray:
        a, b = self.basis
        uv = np.asarray(uv, float).reshape(-1, 2)
        return uv[:, :1] * a + uv[:, 1:2] * b

    def to_dict(self) -> dict[str, Any]:
        return {"format": "scanbound_surface_model", "name": self.name, "a": self.a, "b": self.b, "gamma": self.gamma,
                "atoms": [x.to_dict() for x in self.atoms], "sites": [s.to_dict() for s in self.sites],
                "notes": self.notes, "mirror_allowed": self.mirror_allowed}

    @classmethod
    def from_dict(cls, d: dict) -> "SurfaceModel":
        atoms = [Atom(str(x["species"]), float(x["u"]), float(x["v"]), int(x.get("layer", 0)),
                      float(x.get("height", 0.0))) for x in d.get("atoms", [])]
        sites = [SiteDef(str(s["name"]), float(s["u"]), float(s["v"]), str(s.get("kind", "custom")))
                 for s in d.get("sites", [])]
        return cls(str(d.get("name", "model")), float(d["a"]), float(d.get("b", d["a"])), float(d.get("gamma", 90.0)),
                   atoms, sites, str(d.get("notes", "")), bool(d.get("mirror_allowed", True)))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "SurfaceModel":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if path.lower().endswith(".xyz"):
            raise ValueError("use from_extxyz for .xyz files")
        return cls.from_dict(d)


# ======================================================================== sites
def _periodic(uv: np.ndarray, reps: int = 1) -> tuple[np.ndarray, np.ndarray]:
    shifts = np.array(list(itertools.product(range(-reps, reps + 1), repeat=2)), float)
    pts = (uv[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    owner = np.tile(np.arange(len(uv)), len(shifts))
    return pts, owner


def _dedupe(sites: list[SiteDef], model: SurfaceModel, tol: float = 0.02) -> list[SiteDef]:
    out: list[SiteDef] = []
    for s in sites:
        u, v = s.u % 1.0, s.v % 1.0
        dup = False
        for o in out:
            du, dv = (u - o.u + 0.5) % 1 - 0.5, (v - o.v + 0.5) % 1 - 0.5
            if np.linalg.norm(model.cartesian(np.array([[du, dv]]))) < tol * min(model.a, model.b):
                dup = True
                break
        if not dup:
            out.append(SiteDef(s.name, round(u, 6) % 1.0, round(v, 6) % 1.0, s.kind))
    return out


def generate_sites(model: SurfaceModel) -> list[SiteDef]:
    """
    Top sites on every outermost-layer atom, bridge sites between nearest neighbours and hollow
    sites in the triangles (or squares) they form.  A hollow with a second-layer atom right below
    is named "hollow (above layer-2 <species>)", one without "hollow (no atom below)" - for fcc(111)
    these are the hcp and fcc hollows.
    """
    top = [a for a in model.atoms if a.layer == 0]
    if not top:
        return []
    uv = np.array([[a.u, a.v] for a in top])
    sites = [SiteDef(f"top {a.species}" if len({t.species for t in top}) > 1 else "top", a.u, a.v, "top")
             for a in top]
    pts, owner = _periodic(uv, 1)
    xy = model.cartesian(pts)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    dmin = float(d.min())
    # bridges: midpoints of nearest-neighbour pairs with at least one end inside the home cell, plus
    # "long bridges" across the second-shortest distance when it is short (rectangular cells)
    home = np.all((pts >= -1e-9) & (pts < 1 - 1e-9), axis=1)
    dists = np.unique(np.round(d[np.isfinite(d)] / dmin, 3))
    d2 = float(dists[1]) * dmin if len(dists) > 1 and dists[1] < 1.6 else None
    multi = len({t.species for t in top}) > 1
    for i, j in zip(*np.nonzero(d < (d2 * 1.02 if d2 else 1.12 * dmin))):
        if i < j and (home[i] or home[j]):
            m = 0.5 * (pts[i] + pts[j])
            sp = {top[owner[i]].species, top[owner[j]].species}
            long_ = d[i, j] > 1.12 * dmin
            name = ("long bridge" if long_ else "bridge") + (f" {'-'.join(sorted(sp))}" if multi else "")
            sites.append(SiteDef(name, m[0], m[1], "bridge"))
    # hollows: local maxima of the distance to the nearest outermost atom (triangle centres on
    # hexagonal cells, cell centres on square and rectangular ones, hexagon centres on honeycombs)
    below = [a for a in model.atoms if a.layer == 1]
    buv = np.array([[a.u, a.v] for a in below]).reshape(-1, 2)
    bpts, bown = _periodic(buv, 1) if len(buv) else (np.zeros((0, 2)), np.zeros(0, int))
    bxy = model.cartesian(bpts)
    n = 72
    g = (np.arange(n) + 0.0) / n
    gu, gv = np.meshgrid(g, g, indexing="ij")
    G = model.cartesian(np.column_stack([gu.ravel(), gv.ravel()]))
    from scipy.spatial import cKDTree
    from scipy import ndimage
    dist = cKDTree(xy).query(G)[0].reshape(n, n)
    peak = (dist == ndimage.maximum_filter(dist, size=5, mode="wrap")) & (dist > 0.3 * dmin)
    hollows = []
    for i, j in zip(*np.nonzero(peak)):
        c = np.array([g[i], g[j]])
        for q in range(2):                                   # snap to simple fractions
            for den in (2, 3, 4, 6):
                r = round(c[q] * den) / den
                if abs(r - c[q]) < 1.5 / n:
                    c[q] = r % 1.0
                    break
        name = "hollow"
        if len(bxy):
            cx = model.cartesian(c[None, :])[0]
            dd = np.linalg.norm(bxy - cx, axis=1)
            if dd.min() < 0.15 * dmin:
                name = f"hollow (above layer-2 {below[bown[int(dd.argmin())]].species})"
            else:
                name = "hollow (no atom below)"
        hollows.append(SiteDef(name, c[0], c[1], "hollow"))
    bridges = sites[len(top):]
    sites = sites[:len(top)] + hollows + sorted(bridges, key=lambda x: x.name.startswith("long"))
    return _dedupe(sites, model)


# ======================================================================== presets
BULK_FCC = {"Cu": 3.615e-10, "Ag": 4.086e-10, "Au": 4.078e-10, "Pt": 3.924e-10, "Pd": 3.891e-10, "Ni": 3.524e-10,
            "Al": 4.050e-10, "Ir": 3.839e-10, "Rh": 3.803e-10}
BULK_BCC = {"W": 3.165e-10, "Fe": 2.867e-10, "Mo": 3.147e-10}


def preset(name: str, element: str = "", a_bulk: Optional[float] = None) -> SurfaceModel:
    """
    Standard terminations: "fcc(111)", "fcc(100)", "fcc(110)", "bcc(100)", "hcp(0001)",
    "graphene", "hexagonal", "square", "rectangular" (generic one-atom cells; give ``a_bulk`` as the
    surface lattice constant for those).  ``element`` picks a bulk lattice constant from a short
    built-in table (room-temperature values; override with ``a_bulk``).
    """
    el = element or "X"
    if name in ("fcc(111)", "fcc(100)", "fcc(110)"):
        ab = a_bulk or BULK_FCC.get(element)
        if ab is None:
            raise ValueError(f"no bulk lattice constant for {element!r}: give a_bulk")
        s = ab / math.sqrt(2)
        if name == "fcc(111)":
            return SurfaceModel(f"{el}(111)", s, s, 120.0,
                                [Atom(el, 0, 0, 0), Atom(el, 2 / 3, 1 / 3, 1, -ab / math.sqrt(3)),
                                 Atom(el, 1 / 3, 2 / 3, 2, -2 * ab / math.sqrt(3))],
                                notes="fcc(111): layer 2 below one hollow (hcp), layer 3 below the other (fcc). "
                                      "Which hollow is which in the image depends on the stacking orientation, "
                                      "which one terrace does not show: check against step-edge directions.")
        if name == "fcc(100)":
            return SurfaceModel(f"{el}(100)", s, s, 90.0, [Atom(el, 0, 0, 0), Atom(el, 0.5, 0.5, 1, -ab / 2)])
        return SurfaceModel(f"{el}(110)", s, ab, 90.0, [Atom(el, 0, 0, 0), Atom(el, 0.5, 0.5, 1, -s / 2)],
                            notes="a along the close-packed [1-10] rows, b along [001]")
    if name == "bcc(100)":
        ab = a_bulk or BULK_BCC.get(element)
        if ab is None:
            raise ValueError(f"no bulk lattice constant for {element!r}: give a_bulk")
        return SurfaceModel(f"{el}(100)", ab, ab, 90.0, [Atom(el, 0, 0, 0), Atom(el, 0.5, 0.5, 1, -ab / 2)])
    if name == "hcp(0001)":
        if a_bulk is None:
            raise ValueError("give the hcp lattice constant a as a_bulk")
        return SurfaceModel(f"{el}(0001)", a_bulk, a_bulk, 120.0,
                            [Atom(el, 0, 0, 0), Atom(el, 2 / 3, 1 / 3, 1, -1.633 * a_bulk / 2)])
    if name == "graphene":
        a = a_bulk or 2.46e-10
        return SurfaceModel("graphene", a, a, 120.0, [Atom("C", 0, 0, 0), Atom("C", 2 / 3, 1 / 3, 0)],
                            notes="two carbon atoms per cell; on graphite (HOPG) STM usually shows only one "
                                  "sublattice")
    if name in ("hexagonal", "square", "rectangular"):
        if a_bulk is None:
            raise ValueError("give the surface lattice constant as a_bulk")
        g = {"hexagonal": 120.0, "square": 90.0, "rectangular": 90.0}[name]
        return SurfaceModel(f"{name} {el}", a_bulk, a_bulk, g, [Atom(el, 0, 0, 0)])
    raise ValueError(f"unknown preset {name!r}")


PRESETS = ("fcc(111)", "fcc(100)", "fcc(110)", "bcc(100)", "hcp(0001)", "graphene", "hexagonal", "square",
           "rectangular")


def from_extxyz(path: str, layers: int = 2, layer_tol: float = 0.3e-10, name: str = "") -> SurfaceModel:
    """
    The outermost ``layers`` atomic layers of a slab in extended XYZ (as written by ASE), with the
    surface normal along z and the first two cell vectors in the surface plane.  Positions in Å.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    header = lines[1]
    m = re.search(r'Lattice="([^"]+)"', header)
    if not m:
        raise ValueError("no Lattice=\"...\" in the second line: write the slab with ase.io.write(..., format='extxyz')")
    L = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3) * 1e-10
    species, pos = [], []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        species.append(parts[0])
        pos.append([float(x) for x in parts[1:4]])
    P = np.array(pos) * 1e-10
    z = P[:, 2]
    order = np.argsort(-z)
    layer_z: list[float] = []
    layer_of = np.full(n, -1)
    for i in order:
        for k, zz in enumerate(layer_z):
            if abs(z[i] - zz) < layer_tol:
                layer_of[i] = k
                break
        else:
            if len(layer_z) >= layers:
                continue
            layer_z.append(z[i])
            layer_of[i] = len(layer_z) - 1
    a3, b3 = L[0], L[1]
    a, b = float(np.linalg.norm(a3[:2])), float(np.linalg.norm(b3[:2]))
    gamma = math.degrees(math.acos(np.clip(np.dot(a3[:2], b3[:2]) / (a * b), -1, 1)))
    M = np.column_stack([a3[:2], b3[:2]])
    atoms = []
    for i in range(n):
        if layer_of[i] < 0:
            continue
        u, v = np.linalg.solve(M, P[i, :2]) % 1.0
        atoms.append(Atom(species[i], float(u), float(v), int(layer_of[i]), float(z[i] - layer_z[0])))
    return SurfaceModel(name or path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1], a, b, gamma, atoms,
                        notes=f"top {layers} layer(s) of {path}")


# ======================================================================== alignment
@dataclass
class Alignment:
    model: SurfaceModel
    fit: LatticeFit
    T: np.ndarray                       # model cell vectors in the fitted basis: [ma mb] = [a1 a2] T
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))   # model fractional shift of the origin
    length_error: float = 0.0           # relative mismatch of the cell lengths (after the best scale)
    angle_error: float = 0.0            # degrees
    scale_ratio: float = 1.0            # fitted / model lattice constant
    rotation: float = 0.0               # screen angle of the model's a vector
    mirrored: bool = False

    def describe(self) -> str:
        return (f"{self.model.name}: a along {self.rotation:.1f}°{' (mirrored)' if self.mirrored else ''}, "
                f"cell {'×'.join(str(int(round(x))) for x in (abs(np.linalg.det(self.T)),))} fitted cell(s), "
                f"fitted/model scale {self.scale_ratio:.4f}, angle mismatch {self.angle_error:.2f}°")

    def model_to_fit(self, uv_model: np.ndarray) -> np.ndarray:
        """Model fractional coordinates -> fitted-lattice fractional coordinates."""
        uv = np.asarray(uv_model, float).reshape(-1, 2) + self.offset
        return uv @ self.T.T

    def site_fractional(self) -> list[tuple[SiteDef, np.ndarray]]:
        """Every model site in fitted-lattice coordinates, reduced to the fitted cell [0, 1)²; a model
        cell larger than the fitted one contributes all its distinct positions."""
        out = []
        det = int(round(abs(np.linalg.det(self.T))))
        for s in self.model.sites:
            p = self.model_to_fit(np.array([[s.u, s.v]]))[0] % 1.0
            out.append((s, p))
        if det > 1:
            # a supercell: its sites do not all repeat with the fitted cell; keep them as they are
            pass
        return out

    def atom_pixels(self, shape: tuple[int, int], layer: Optional[int] = None) -> list[tuple[Atom, np.ndarray]]:
        """Pixel positions of every model atom (optionally one layer) inside the image."""
        rows, cols = shape
        corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], float)
        f = self.fit.to_fractional(corners)
        Tinv = np.linalg.inv(self.T)
        mc = (f @ Tinv.T) - self.offset
        lo, hi = np.floor(mc.min(0)) - 1, np.ceil(mc.max(0)) + 1
        if np.prod(hi - lo) > 2e6:
            raise ValueError("too many model cells in the image")
        out = []
        for atom in self.model.atoms:
            if layer is not None and atom.layer != layer:
                continue
            m, n = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            uv = np.column_stack([m.ravel() + atom.u, n.ravel() + atom.v])
            px = self.fit.to_pixel(self.model_to_fit(uv))
            ok = (px[:, 0] >= 0) & (px[:, 0] <= cols) & (px[:, 1] >= 0) & (px[:, 1] <= rows)
            out.append((atom, px[ok]))
        return out

    def direction(self, h: float, k: float) -> np.ndarray:
        """Physical vector of the model direction [h k] (in model cell units) in the image."""
        v = self.T @ np.array([h, k], float)
        return self.fit.direction(v[0], v[1])

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model.to_dict(), "T": self.T.tolist(), "offset": self.offset.tolist(),
                "mirrored": self.mirrored}


def align(model: SurfaceModel, fit: LatticeFit, max_entry: int = 3, max_det: int = 4,
          tol_length: float = 0.08, tol_angle: float = 4.0) -> list[Alignment]:
    """
    Every way the model cell fits onto the fitted lattice: integer matrices T (entries up to
    ``max_entry``) with the model cell = fitted cell × T, ranked by shape mismatch (lengths after a
    common scale, angle).  Ties are symmetry-equivalent alignments the image alone cannot tell
    apart.
    """
    A = np.column_stack([fit.a1, fit.a2])
    ratio_target = model.b / model.a
    out = []
    rng = range(-max_entry, max_entry + 1)
    for t in itertools.product(rng, repeat=4):
        T = np.array([[t[0], t[1]], [t[2], t[3]]], float)
        det = np.linalg.det(T)
        if abs(det) < 0.5 or abs(det) > max_det + 0.5:
            continue
        mirrored = det < 0
        if mirrored and not model.mirror_allowed:
            continue
        va, vb = A @ T[:, 0], A @ T[:, 1]
        la, lb = np.linalg.norm(va), np.linalg.norm(vb)
        g = gamma_deg(va, vb)
        # the model's cell is counter-clockwise; a clockwise pair is its mirror image
        ccw = (va[0] * vb[1] - va[1] * vb[0]) < 0
        if not ccw and not mirrored:
            continue
        if ccw and mirrored:
            continue
        lerr = abs((lb / la) / ratio_target - 1)
        aerr = abs(g - model.gamma)
        if lerr > tol_length or aerr > tol_angle:
            continue
        scale = la / model.a
        out.append(Alignment(model, fit, T, np.zeros(2), lerr, aerr, scale, (math.degrees(math.atan2(-va[1], va[0]))) % 360,
                             bool(mirrored)))
    if not out:
        return []
    # the smallest cell that matches; then by mismatch, then a closest to the image x axis
    best_det = min(abs(np.linalg.det(a.T)) for a in out)
    keep = [a for a in out if abs(abs(np.linalg.det(a.T)) - best_det) < 0.5]
    keep.sort(key=lambda a: (round(a.length_error, 2), round(a.angle_error * 2) / 2, a.mirrored,
                             min(a.rotation, 360 - a.rotation)))
    # alignments that put the same atoms on the same places are one picture: keep one of each
    uniq: list[Alignment] = []
    sigs: list[np.ndarray] = []
    for a in keep:
        sig = _signature(a)
        if any(_same(sig, o) for o in sigs):
            continue
        sigs.append(sig)
        uniq.append(a)
    return uniq


def alignment_from(model: SurfaceModel, fit: LatticeFit, T: Any, offset: Any = (0.0, 0.0),
                   mirrored: bool = False) -> Alignment:
    """An Alignment for a given T (e.g. one stored in a project), with its mismatch, scale and rotation."""
    T = np.asarray(T, float)
    A = np.column_stack([fit.a1, fit.a2])
    va, vb = A @ T[:, 0], A @ T[:, 1]
    la, lb = np.linalg.norm(va), np.linalg.norm(vb)
    lerr = abs((lb / la) / (model.b / model.a) - 1) if la > 0 and model.a > 0 else 0.0
    aerr = abs(gamma_deg(va, vb) - model.gamma)
    return Alignment(model, fit, T, np.asarray(offset, float), float(lerr), float(aerr), float(la / model.a),
                     math.degrees(math.atan2(-va[1], va[0])) % 360, bool(mirrored))


def _cell_kind(a: float, b: float, gamma: float) -> str:
    g = min(gamma, 180.0 - gamma)
    equal = abs(a / b - 1) < 0.05 if b > 0 else False
    if equal and abs(g - 60.0) < 4:
        return "hexagonal"
    if abs(g - 90.0) < 4:
        return "square" if equal else "rectangular"
    return "oblique"


def explain_no_alignment(model: SurfaceModel, fit: LatticeFit, max_entry: int = 3, max_det: int = 4,
                         tol_length: float = 0.08, tol_angle: float = 4.0) -> str:
    """Why ``align`` found nothing: the closest the model cell comes to the fitted lattice, against
    the tolerances, with what usually causes the mismatch."""
    A = np.column_stack([fit.a1, fit.a2])
    ratio_target = model.b / model.a
    best = None
    for t in itertools.product(range(-max_entry, max_entry + 1), repeat=4):
        T = np.array([[t[0], t[1]], [t[2], t[3]]], float)
        det = np.linalg.det(T)
        if abs(det) < 0.5 or abs(det) > max_det + 0.5:
            continue
        va, vb = A @ T[:, 0], A @ T[:, 1]
        la, lb = np.linalg.norm(va), np.linalg.norm(vb)
        lerr = abs((lb / la) / ratio_target - 1)
        aerr = min(abs(gamma_deg(va, vb) - model.gamma), abs(180.0 - gamma_deg(va, vb) - model.gamma))
        score = lerr / tol_length + aerr / tol_angle
        if best is None or score < best[0]:
            best = (score, lerr, aerr, abs(det))
    la1, la2 = float(np.linalg.norm(fit.a1)), float(np.linalg.norm(fit.a2))
    g = gamma_deg(fit.a1, fit.a2)
    from ..model.units import format_value
    lines = [f"The model cell (a = {format_value(model.a, 'm')}, b = {format_value(model.b, 'm')}, "
             f"γ = {model.gamma:.1f}°) does not match the fitted lattice (a1 = {format_value(la1, fit.unit)}, "
             f"a2 = {format_value(la2, fit.unit)}, γ = {g:.1f}°) in shape."]
    if best is not None:
        _, lerr, aerr, det = best
        lines.append(f"Closest match ({det:.0f} fitted cell(s) per model cell): side ratio off by {100 * lerr:.1f}% "
                     f"(allowed {100 * tol_length:.0f}%), angle off by {aerr:.1f}° (allowed {tol_angle:.0f}°).")
        sym_fit, sym_model = _cell_kind(la1, la2, g), _cell_kind(model.a, model.b, model.gamma)
        if sym_fit != sym_model and "oblique" not in (sym_fit, sym_model):
            lines.append(f"The fitted lattice is {sym_fit} but the model cell is {sym_model}: this is a different "
                         "surface (for example fcc(100) instead of fcc(111)). Choose a model with the image's symmetry.")
        elif lerr > tol_length and aerr <= tol_angle:
            lines.append("The angle fits but the sides do not: the image may be stretched along one direction "
                         "(drift, creep, or a different x and y pixel size), or the model's a/b ratio is wrong.")
        elif aerr > tol_angle and lerr <= tol_length:
            lines.append("The sides fit but the angle does not: thermal drift or creep shears the image, or the "
                         "fit picked the wrong pair of vectors (check the cell drawn on the image).")
    lines.append("Also check that the fit found the substrate lattice and not a molecular superstructure (compare "
                 "a1 and a2 with the model), and that this model is the right surface.")
    return " ".join(lines)


def alignment_warnings(al: Alignment, calibrated: bool) -> list[str]:
    """Things worth knowing about an alignment the user is about to use."""
    out = []
    if calibrated and abs(al.scale_ratio - 1) > 0.05:
        out.append(f"The fitted lattice constant differs from the model's by {100 * (al.scale_ratio - 1):+.0f}%: "
                   "check the calibration or the model's lattice constant (the alignment uses the shape only).")
    elif not calibrated:
        out.append("The image is not calibrated: the alignment uses the shape of the cell only, and distances stay "
                   "in pixels.")
    if al.length_error > 0.04 or al.angle_error > 2.0:
        out.append(f"The match is loose (side ratio {100 * al.length_error:.1f}%, angle {al.angle_error:.1f}°): "
                   "the image is probably distorted, so positions within the cell are less certain.")
    return out


def _signature(al: Alignment) -> list:
    out = []
    for atom in al.model.atoms:
        uv = al.model_to_fit(np.array([[atom.u, atom.v]]))[0] % 1.0
        out.append((atom.species, atom.layer, uv))
    return out


def _same(a: list, b: list, tol: float = 0.02) -> bool:
    used = set()
    for sp, ly, uv in a:
        hit = False
        for k, (sp2, ly2, uv2) in enumerate(b):
            if k in used or sp != sp2 or ly != ly2:
                continue
            d = (uv - uv2 + 0.5) % 1.0 - 0.5
            if np.all(np.abs(d) < tol):
                used.add(k)
                hit = True
                break
        if not hit:
            return False
    return True
