"""
Where surface-analysis state lives in a project, and how annotations become analysis inputs.

    st = SurfaceStore(project)
    st.set_fit(image_id, fit)              # saved in the project file (lattice, sites, indices)
    st.fit(image_id)                       # back, after reopening the project
    st.set_model(model); st.set_alignment(image_id, model.name, T, offset)
    st.objects(image_id, image)            # molecules (feature annotations, not defects), ground truth by default
    st.context(image_id, image)            # step edges, defects (as SurfaceContext)
    st.settings["chirality"] = {"mol-L": "L", "mol-R": "R"}

The project file keeps this under its ``"surface"`` key (images -> lattice fits and model alignment;
models; registrations between images; settings).  Nothing here is needed to open the project in an
older ScanBound: the key is simply ignored there.  Large derived results (distortion maps, domains,
statistics) are recomputed on demand rather than stored; their inputs and parameters are.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..annotations import GROUND_TRUTH
from .adsorption import SurfaceContext
from .lattice import LatticeFit
from .objects import SurfaceObject, objects_from_annotations
from .registration import Registration
from .structure import Alignment, SurfaceModel, alignment_from

DEFAULT_SETTINGS: dict[str, Any] = {
    "defect_labels": ["vacancy", "adatom", "extra site", "unusual site"],
    "exclude_labels": [],                 # feature labels never treated as molecules
    "molecule_labels": None,              # None = every feature label that is not a defect label
    "chirality": {},                      # label -> "L" / "R"
    "mol_fold": 1,
    "include_proposed": False,            # unreviewed proposals enter the statistics
    "site_tolerance": 0.12,
    "direction": [1, 0],
    "mirror": False,
}


def remap_image_ids(surface: dict[str, Any], id_map: dict[str, str]) -> None:
    imgs = surface.get("images")
    if isinstance(imgs, dict):
        surface["images"] = {id_map.get(k, k): v for k, v in imgs.items()}
    b = surface.get("builder")
    if isinstance(b, dict) and isinstance(b.get("images"), dict):
        b["images"] = {id_map.get(k, k): v for k, v in b["images"].items()}
    regs = surface.get("registrations")
    if isinstance(regs, dict):
        new = {}
        for k, v in regs.items():
            a, _, b = k.partition("||")
            new[f"{id_map.get(a, a)}||{id_map.get(b, b)}"] = v
        surface["registrations"] = new


class SurfaceStore:
    def __init__(self, project: Any):
        self.project = project
        s = project.surface
        s.setdefault("images", {})
        s.setdefault("models", {})
        s.setdefault("registrations", {})
        st = s.setdefault("settings", {})
        for k, v in DEFAULT_SETTINGS.items():
            st.setdefault(k, v if not isinstance(v, (list, dict)) else type(v)(v))

    @property
    def data(self) -> dict[str, Any]:
        return self.project.surface

    @property
    def settings(self) -> dict[str, Any]:
        return self.data["settings"]

    def changed(self) -> None:
        self.project.dirty = True

    def _img(self, image_id: str, create: bool = True) -> dict[str, Any]:
        imgs = self.data["images"]
        if create:
            return imgs.setdefault(image_id, {})
        return imgs.get(image_id, {})

    # ---- lattices ------------------------------------------------------------------
    def fits(self, image_id: str) -> list[LatticeFit]:
        out = []
        for d in self._img(image_id, False).get("lattices", []):
            try:
                out.append(LatticeFit.from_dict(d))
            except Exception:  # noqa: BLE001 - a damaged entry is skipped, not fatal
                continue
        return out

    def fit(self, image_id: str, name: Optional[str] = None) -> Optional[LatticeFit]:
        fs = self.fits(image_id)
        if name is not None:
            fs = [f for f in fs if f.name == name]
        return fs[0] if fs else None

    def set_fit(self, image_id: str, fit: LatticeFit) -> None:
        img = self._img(image_id)
        lst = [d for d in img.get("lattices", []) if d.get("name") != fit.name]
        lst.insert(0 if fit.name == "lattice" else len(lst), fit.to_dict())
        img["lattices"] = lst
        self.changed()

    def remove_fit(self, image_id: str, name: Optional[str] = None) -> None:
        img = self._img(image_id)
        img["lattices"] = [d for d in img.get("lattices", []) if name is not None and d.get("name") != name]
        if name is None:
            img.pop("alignment", None)
        self.changed()

    # ---- models and alignment --------------------------------------------------------
    def models(self) -> list[SurfaceModel]:
        return [SurfaceModel.from_dict(d) for d in self.data["models"].values()]

    def model(self, name: str) -> Optional[SurfaceModel]:
        d = self.data["models"].get(name)
        return SurfaceModel.from_dict(d) if d else None

    def set_model(self, model: SurfaceModel) -> None:
        self.data["models"][model.name] = model.to_dict()
        self.changed()

    def set_alignment(self, image_id: str, model: str, T: np.ndarray, offset: Sequence[float] = (0.0, 0.0),
                      mirrored: bool = False) -> None:
        self._img(image_id)["alignment"] = {"model": model, "T": np.asarray(T, float).tolist(),
                                            "offset": [float(x) for x in offset], "mirrored": bool(mirrored)}
        self.changed()

    def clear_alignment(self, image_id: str) -> None:
        self._img(image_id).pop("alignment", None)
        self.changed()

    def alignment(self, image_id: str, fit: Optional[LatticeFit] = None) -> Optional[Alignment]:
        d = self._img(image_id, False).get("alignment")
        fit = fit or self.fit(image_id)
        if not d or fit is None:
            return None
        model = self.model(d["model"])
        if model is None:
            return None
        return alignment_from(model, fit, d["T"], d.get("offset", [0, 0]), bool(d.get("mirrored", False)))

    # ---- registrations ---------------------------------------------------------------
    def registration(self, a: str, b: str) -> Optional[Registration]:
        regs = self.data["registrations"]
        if a == b:
            return Registration.identity()
        if f"{a}||{b}" in regs:
            return Registration.from_dict(regs[f"{a}||{b}"])
        if f"{b}||{a}" in regs:
            return Registration.from_dict(regs[f"{b}||{a}"]).inverse()
        ra, _ = self.project.split_image_id(a)
        rb, _ = self.project.split_image_id(b)
        if ra == rb:                                    # channels of one file share the pixel grid
            return Registration.identity("channels of the same scan")
        return None

    def set_registration(self, a: str, b: str, reg: Registration) -> None:
        self.data["registrations"].pop(f"{b}||{a}", None)
        self.data["registrations"][f"{a}||{b}"] = reg.to_dict()
        self.changed()

    # ---- inputs from annotations ------------------------------------------------------
    @property
    def statuses(self) -> tuple[str, ...]:
        return tuple(GROUND_TRUTH) + (("proposed",) if self.settings.get("include_proposed") else ())

    def defect_labels(self) -> set[str]:
        return set(self.settings.get("defect_labels") or [])

    def molecule_labels(self, image_ids: Optional[Iterable[str]] = None) -> list[str]:
        chosen = self.settings.get("molecule_labels")
        if chosen:
            return list(chosen)
        excl = self.defect_labels() | set(self.settings.get("exclude_labels") or [])
        ids = set(image_ids) if image_ids is not None else None
        labels = {a.label for a in self.project.annotations if a.kind == "feature" and a.label
                  and (ids is None or a.image_id in ids) and a.attrs.get("surface") != "defect"}
        return sorted(labels - excl)

    def objects(self, image_id: str, image: Any = None, labels: Optional[Iterable[str]] = None) -> list[SurfaceObject]:
        anns = [a for a in self.project.annotations.for_image(image_id) if a.attrs.get("surface") != "defect"]
        labs = list(labels) if labels is not None else self.molecule_labels([image_id])
        objs = objects_from_annotations(anns, image, labels=labs, statuses=self.statuses,
                                        chirality=self.settings.get("chirality") or {})
        mask = self.exclude_mask(image_id, None if image is None else np.asarray(getattr(image, "data", image)).shape)
        if mask is not None:
            rows, cols = mask.shape
            objs = [o for o in objs if not mask[int(np.clip(o.y, 0, rows - 1)), int(np.clip(o.x, 0, cols - 1))]]
        return objs

    def defects(self, image_id: str) -> tuple[np.ndarray, list[str]]:
        labs = self.defect_labels()
        sel = [a for a in self.project.annotations.for_image(image_id)
               if a.kind == "feature" and a.status in self.statuses
               and (a.label in labs or a.attrs.get("surface") == "defect")]
        xy = np.array([a.geometry.centroid() for a in sel], float).reshape(-1, 2)
        return xy, [str(a.label) for a in sel]

    def steps(self, image_id: str) -> list[np.ndarray]:
        return [np.asarray(a.geometry.points, float) for a in self.project.annotations.for_image(image_id, kinds=("step_edge",))
                if a.status in self.statuses]

    def regions(self, image_id: str, label: Optional[str] = None) -> list[Any]:
        return [a for a in self.project.annotations.for_image(image_id, kinds=("region",))
                if a.status != "rejected" and (label is None or a.label == label)]

    def exclude_mask(self, image_id: str, shape: Optional[tuple[int, int]]) -> Optional[np.ndarray]:
        regs = self.regions(image_id, "exclude")
        if not regs or shape is None:
            return None
        m = np.zeros(shape, bool)
        for a in regs:
            m |= a.geometry.mask(shape)
        return m

    def context(self, image_id: str, image: Any = None, domain_map: Optional[np.ndarray] = None,
                terrace_map: Optional[np.ndarray] = None) -> SurfaceContext:
        xy, labels = self.defects(image_id)
        return SurfaceContext(self.steps(image_id), xy, labels, domain_map, terrace_map)
