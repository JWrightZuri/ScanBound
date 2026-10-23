"""
Ranking candidate explanations of an object, with the evidence behind each - a way to organise
what is known and what is missing, not an identification.

    ev = Evidence(objects, shapes=shape_descriptors(objects, image), adsorption=res,
                  defects=defect_xy, quality=report, channels=channel_rows, templates=ncc_rows)
    for h in rank(target_id, ev):
        print(h.name, h.score, [(t.name, t.value, t.why) for t in h.terms], h.missing)

How it scores
-------------
Each hypothesis multiplies evidence terms between 0 and 1 (geometric mean, so one strong
contradiction pulls it down without one missing term deciding it).  Terms compare the object with
the *reviewed* population of each class (manual and accepted annotations, the object itself left
out):

``shape``      how typical its size, apparent height, elongation and lobe count are for the class
               (robust z-scores, combined)
``template``   correlation with the class's mean image in the molecule's own frame
``site``       whether it sits on the class's most common adsorption site
``defect``     how close the nearest defect is (1 = on it, 0.5 at one lattice constant / 0.5 nm)
``size``       how much smaller than the class it is (fragments, products)
``channels``   how typical its values in the registered channels are for the class
``quality``    whether its scan lines are flagged (tip / feedback artefacts)

The score ranks hypotheses for this object under these terms; it is not a probability.  Missing
evidence (no lattice fit, no second channel, no templates) is listed with every hypothesis, which is
often the most useful output: it says what to measure next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..annotations import GROUND_TRUTH
from .objects import SurfaceObject

SHAPE_KEYS = ("radius", "apparent_height", "elongation", "n_lobes")


@dataclass
class Term:
    name: str
    value: float
    why: str


@dataclass
class Hypothesis:
    name: str
    score: float
    terms: list[Term]
    missing: list[str] = field(default_factory=list)

    def describe(self) -> str:
        t = "; ".join(f"{x.name} {x.value:.2f} ({x.why})" for x in self.terms)
        m = f" — missing: {', '.join(self.missing)}" if self.missing else ""
        return f"{self.name}: {self.score:.2f} [{t}]{m}"


@dataclass
class Evidence:
    objects: Sequence[SurfaceObject]
    shapes: Sequence[dict[str, Any]] = ()
    adsorption: Any = None                     # AdsorptionResult
    defects: Optional[np.ndarray] = None       # pixel positions
    quality: Any = None                        # QualityReport
    channels: Sequence[dict[str, Any]] = ()    # channel_features rows
    templates: Sequence[dict[str, Any]] = ()   # template_scores rows
    scale: tuple[float, float] = (1.0, 1.0)
    lattice_constant: Optional[float] = None
    min_class_size: int = 3

    def by_id(self, rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(r.get("id")): r for r in rows}


def _robust(vals: list[float]) -> tuple[float, float]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return math.nan, math.nan
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    if mad <= 0:
        mad = float(v.std()) or abs(med) * 0.1 or 1.0
    return med, mad


def _typicality(z: list[float]) -> float:
    z = [x for x in z if np.isfinite(x)]
    if not z:
        return math.nan
    return float(math.exp(-0.5 * float(np.mean(np.square(np.clip(z, -6, 6))))))


def rank(target: str, ev: Evidence) -> list[Hypothesis]:
    objs = {o.id: o for o in ev.objects}
    if target not in objs:
        raise KeyError(f"no object {target!r}")
    o = objs[target]
    shapes = ev.by_id(ev.shapes)
    chans = ev.by_id(ev.channels)
    temps = ev.by_id(ev.templates)
    ads = ev.by_id(ev.adsorption.rows) if ev.adsorption is not None else {}
    sc = np.asarray(ev.scale, float)
    classes: dict[str, list[SurfaceObject]] = {}
    for x in ev.objects:
        if x.id != target and x.label and x.status in GROUND_TRUTH:
            classes.setdefault(x.label, []).append(x)
    classes = {k: v for k, v in classes.items() if len(v) >= ev.min_class_size}
    missing = []
    if not ev.shapes:
        missing.append("shape descriptors")
    if ev.adsorption is None:
        missing.append("lattice registration (adsorption sites)")
    if not ev.templates:
        missing.append("class templates")
    if not ev.channels:
        missing.append("a registered second channel (other bias / ncAFM)")
    if ev.quality is None:
        missing.append("a quality report")
    if not classes:
        missing.append(f"reviewed classes with at least {ev.min_class_size} members")

    # --- generic terms
    q_term = None
    if ev.quality is not None:
        bad = set()
        for r in ev.quality.bad_rows:
            bad.update(range(r - 1, r + 2))
        tipc = [f.rows[0] for f in ev.quality.issues if f.code == "tip_change" and f.rows]
        row = int(math.floor(o.y))
        if row in bad:
            q_term = Term("quality", 0.9, "on a flagged scan line")
        elif any(abs(row - t) <= 3 for t in tipc):
            q_term = Term("quality", 0.7, "at the line where the tip changed")
        else:
            q_term = Term("quality", 0.1, "no artefact flagged at its position")
    d_term = None
    if ev.defects is not None and len(ev.defects):
        d = float(np.min(np.hypot(*((np.asarray(ev.defects, float).reshape(-1, 2) - o.xy) * sc).T)))
        r0 = ev.lattice_constant or (0.5e-9 if sc[0] != 1.0 else 10.0)
        val = float(math.exp(-d / r0 * math.log(2)))
        d_term = Term("defect", val, f"nearest defect at {d:.3g} (units of the image)")

    def shape_z(cls: list[SurfaceObject], keys: Sequence[str] = SHAPE_KEYS) -> tuple[float, str]:
        row = shapes.get(o.id)
        if not row:
            return math.nan, "no descriptors"
        zs, worst = [], ("", 0.0)
        for k in keys:
            med, mad = _robust([shapes.get(x.id, {}).get(k) for x in cls])
            v = row.get(k)
            if v is None or not np.isfinite(med) or not np.isfinite(v):
                continue
            z = (v - med) / mad
            zs.append(z)
            if abs(z) > abs(worst[1]):
                worst = (k, z)
        return _typicality(zs), (f"largest deviation: {worst[0]} {worst[1]:+.1f} σ" if worst[0] else "typical")

    def channel_term(label: str, cls: list[SurfaceObject]) -> Optional[Term]:
        row = chans.get(o.id)
        if not row:
            return None
        keys = [k for k in row if ":" in k and k.endswith((":contrast", ":mean")) and np.isfinite(row.get(k, np.nan))]
        zs, worst = [], ("", 0.0)
        for k in keys:
            med, mad = _robust([chans.get(x.id, {}).get(k) for x in cls])
            if not np.isfinite(med):
                continue
            z = (row[k] - med) / mad
            zs.append(z)
            if abs(z) > abs(worst[1]):
                worst = (k, z)
        t = _typicality(zs)
        if not np.isfinite(t):
            return None
        return Term("channels", t, f"largest deviation: {worst[0]} {worst[1]:+.1f} σ" if worst[0] else "typical")

    hyps: list[Hypothesis] = []
    best_shape = 0.0
    for label, cls in classes.items():
        terms_common: list[Term] = []
        sh, why = shape_z(cls)
        if np.isfinite(sh):
            terms_common.append(Term("shape", sh, why))
            best_shape = max(best_shape, sh)
        tp = temps.get(o.id, {}).get(f"ncc:{label}")
        t_term = Term("template", (tp + 1) / 2, f"correlation {tp:.2f} with the {label} template") \
            if tp is not None and np.isfinite(tp) else None
        c_term = channel_term(label, cls)
        site_term = None
        if ads and o.id in ads:
            sites = [ads[x.id]["site"] for x in cls if x.id in ads]
            if sites:
                common = max(set(sites), key=sites.count)
                frac = sites.count(common) / len(sites)
                p = float(ads[o.id]["site_probabilities"].get(common, 0.0))
                site_term = Term("site", p, f"{label} usually sits on '{common}' ({100 * frac:.0f} %); this one: "
                                            f"'{ads[o.id]['site']}'")
        size_ratio = None
        row = shapes.get(o.id)
        if row and row.get("radius"):
            med, _ = _robust([shapes.get(x.id, {}).get("radius") for x in cls])
            if np.isfinite(med) and med > 0:
                size_ratio = row["radius"] / med

        def make(name: str, terms: list[Optional[Term]]) -> None:
            ts = [t for t in terms if t is not None and np.isfinite(t.value)]
            if not ts:
                return
            score = float(np.exp(np.mean([math.log(max(t.value, 1e-6)) for t in ts])))
            hyps.append(Hypothesis(name, score, ts, list(missing)))
        q_ok = Term("quality", 1 - q_term.value, q_term.why) if q_term is not None else None
        make(f"{label} (typical)", [*terms_common, t_term, site_term, c_term, q_ok])
        if site_term is not None:
            make(f"{label} on another adsorption site",
                 [*terms_common, t_term, Term("site", 1 - site_term.value, site_term.why), q_ok])
        if t_term is not None:
            size_ok = Term("size", math.exp(-0.5 * ((size_ratio or 1) - 1) ** 2 / 0.15 ** 2),
                           f"size {size_ratio:.2f} × the class" if size_ratio else "size unknown")
            make(f"{label}, another conformer or orientation",
                 [size_ok, Term("template", 1 - t_term.value, t_term.why), q_ok])
        if d_term is not None:
            make(f"{label} interacting with a defect", [*terms_common, d_term, q_ok])
        if size_ratio is not None:
            frag = float(np.clip((1 - size_ratio) / 0.5, 0, 1))
            make(f"fragment or product of {label}", [Term("size", frag, f"size {size_ratio:.2f} × {label}"), q_ok])
    if q_term is not None:
        terms = [q_term]
        if classes and best_shape:
            terms.append(Term("shape", 1 - best_shape, "unlike every class" if best_shape < 0.3 else
                              "resembles a known class"))
        hyps.append(Hypothesis("tip artefact or unresolved feature",
                               float(np.exp(np.mean([math.log(max(t.value, 1e-6)) for t in terms]))), terms, list(missing)))
    if classes and shapes.get(o.id):
        hyps.append(Hypothesis("an unknown species", 1 - best_shape,
                               [Term("shape", 1 - best_shape, "how unlike the best-matching class it is")], list(missing)))
    hyps.sort(key=lambda h: -h.score)
    return hyps
