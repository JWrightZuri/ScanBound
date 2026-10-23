"""
The integrated surface workflows: each takes a project (through ``SurfaceStore``) and image ids,
runs the analysis modules with the project's settings and returns a ``Report``.

    st = SurfaceStore(project)
    rep = structure_explorer(st, image_id)          # lattice, symmetry, periodicities, strain, domains
    rep = adsorption_map(st, image_id)              # sites, orientations, chirality, defect/step proximity
    rep = assembly_analysis(st, image_id)           # neighbours, g(r), order, chains, randomized references
    rep = defect_map(st, image_id)                  # vacancies, adatoms, complexes, environments
    rep = correlation_workbench(st, [stm_id, afm_id])
    rep = evolution_tracker(st, [id_t0, id_t1, ...])
    rep = quality_check(st, image_id)
    rep = explain_object(st, image_id, annotation_id)
    rep.save("folder")                             # one CSV per table, report.json, summary.txt

A ``Report`` is plain data: summary lines, tables (lists of dicts), plot specifications (data plus
kind) and provenance (images, file hashes, parameters, ScanBound version, time).  The GUI shows it;
scripts can use it directly.  Interpretive results (sites, hypotheses) are candidates with the
evidence attached; nothing here edits annotations - proposals are returned for the user to review.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .lattice import LatticeFit, bravais_candidates, fit_image, image_scale, symmetry_fold
from .objects import SurfaceObject, distance_to_polylines
from .store import SurfaceStore


class SurfaceError(RuntimeError):
    """An analysis cannot run (e.g. no lattice fit yet); the message says what to do."""


@dataclass
class Report:
    title: str
    summary: list[str] = field(default_factory=list)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    plots: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    proposals: list[Any] = field(default_factory=list)      # annotations for the user to review
    extra: dict[str, Any] = field(default_factory=dict)     # live objects (fits, maps) for overlays

    def add(self, line: str) -> None:
        self.summary.append(line)

    def text(self) -> str:
        return "\n".join([self.title, ""] + self.summary)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "summary": self.summary, "tables": _jsonable(self.tables),
                "provenance": _jsonable(self.provenance)}

    def save(self, folder: str) -> list[str]:
        os.makedirs(folder, exist_ok=True)
        out = []
        for name, rows in self.tables.items():
            p = os.path.join(folder, f"{_slug(name)}.csv")
            write_csv(p, rows, header=[f"# {self.title}: {name}", f"# {self.provenance.get('created', '')}"])
            out.append(p)
        p = os.path.join(folder, "report.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        out.append(p)
        p = os.path.join(folder, "summary.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(self.text() + "\n\nProvenance\n" + json.dumps(_jsonable(self.provenance), indent=1))
        out.append(p)
        return out


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_") or "table"


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def write_csv(path: str, rows: Sequence[dict[str, Any]], header: Sequence[str] = ()) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for h in header:
            fh.write(h + "\n")
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([_cell(r.get(k)) for k in keys])


def _cell(v: Any) -> Any:
    if isinstance(v, dict):
        return "; ".join(f"{k} {_cell(x)}" for k, x in v.items())
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(float(v)) else f"{float(v):.6g}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(str(_cell(x)) for x in v)
    return "" if v is None else v


# ----------------------------------------------------------------------------------------------- helpers
def _version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return "?"


def provenance(st: SurfaceStore, image_ids: Sequence[str], params: dict[str, Any], method: str) -> dict[str, Any]:
    pr = st.project
    imgs = []
    for i in image_ids:
        rel, ch = pr.split_image_id(i)
        e = pr.entry(i, create=False)
        h = ""
        try:
            h = pr.hash_file(rel)
        except Exception:  # noqa: BLE001
            pass
        imgs.append({"image_id": i, "file": rel, "channel": ch, "sha1": h,
                     "processing": [s.get("op") if isinstance(s, dict) else str(s)
                                    for s in (getattr(getattr(e, "chain", None), "to_dict", lambda: [])() or [])]
                     if e is not None else []})
    return {"method": method, "created": _dt.datetime.now().isoformat(timespec="seconds"), "scanbound": _version(),
            "images": imgs, "parameters": _jsonable(params),
            "statuses_used": list(st.statuses), "settings": _jsonable(st.settings)}


def image_of(st: SurfaceStore, image_id: str) -> Any:
    return st.project.processed_image(image_id)


def require_fit(st: SurfaceStore, image_id: str) -> LatticeFit:
    fit = st.fit(image_id)
    if fit is None:
        raise SurfaceError("This image has no lattice fit yet. Fit the lattice first (Surface panel → Fit lattice).")
    return fit


def run_lattice_fit(st: SurfaceStore, image_id: str, **kw: Any) -> LatticeFit:
    """Fit the lattice of an image (keyword arguments go to ``fit_image``), honouring 'exclude' regions,
    and store it in the project."""
    img = image_of(st, image_id)
    excluded = st.exclude_mask(image_id, img.shape)
    mask = kw.pop("mask", None)
    if excluded is not None:
        mask = ~excluded if mask is None else (np.asarray(mask, bool) & ~excluded)
    fit = fit_image(img, mask=mask, **kw)
    fit.image_id = image_id
    st.set_fit(image_id, fit)
    return fit


def lattice_fit_advice(error: str, image: Any = None) -> str:
    """A failed lattice fit explained in plain words, with what to try."""
    msg = error.strip().splitlines()[-1] if error.strip() else "unknown error"
    msg = msg.split(": ", 1)[1] if msg.split(": ", 1)[0].endswith(("Error", "Exception")) else msg
    tips = []
    low = msg.lower()
    if "site(s) to fit" in low or "fewer than 4" in low:
        tips += ["Lower the Threshold (tick Preview sites to see which sites are found).",
                 "Check Bright/Dark: on ncAFM Δf images or inverted contrast the sites are minima.",
                 "Enter the Spacing (Estimate, then adjust): a wrong spacing smooths the sites away."]
    if "lattice vectors" in low:
        tips += ["Draw two Lattice vector annotations along a1 and a2 as a starting guess, then fit again.",
                 "Enter the Spacing: the Fourier estimate may have found a superstructure or noise."]
    if "too fine" in low:
        tips += ["Enter a larger Spacing: the estimate is far below the atomic spacing."]
    if "parallel" in low:
        tips += ["Redraw the two lattice vector annotations along different lattice directions."]
    if not tips:
        tips = ["Tick Preview sites and adjust Spacing and Threshold until each lattice maximum has one green dot.",
                "Exclude damaged or molecule-covered areas with a Region labelled 'exclude'."]
    if image is not None and not getattr(image, "calibrated", True):
        tips.append("The image has no physical pixel size: Spacing is in pixels.")
    return f"The lattice fit failed: {msg}\n\nWhat to try:\n• " + "\n• ".join(tips)


def fit_warnings(fit: LatticeFit, spacing: Optional[float] = None) -> list[str]:
    """Reasons to distrust a lattice fit that did succeed."""
    out = []
    n_all = len(fit.sites.xy)
    n = fit.n_sites
    a = float(min(np.linalg.norm(fit.a1), np.linalg.norm(fit.a2)))
    if n < 30:
        out.append(f"Only {n} sites were fitted: the lattice constants are rough.")
    if n_all and n / n_all < 0.6:
        out.append(f"Only {100 * n / n_all:.0f}% of the detected sites lie on the fitted lattice: the threshold may "
                   "be picking up noise or molecules, or the fit found the wrong lattice.")
    if a > 0 and fit.rms > 0.15 * a:
        out.append(f"The sites scatter by {100 * fit.rms / a:.0f}% of the lattice constant around the fit: the "
                   "image is distorted or the sites are poorly detected.")
    if spacing and a > 0 and abs(a / spacing - 1) > 0.25:
        from ..model.units import format_value
        out.append(f"The fitted spacing ({format_value(a, fit.unit)}) is far from the Spacing you entered "
                   f"({format_value(spacing, fit.unit)}): the fit may have found a superstructure or a sub-lattice.")
    return out


def preview_sites(st: SurfaceStore, image_id: str, *, spacing: Optional[float] = None, bright: bool = True,
                  floor: float = 0.0) -> tuple[Any, Optional[float]]:
    """Every candidate site down to a low threshold, honouring 'exclude' regions, for tuning the
    threshold interactively: keep ``sites.score >= threshold``.  Also returns the spacing used."""
    from .lattice import detect_sites, estimate_spacing, image_scale
    img = image_of(st, image_id)
    excluded = st.exclude_mask(image_id, img.shape)
    scale = image_scale(img)[0]
    if spacing is None:
        spacing = estimate_spacing(img, scale)
    sites = detect_sites(img, scale, spacing=spacing, bright=bright, threshold=floor,
                         mask=None if excluded is None else ~excluded)
    return sites, spacing


def _scale(fit: LatticeFit) -> tuple[float, float]:
    return (float(fit.scale[0]), float(fit.scale[1]))


def _amin(fit: LatticeFit) -> float:
    return float(min(np.hypot(*fit.a1), np.hypot(*fit.a2)))


def _fmt(v: float, unit: str) -> str:
    from ..model.units import format_value
    return format_value(v, unit)


# ----------------------------------------------------------------------------------- 1 structure explorer
def structure_explorer(st: SurfaceStore, image_id: str, *, distortion: bool = True, domains: bool = True,
                       periodicities: bool = True, radius: float = 1.6) -> Report:
    from .distortion import local_distortion
    from .domains import find_domains
    from .structure import align
    from .superstructure import analyse_periodicities

    fit = require_fit(st, image_id)
    img = image_of(st, image_id)
    unit = fit.unit
    rep = Report("Surface structure")
    rep.extra["fit"] = fit
    rep.add(fit.describe())
    ms = fit.measurements()
    rep.tables["lattice"] = [{"quantity": k, "value": m.value, "uncertainty": m.uncertainty, "unit": m.unit,
                              "method": m.method, "assumptions": "; ".join(m.assumptions)} for k, m in ms.items()]
    if fit.calibration_notes:
        rep.add("Calibration: " + "; ".join(fit.calibration_notes))
    cands = bravais_candidates(fit)
    rep.tables["symmetry candidates"] = [{"lattice": c.name, "consistent": c.consistent, "length_dev": c.length_dev,
                                          "angle_dev_deg": c.angle_dev, "chi2": c.chi2, "note": c.note} for c in cands]
    rep.add(f"Most consistent lattice type: {symmetry_fold(fit)}-fold "
            f"({', '.join(c.name for c in cands if getattr(c, 'consistent', False)) or 'oblique'})")
    disp = fit.displacement[fit.on_lattice]
    if len(disp):
        rep.plots.append({"kind": "hist", "title": "Site displacement from the ideal lattice", "x_label": f"|d| ({unit})",
                          "data": disp.tolist()})
    if periodicities:
        try:
            pr = analyse_periodicities(img, fit)
            rep.tables["periodicities"] = pr.rows()
            if pr.superstructure is not None:
                ss = pr.superstructure
                rep.add(f"Periodicity beyond the lattice ({ss.kind}): {ss.matrix_text()} {ss.wood}".rstrip()
                        + " — a candidate; check it against the overlay")
            rep.tables["superstructure candidates"] = [
                {"candidate": m.name, "explained_fraction": m.explained_fraction, "matched": m.matched,
                 "predicted": m.predicted, "mean_error": m.mean_error, "score": m.score} for m in pr.candidates]
            for m in [m for m in pr.candidates if m.matched > 0][:3]:
                rep.add(f"Candidate {m.name}: explains {100 * m.explained_fraction:.0f} % of the extra peaks "
                        f"(score {m.score:.2f})")
            if pr.superstructure is None and not pr.candidates:
                rep.add("No periodicity beyond the substrate lattice above the detection threshold.")
        except Exception as e:  # noqa: BLE001
            rep.add(f"Periodicity analysis failed: {e}")
    if distortion:
        dm = local_distortion(fit, radius=radius)
        rep.extra["distortion"] = dm
        s = dm.summary()
        rep.tables["distortion summary"] = [{"quantity": k, "value": v} for k, v in s.items()]
        rep.tables["distortion per site"] = dm.rows()
        rep.add(f"Local distortion at {s['reliable']} reliable sites: dilatation spread {100 * s['dilatation_std']:.2f} %, "
                f"rotation spread {s['rotation_std_deg']:.2f}°, displacement rms {_fmt(s['displacement_rms'], unit)} "
                "(thermal drift and piezo creep also appear here; compare the map with the slow-scan direction).")
    if domains:
        dr = find_domains(fit.sites, img, scale=_scale(fit), shape=img.shape)
        rep.extra["domains"] = dr
        rep.tables["domains"] = dr.summary_rows()
        rep.add(f"{len(dr.domains)} domain(s) found (fold {dr.fold})" +
                (f", {len(dr.boundaries)} boundary segment(s)" if dr.boundaries else ""))
    al = st.alignment(image_id, fit)
    if al is not None:
        rep.add("Structure model: " + al.describe())
    else:
        models = st.models()
        for m in models[:1]:
            opts = align(m, fit)
            rep.tables["model alignments"] = [o.to_dict() for o in opts]
            rep.add(f"{len(opts)} way(s) to place model '{m.name}' on this lattice; choose one in the Surface panel.")
    rep.provenance = provenance(st, [image_id], {"radius": radius, "fit": fit.params_used}, "structure_explorer")
    return rep


# -------------------------------------------------------------------------------- 2 adsorption registry
def adsorption_map(st: SurfaceStore, image_id: str, *, labels: Optional[Sequence[str]] = None,
                   with_tests: bool = True, n_sim: int = 499) -> Report:
    from .adsorption import adsorption_registry, default_alignment
    from .statistics import chirality_stats, orientation_families, proximity_test, symmetry_period
    from .structure import Alignment

    fit = require_fit(st, image_id)
    img = image_of(st, image_id)
    objs = st.objects(image_id, img, labels)
    if not objs:
        raise SurfaceError("No molecules (feature annotations) on this image. Mark molecules first.")
    al = st.alignment(image_id, fit) or default_alignment(fit)
    ctx = st.context(image_id, img)
    sett = st.settings
    res = adsorption_registry(objs, fit, al, ctx, site_tolerance=sett["site_tolerance"],
                              direction=tuple(sett["direction"]), mol_fold=int(sett["mol_fold"]),
                              sub_fold=symmetry_fold(fit), mirror=bool(sett["mirror"]))
    rep = Report("Adsorption registry")
    rep.extra.update(adsorption=res, fit=fit, objects=objs)
    rep.tables["molecules"] = res.rows
    occ = res.occupancy()
    rep.tables["site occupancy"] = [{"site": k, "count": v, "weighted": occ["weighted"].get(k, 0.0)}
                                    for k, v in sorted(occ["counts"].items(), key=lambda t: -t[1])]
    rep.add(f"{occ['n']} molecules on model '{al.model.name}'"
            + (" (no model aligned: sites are relative to image maxima)" if al.model.name == "image maxima" else ""))
    rep.add("Site counts: " + ", ".join(f"{k} {v}" for k, v in sorted(occ["counts"].items(), key=lambda t: -t[1]))
            + f"; {occ['uncertain']} uncertain (best site p < 0.6)")
    p = res.uniformity_p()
    if np.isfinite(p):
        rep.add(f"Positions in the cell vs random: p = {p:.3g}"
                + (" (positions are not random: registry exists)" if p < 0.01 else ""))
    for c in res.clusters[:4]:
        rep.add(f"Position cluster at (u, v) = ({c['u']:.2f}, {c['v']:.2f}): {c['count']} molecules"
                + (f", nearest site '{c['nearest_site']}'" if c.get("nearest_site") else ""))
    eu, ev, H = res.heatmap()
    rep.plots.append({"kind": "heatmap", "title": "Occupancy in the unit cell (1 = random)", "x_label": "u (a1)",
                      "y_label": "v (a2)", "data": H.T.tolist(), "extent": [0, 1, 0, 1],
                      "points": [[r["u_cell"], r["v_cell"]] for r in res.rows]})
    labels_present = sorted({r["label"] for r in res.rows if r["label"]})
    if len(labels_present) > 1:
        cmp = res.compare("label")
        if np.isfinite(cmp.p):
            rep.add("Sites by species: " + cmp.describe())
        occ_l = res.occupancy("label")
        rep.tables["occupancy by species"] = [{"label": g, **v["counts"]} for g, v in occ_l.items()]
    hands = [r["handedness"] or None for r in res.rows]
    if any(hands):
        cs = chirality_stats(hands)
        rep.tables["chirality"] = [{"quantity": k, "value": v} for k, v in cs.items()]
        if cs.get("n_known"):
            lo, hi = cs["fraction_L_ci95"]
            rep.add(f"Handedness: {cs['L']} L, {cs['R']} R (fraction L {cs['fraction_L']:.2f}, 95 % CI {lo:.2f}–{hi:.2f}; "
                    f"racemic p = {cs['racemic_p']:.3g}); {cs['unknown']} unassigned")
            occ_h = res.occupancy("handedness")
            cmp_h = res.compare("handedness") if len([k for k in occ_h if k]) > 1 else None
            if cmp_h is not None and np.isfinite(cmp_h.p):
                rep.add("Sites by handedness: " + cmp_h.describe())
    rel = [r["orientation_rel"] for r in res.rows if r.get("orientation_rel") is not None]
    if rel:
        period = symmetry_period(int(sett["mol_fold"]), symmetry_fold(fit))
        fam = orientation_families(rel, period=period)
        rep.tables["orientation families"] = fam.rows()
        rep.add(f"Orientation relative to the lattice direction (period {period:g}°): "
                + ("; ".join(f.describe(len(rel)) for f in fam.families) or "no significant families")
                + f"; uniform p = {fam.uniform_p:.3g}")
        rep.plots.append({"kind": "polar", "title": "Orientation relative to the lattice", "period": period,
                          "data": rel})
    if with_tests:
        shape = img.shape
        mask = st.exclude_mask(image_id, shape)
        allowed = None if mask is None else ~mask
        P = np.array([o.xy for o in objs])
        sc = _scale(fit)
        tests = []
        if len(ctx.defects):
            t = proximity_test(P, ctx.defects, shape, sc, radius=2 * _amin(fit),
                               n_sim=n_sim, mask=allowed)
            tests.append({"test": "molecules near defects", **_test_row(t)})
            rep.add("Defects: " + t.describe(fit.unit))
        if ctx.steps:
            t = proximity_test(P, np.zeros((0, 2)), shape, sc, radius=3 * _amin(fit),
                               n_sim=n_sim, mask=allowed,
                               distance_fn=lambda Q: distance_to_polylines(Q * np.asarray(sc), [s * np.asarray(sc) for s in ctx.steps])[0])
            tests.append({"test": "molecules near step edges", **_test_row(t)})
            rep.add("Step edges: " + t.describe(fit.unit))
        if tests:
            rep.tables["proximity tests"] = tests
    rep.provenance = provenance(st, [image_id], {**res.params, "n_objects": len(objs)}, "adsorption_map")
    return rep


def _test_row(t: Any) -> dict[str, Any]:
    return {k: getattr(t, k) for k in ("observed_mean", "random_mean", "observed_within", "random_within",
                                       "enrichment", "p_closer", "p_farther", "radius", "n_objects", "n_targets",
                                       "n_sim", "null")}


# ----------------------------------------------------------------------------------- 3 assembly analyzer
def assembly_analysis(st: SurfaceStore, image_id: str, *, labels: Optional[Sequence[str]] = None,
                      cutoff: Optional[float] = None, n_sim: int = 99) -> Report:
    from .assembly import (bond_order, clark_evans, coverage, find_chains, neighbor_graph,
                           orientation_correlation, pair_correlation)
    from .statistics import label_permutation_test

    img = image_of(st, image_id)
    fit = st.fit(image_id)
    sc, unit, _, _ = image_scale(img)
    objs = st.objects(image_id, img, labels)
    if len(objs) < 3:
        raise SurfaceError("Assembly analysis needs at least 3 molecules on the image.")
    ctx = st.context(image_id, img)
    g = neighbor_graph(objs, sc, cutoff=cutoff, mol_fold=int(st.settings["mol_fold"]), lattice=fit,
                       steps=ctx.steps, defects=ctx.defects if len(ctx.defects) else None)
    rep = Report("Molecular assembly")
    rep.extra.update(graph=g, objects=objs)
    rep.tables["neighbour pairs"] = g.pairs
    sizes = g.cluster_sizes()
    rep.add(f"{len(objs)} molecules, {len(g.edges)} neighbour pairs (cutoff {_fmt(g.cutoff, unit)}); "
            f"{len(sizes)} cluster(s), largest {max(sizes) if sizes else 0}")
    deg = Counter(g.degree().tolist())
    rep.tables["coordination"] = [{"neighbours": k, "count": v} for k, v in sorted(deg.items())]
    pc = pair_correlation(objs, img.shape, sc, n_sim=n_sim)
    rep.tables["pair correlation"] = pc.rows()
    rep.plots.append({"kind": "band", "title": "Pair correlation g(r) with random envelope", "x_label": f"r ({unit})",
                      "x": pc.r.tolist(), "y": pc.g.tolist(), "lo": pc.lo.tolist(), "hi": pc.hi.tolist()})
    if pc.peaks:
        rep.add("g(r) peaks at " + ", ".join(_fmt(p, unit) for p in pc.peaks[:5]))
    ce = clark_evans(objs, img.shape, sc)
    rep.tables["spatial statistics"] = [{"quantity": k, "value": v} for k, v in ce.items()]
    R = ce.get("R", math.nan)
    rep.add(f"Clark–Evans R = {R:.2f} (1 random, < 1 clustered, > 1 regular); p(clustered) = "
            f"{ce.get('p_clustered', math.nan):.3g}, p(regular) = {ce.get('p_regular', math.nan):.3g}")
    for fold in (4, 6):
        _, mean = bond_order(objs, sc, fold, g)
        rep.tables["spatial statistics"].append({"quantity": f"mean |psi{fold}|", "value": mean})
    rep.add("Bond order: " + ", ".join(f"{r['quantity']} {r['value']:.2f}" for r in rep.tables["spatial statistics"]
                                       if str(r["quantity"]).startswith("mean |psi")))
    cov = coverage(objs, img.shape, sc, cell_area=(fit.measurements()["cell_area"].value if fit else None))
    rep.tables["coverage"] = [{"quantity": k, "value": v} for k, v in cov.items()]
    if any(o.orientation is not None for o in objs):
        oc = orientation_correlation(objs, sc, int(st.settings["mol_fold"]))
        rep.tables["orientation correlation"] = [{"r": float(r), "correlation": float(c), "pairs": int(n)}
                                                 for r, c, n in zip(oc["r"], oc["C"], oc["n"])]
        rep.plots.append({"kind": "line", "title": "Orientation correlation", "x_label": f"r ({unit})",
                          "x": list(map(float, oc["r"])), "y": list(map(float, oc["C"]))})
    chains = find_chains(g)
    if chains:
        rep.tables["chains"] = chains
        rep.add(f"{len(chains)} straight chain(s) of ≥ 3 molecules, longest {max(c['size'] for c in chains)} molecules "
                f"({_fmt(max(c['length'] for c in chains), unit)})")
    labs = sorted({o.label for o in objs if o.label})
    if len(labs) >= 2:
        xy = np.array([o.xy for o in objs])
        L = [o.label for o in objs]
        a, b = labs[0], labs[1]
        t = label_permutation_test(xy, L, a, b, sc, radius=1.5 * g.cutoff, n_sim=999)
        rep.tables["mixing"] = [{"pair": f"{a}–{b}", **t}]
        rep.add(f"Mixing of {a} and {b}: {t['observed']:.2f} {b} neighbours per {a} vs {t['expected']:.2f} with "
                f"shuffled labels; p(more mixed) = {t['p_more']:.3g}, p(more segregated) = {t['p_less']:.3g}")
    rep.provenance = provenance(st, [image_id], {"cutoff": g.cutoff, "n_sim": n_sim, "labels": labs}, "assembly_analysis")
    return rep


# --------------------------------------------------------------------------------------- 4 defect map
def defect_map(st: SurfaceStore, image_id: str, *, propose: bool = True, **kw: Any) -> Report:
    from .defects import find_defects
    fit = require_fit(st, image_id)
    img = image_of(st, image_id)
    mask = st.exclude_mask(image_id, img.shape)
    res = find_defects(fit, img, exclude=mask, **kw)
    rep = Report("Lattice defects")
    rep.extra.update(defects=res, fit=fit)
    rep.tables["defects"] = res.rows()
    c = res.counts()
    rep.add("Found: " + (", ".join(f"{v} {k}" for k, v in c.most_common()) or "no defects")
            + f" in {res.analysed_sites} analysed lattice sites")
    if res.analysed_sites:
        rep.add(f"Vacancy concentration {100 * res.concentration('vacancy'):.2f} % of sites")
    cx = res.complexes()
    if cx:
        rep.tables["complexes"] = cx
        rep.add(f"{len(cx)} kind(s) of vacancy cluster: " + "; ".join(f"{x['count']}× ({x['complex']})" for x in cx[:5]))
    env = res.environments()
    if env:
        rep.tables["environments"] = env
    if res.lines:
        rep.tables["line defects"] = res.lines
        rep.add(f"{len(res.lines)} candidate line defect(s)")
    nn = res.nearest_neighbor_distances()
    if len(nn):
        rep.plots.append({"kind": "hist", "title": "Defect nearest-neighbour distance", "x_label": f"d ({fit.unit})",
                          "data": nn.tolist()})
    if propose:
        run = st.project.log_run("surface", "lattice defect finder", _jsonable(res.params), [image_id],
                                 {k: v for k, v in c.items()}) if hasattr(st.project, "log_run") else None
        rep.proposals = res.to_annotations(image_id, getattr(run, "id", None) or "")
        rep.add(f"{len(rep.proposals)} defect(s) can be added as proposals for review.")
    rep.provenance = provenance(st, [image_id], res.params, "defect_map")
    return rep


# --------------------------------------------------------------------------- 5 STM–ncAFM correlation
def correlation_workbench(st: SurfaceStore, image_ids: Sequence[str], *, labels: Optional[Sequence[str]] = None,
                          k: Optional[int] = None) -> Report:
    """Measure the molecules of the first image in every listed image (registered to the first), then
    cluster them on all channels together and flag outliers."""
    from .features import channel_features, cluster, outliers, shape_descriptors
    if len(image_ids) < 1:
        raise SurfaceError("Choose at least one image.")
    ref = image_ids[0]
    img = image_of(st, ref)
    sc, unit, _, _ = image_scale(img)
    objs = st.objects(ref, img, labels)
    if not objs:
        raise SurfaceError("No molecules on the reference image.")
    chans: dict[str, tuple[Any, Any]] = {}
    missing = []
    for i in image_ids:
        reg = st.registration(ref, i)
        if reg is None:
            missing.append(i)
            continue
        rel, ch = st.project.split_image_id(i)
        stem = os.path.splitext(os.path.basename(rel))[0]
        chans[f"{stem}/{ch}" if ch else stem] = (image_of(st, i), reg)
    rep = Report("Cross-channel correlation")
    if missing:
        rep.add(f"{len(missing)} image(s) are not registered to the reference and were skipped: "
                + ", ".join(missing) + " (Surface → Register images).")
    rows = channel_features(objs, chans, reference_scale=sc)
    shapes = {r["id"]: r for r in shape_descriptors(objs, img)}
    for r in rows:
        for kk, v in shapes.get(r["id"], {}).items():
            if kk not in r:
                r[kk] = v
        o = next(o for o in objs if o.id == r["id"])
        r.setdefault("label", o.label)
    keys = [kk for kk in rows[0] if kk.endswith((":contrast", ":mean", ":max")) or kk in ("radius", "elongation", "apparent_height", "n_lobes")]
    keys = [kk for kk in keys if np.all(np.isfinite([float(r.get(kk, np.nan)) for r in rows]))]
    rep.tables["features"] = rows
    if len(rows) >= 4 and keys:
        cl = cluster(rows, keys, k=k)
        for r, c in zip(rows, cl.labels):
            r["cluster"] = int(c)
        rep.add(cl.describe())
        rep.tables["cluster vs label"] = [{"cluster": int(c), "label": lab, "count": n}
                                          for (c, lab), n in sorted(Counter((int(c), r.get("label"))
                                                                            for r, c in zip(rows, cl.labels)).items())]
        out = outliers(rows, keys)
        if out:
            rep.tables["outliers"] = out
            rep.add(f"{len(out)} molecule(s) unlike the rest (robust z > 3.5): " + ", ".join(str(o['id'])[:8] for o in out[:8]))
        rep.extra["cluster"] = cl
        if len(keys) >= 2:
            rep.plots.append({"kind": "scatter", "title": f"{keys[0]} vs {keys[1]}", "x_label": keys[0], "y_label": keys[1],
                              "x": [float(r[keys[0]]) for r in rows], "y": [float(r[keys[1]]) for r in rows],
                              "groups": [int(r.get("cluster", 0)) for r in rows]})
    rep.provenance = provenance(st, list(image_ids), {"channels": list(chans), "features": keys, "k": k},
                                "correlation_workbench")
    rep.provenance["registrations"] = {n: reg.to_dict() for n, (_, reg) in chans.items()}
    return rep


# ------------------------------------------------------------------------------ 6 evolution tracker
def evolution_tracker(st: SurfaceStore, image_ids: Sequence[str], *, max_distance: Optional[float] = None,
                      dt: Optional[float] = None, labels: Optional[Sequence[str]] = None) -> Report:
    from .tracking import Frame, track
    if len(image_ids) < 2:
        raise SurfaceError("Choose at least two images in time order.")
    ref = image_ids[0]
    img0 = image_of(st, ref)
    sc, unit, _, _ = image_scale(img0)
    frames = []
    missing = []
    for i in image_ids:
        reg = st.registration(i, ref)
        if reg is None:
            missing.append(i)
            continue
        img = image_of(st, i)
        frames.append(Frame(st.objects(i, img, labels), reg, st.fit(i), name=st.project.file_name(i)))
    if len(frames) < 2:
        raise SurfaceError("Register the images to the first one first (Surface → Register images).")
    if max_distance is None:                # half the typical spacing between molecules: unambiguous links
        from .objects import nearest
        P0 = np.array([o.xy for o in frames[0].objects]).reshape(-1, 2) * np.asarray(sc)
        dnn = nearest(P0, P0, exclude_self=True)[0] if len(P0) > 1 else np.array([20 * float(np.mean(sc))])
        max_distance = 0.5 * float(np.median(dnn))
    res = track(frames, max_distance, sc, mol_fold=int(st.settings["mol_fold"]))
    rep = Report("Surface evolution")
    rep.extra["tracking"] = res
    rep.tables["events"] = res.rows()
    ev = Counter(e["type"] for e in res.events)
    rep.add(f"{len(frames)} frames, {len(res.tracks)} tracks; events: "
            + (", ".join(f"{v} {k}" for k, v in ev.most_common()) or "none (nothing moved, appeared or disappeared)"))
    if missing:
        rep.add(f"Skipped {len(missing)} unregistered image(s).")
    tm = res.transition_matrix()
    labs_t = list(tm.get("labels", []))
    if labs_t:
        C, Pm = np.asarray(tm["counts"]), np.asarray(tm["probabilities"])
        rep.tables["label transitions"] = [{"from": a, "to": b, "count": int(C[i, j]), "probability": float(Pm[i, j])}
                                           for i, a in enumerate(labs_t) for j, b in enumerate(labs_t) if C[i, j]]
    tr = res.transformations()
    if tr:
        rep.tables["candidate transformations"] = tr
        rep.add(f"{len(tr)} candidate transformation(s) (splits, merges, replacements) — check each against the images")
    m = res.msd(dt)
    if m.get("lag"):
        rep.tables["mean squared displacement"] = [{"lag": l, "msd": v, "n": n} for l, v, n in zip(m["lag"], m["msd"], m["n"])]
        if m.get("D") is not None:
            rep.add(f"Diffusion estimate D = {m['D']:.3g} {unit}²/s from the first lag")
    rep.tables["residence"] = res.residence()
    rep.provenance = provenance(st, list(image_ids), {"max_distance": max_distance, "dt": dt}, "evolution_tracker")
    return rep


# --------------------------------------------------------------------------------------- quality
def quality_check(st: SurfaceStore, image_id: str) -> Report:
    from .quality import assess
    img = image_of(st, image_id)
    q = assess(img, st.fit(image_id))
    rep = Report("Scan quality")
    rep.extra["quality"] = q
    rep.summary = q.text().splitlines()
    rep.tables["issues"] = q.rows()
    rep.tables["metrics"] = [{"quantity": k, "value": v} for k, v in q.metrics.items()]
    rep.provenance = provenance(st, [image_id], {}, "quality_check")
    return rep


# ------------------------------------------------------------------------------------- explain
def explain_object(st: SurfaceStore, image_id: str, ann_id: str) -> Report:
    from .adsorption import adsorption_registry, default_alignment
    from .features import class_templates, shape_descriptors, template_scores
    from .hypotheses import Evidence, rank
    from .quality import assess
    img = image_of(st, image_id)
    sc, unit, _, _ = image_scale(img)
    ann = st.project.annotations.get(ann_id)
    if ann is None:
        raise SurfaceError("Select one feature annotation first.")
    statuses = set(st.statuses) | {ann.status}
    from .objects import objects_from_annotations
    anns = [a for a in st.project.annotations.for_image(image_id, kinds=("feature",))
            if (a.status in statuses or a.id == ann_id) and a.attrs.get("surface") != "defect"
            and (a.label not in st.defect_labels() or a.id == ann_id)]
    objs = objects_from_annotations(anns, img, statuses=tuple(statuses) + ("proposed",),
                                    chirality=st.settings.get("chirality") or {})
    if ann_id not in {o.id for o in objs}:
        raise SurfaceError("The selected annotation is not a feature marker.")
    fit = st.fit(image_id)
    ads = None
    if fit is not None:
        ads = adsorption_registry(objs, fit, st.alignment(image_id, fit) or default_alignment(fit), st.context(image_id, img))
    reviewed = [o for o in objs if o.status in ("manual", "accepted")]
    temps = class_templates(img, reviewed) if reviewed else {}
    xy, _ = st.defects(image_id)
    # other channels registered to this image
    chans = {}
    regs = st.data["registrations"]
    for other in st.project.image_ids():
        if other != image_id and (f"{image_id}||{other}" in regs or f"{other}||{image_id}" in regs):
            chans[other] = (image_of(st, other), st.registration(image_id, other))
    from .features import channel_features
    ev = Evidence(objs, shape_descriptors(objs, img), ads, xy if len(xy) else None, assess(img, fit),
                  channel_features(objs, chans, reference_scale=sc) if chans else (),
                  template_scores(img, objs, temps) if temps else (), sc,
                  _amin(fit) if fit else None)
    hyps = rank(ann_id, ev)
    rep = Report(f"Candidate explanations for {ann.label or 'feature'} {ann_id[:8]}")
    rep.add("Scores rank these explanations against the evidence available; they are not probabilities.")
    for h in hyps[:6]:
        rep.add(h.describe())
    rep.tables["hypotheses"] = [{"hypothesis": h.name, "score": h.score,
                                 "evidence": "; ".join(f"{t.name} {t.value:.2f}: {t.why}" for t in h.terms),
                                 "missing": ", ".join(h.missing)} for h in hyps]
    rep.provenance = provenance(st, [image_id], {"target": ann_id}, "explain_object")
    return rep


# ------------------------------------------------------------------------------------ registration
def register_images(st: SurfaceStore, a: str, b: str, method: str = "auto") -> Any:
    """Register images a and b (the result maps pixel coordinates of a to b) and store it.
    method: "translation", "rigid", "lattice" (both need lattice fits) or "auto"."""
    from .registration import register_lattice, register_rigid, register_translation
    ia, ib = image_of(st, a), image_of(st, b)
    fa, fb = st.fit(a), st.fit(b)
    if method == "lattice" or (method == "auto" and fa is not None and fb is not None):
        coarse = register_translation(ia, ib) if ia.shape == ib.shape else None
        reg = register_lattice(fa, fb, coarse)
    elif method == "rigid" or (method == "auto" and ia.shape != ib.shape):
        reg = register_rigid(ia, ib)
    else:
        reg = register_translation(ia, ib)
    st.set_registration(a, b, reg)
    return reg


# ------------------------------------------------------------------------------ steps and terraces
def step_edges(st: SurfaceStore, image_id: str, **kw: Any) -> Report:
    """Detect step edges and return them as proposed step_edge annotations."""
    from ..annotations import Annotation, Origin, Polyline
    from .steps import detect_steps
    img = image_of(st, image_id)
    sc, unit, _, _ = image_scale(img)
    steps = detect_steps(img, mask=st.exclude_mask(image_id, img.shape), **kw)
    rep = Report("Step edges")
    rep.tables["steps"] = [{"step": i, "length": s.length, "height": s.height, "direction_deg": s.angle,
                            "score": s.score, "points": len(s.points)} for i, s in enumerate(steps)]
    rep.add(f"{len(steps)} step edge candidate(s)" + (f", total length {_fmt(sum(s.length for s in steps), unit)}"
                                                     if steps else ""))
    heights = [abs(s.height) for s in steps if np.isfinite(s.height)]
    if heights:
        rep.add(f"Apparent step heights: median {_fmt(float(np.median(heights)), img.unit or '')} "
                "(compare with the monatomic step height of the substrate)")
    params = {"smooth": 2.0, "threshold": 6.0, "min_length": 10.0, **kw}
    run = st.project.log_run("surface", "step edge finder", _jsonable(params), [image_id], {"steps": len(steps)})
    rep.proposals = [Annotation(image_id, "step_edge", Polyline(s.points), status="proposed",
                                origin=Origin("detector", "step edge finder", run=run.id, confidence=float(min(1.0, s.score / 20))),
                                attrs={"height": s.height})
                     for s in steps]
    rep.provenance = provenance(st, [image_id], params, "step_edges")
    return rep


def terrace_map(st: SurfaceStore, image_id: str, *, band: float = 3.0, step_height: Optional[float] = None) -> Report:
    """Terraces between the reviewed step edges: areas, relative heights, molecules per terrace."""
    from .steps import terraces
    img = image_of(st, image_id)
    steps = st.steps(image_id)
    if not steps:
        raise SurfaceError("No step edges on this image: draw them (Step edge tool, K) or detect and accept them.")
    tr = terraces(img, steps, band=band, step_height=step_height)
    rep = Report("Terraces")
    rep.extra["terraces"] = tr
    objs = st.objects(image_id, img)
    if objs:
        lab = tr.at(np.array([o.xy for o in objs]))
        cnt = Counter(int(x) for x in lab)
        for t in tr.terraces:
            t["molecules"] = cnt.get(int(t.get("terrace", t.get("id", -2))), 0)
    rep.tables["terraces"] = tr.terraces
    rep.add(f"{len(tr.terraces)} terrace(s) between {len(steps)} step edge(s)")
    rep.provenance = provenance(st, [image_id], {"band": band, "step_height": step_height}, "terrace_map")
    return rep
