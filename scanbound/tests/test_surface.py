"""Surface analyses on synthetic images with known truth."""

from __future__ import annotations

import math
import os
from collections import Counter

import numpy as np
import pytest

from scanbound.model import Axis, Image
from scanbound.surface.lattice import (bravais_candidates, calibrate_from_lattice, detect_sites, fit_image, fit_lattice,
                                       gamma_deg, reduce_basis, screen_angle, symmetry_fold)
from scanbound.surface.objects import SurfaceObject
from scanbound.surface.synthetic import lattice_image

A = 0.28e-9
PX = 0.03e-9


@pytest.fixture(scope="module")
def hexagonal():
    img, truth = lattice_image(a=A, gamma=120, angle=12, size=(256, 256), pixel=PX, vacancies=0.02, adatoms=6,
                               noise=0.05, seed=1)
    return img, truth, fit_image(img)


# ------------------------------------------------------------------------------------------- lattice
def test_lattice_fit_recovers_the_cell(hexagonal):
    img, truth, fit = hexagonal
    ms = fit.measurements()
    assert ms["a1"].value == pytest.approx(A, rel=2e-3)
    assert ms["a2"].value == pytest.approx(A, rel=2e-3)
    assert ms["gamma"].value == pytest.approx(120.0, abs=0.2)
    assert 0 < ms["a1"].uncertainty < 0.01 * A
    # direction of a1 is one of the six equivalent directions of the true lattice
    d = (ms["a1_angle"].value - screen_angle(truth["a1"])) % 60.0
    assert min(d, 60 - d) < 0.3
    assert symmetry_fold(fit) == 6
    assert bravais_candidates(fit)[0].name == "hexagonal"
    assert fit.n_sites > 500


def test_reduce_basis_convention():
    t = math.radians(30)
    a1 = np.array([math.cos(t), -math.sin(t)])
    a2 = np.array([math.cos(t + math.pi / 2), -math.sin(t + math.pi / 2)])
    ref = reduce_basis(a1, a2)
    for v in ((a2, -a1), (-a1, -a2), (-a2, a1)):
        r = reduce_basis(*v)
        assert screen_angle(r[0]) == pytest.approx(screen_angle(ref[0]))
        assert gamma_deg(*r) == pytest.approx(90.0)


def test_anisotropic_pixels_and_calibration():
    img, truth = lattice_image(a=0.4e-9, gamma=90, angle=30, size=(150, 300), pixel=0.04e-9, pixel_y=0.06e-9, seed=2)
    fit = fit_image(img)
    ms = fit.measurements()
    assert ms["a1"].value == pytest.approx(0.4e-9, rel=5e-3)
    assert ms["gamma"].value == pytest.approx(90.0, abs=0.5)
    img2, _ = lattice_image(a=A, size=(200, 200), pixel=PX, seed=3)
    r = calibrate_from_lattice(fit_image(img2), A * 1.05, gamma_known=120)
    assert r["scale_x"] == pytest.approx(1.05, rel=2e-3)
    assert r["dx"] == pytest.approx(PX * 1.05, rel=2e-3)


def test_lattice_fit_round_trips(hexagonal):
    from scanbound.surface.lattice import LatticeFit
    _, _, fit = hexagonal
    back = LatticeFit.from_dict(fit.to_dict())
    np.testing.assert_allclose(back.a1, fit.a1)
    np.testing.assert_allclose(back.cov, fit.cov)
    assert back.n_sites == fit.n_sites


# ------------------------------------------------------------------------------- distortion, domains
def test_local_distortion_finds_a_bump():
    from scanbound.surface.distortion import local_distortion
    c = np.array([100 * PX, 100 * PX])

    def disp(x, y):
        r = np.array([x, y]) - c
        return 0.08 * r * np.exp(-(r ** 2).sum() / (1.5e-9) ** 2)
    img, _ = lattice_image(size=(200, 200), displacement=disp, seed=3, noise=0.03)
    d = local_distortion(fit_image(img))
    dil = d.values("dilatation")
    r = np.hypot(*(d.xy - 100).T)
    assert np.nanmean(dil[r < 10]) > 0.08
    far = (r > 60) & (d.confidence >= 0.5)
    assert abs(np.nanmedian(dil[far])) < 0.04          # the bump also shifts the mean lattice slightly
    assert np.isfinite(d.map("dilatation")).mean() > 0.5


def test_rotation_domains():
    from scanbound.surface.domains import find_domains
    a, _ = lattice_image(size=(200, 200), angle=10, seed=4)
    b, _ = lattice_image(size=(200, 200), angle=25, seed=5)
    z = np.array(a.data)
    z[:, 100:] = b.data[:, 100:]
    img = Image(z, x=a.x, y=a.y, unit="m")
    dr = find_domains(detect_sites(img, spacing=A), img)
    rot = [d for d in dr.domains if d.kind == "rotation"] or dr.domains
    angles = sorted(d.orientation % 60 for d in rot[:2])
    assert len(rot) >= 2
    assert angles[0] == pytest.approx(10, abs=1.5) and angles[1] == pytest.approx(25, abs=1.5)
    assert dr.boundaries


def test_superstructure_found_and_not_invented():
    from scanbound.surface.superstructure import analyse_periodicities, moire_period
    img, _ = lattice_image(size=(240, 240), vacancies=0.02, adatoms=30, seed=3)
    assert analyse_periodicities(img, fit_image(img)).superstructure is None      # noise is not a superstructure
    # (2 × 2): every second site brighter along a1 and along a2
    base, t = lattice_image(size=(256, 256), angle=10, seed=7, noise=0.02)
    yy, xx = np.mgrid[0:256, 0:256]
    P = np.stack([(xx + .5) * PX, (yy + .5) * PX], -1) - t["origin"]
    uv = np.linalg.solve(np.column_stack([t["a1"], t["a2"]]), P.reshape(-1, 2).T).T.reshape(256, 256, 2)
    z = np.array(base.data) * (1 + 0.3 * (np.cos(np.pi * uv[..., 0]) + np.cos(np.pi * uv[..., 1])))
    img2 = Image(z, x=base.x, y=base.y, unit="m")
    rep = analyse_periodicities(img2, fit_image(img2, spacing=A))
    assert rep.superstructure is not None and rep.superstructure.kind == "commensurate"
    assert rep.superstructure.wood == "(2 × 2)"
    assert rep.candidates[0].name == "(2 × 2)"
    L, phi = moire_period(0.246e-9, 0.246e-9, 1.0)
    assert L == pytest.approx(0.246e-9 / (2 * math.sin(math.radians(0.5))), rel=1e-6)


# ------------------------------------------------------------------------------- models, sites
def test_presets_and_alignment():
    from scanbound.surface.structure import SurfaceModel, align, preset
    m = preset("fcc(111)", "Cu")
    kinds = Counter(s.kind for s in m.sites)
    assert kinds["top"] == 1 and kinds["hollow"] == 2 and kinds["bridge"] >= 1
    assert len(preset("graphene").sites) >= 3
    img, _ = lattice_image(a=0.2556e-9, angle=12, size=(200, 200), pixel=PX)
    fit = fit_image(img)
    al = align(m, fit)
    assert al and al[0].scale_ratio == pytest.approx(1.0, abs=0.01) and al[0].angle_error < 0.5
    assert len(al[0].atom_pixels(img.shape)[0][1]) > 100
    back = SurfaceModel.from_dict(m.to_dict())
    assert back.sites == m.sites


def test_adsorption_sites_are_assigned():
    from scanbound.surface.adsorption import adsorption_registry
    from scanbound.surface.structure import align, preset
    img, t = lattice_image(a=0.2556e-9, angle=12, size=(256, 256), pixel=PX, seed=2)
    fit = fit_image(img)
    A_ = np.column_stack([t["a1"], t["a2"]])
    rng = np.random.default_rng(0)
    objs = []
    for k in range(60):
        m, n = rng.integers(-8, 8, 2)
        frac = [2 / 3, 1 / 3] if k % 3 else [0.5, 0.0]
        p = (t["origin"] + (np.array([m, n]) + frac) @ A_.T) / PX + rng.normal(0, 0.3, 2)
        objs.append(SurfaceObject(str(k), p, 0.5, "mol", orientation=(12 + 60 * rng.integers(0, 6)) % 360))
    res = adsorption_registry(objs, fit, align(preset("fcc(111)", "Cu"), fit)[0], mol_fold=2, sub_fold=6)
    occ = res.occupancy()["counts"]
    hollows = sum(v for k, v in occ.items() if k.startswith("hollow"))
    bridges = sum(v for k, v in occ.items() if "bridge" in k)
    assert hollows == 40 and bridges == 20
    assert res.uniformity_p() < 1e-6
    assert all(0 <= r["site_p"] <= 1 for r in res.rows)


def test_defects(hexagonal):
    from scipy.spatial import cKDTree
    from scanbound.surface.defects import find_defects
    img, truth, fit = hexagonal
    r = find_defects(fit, img)
    c = r.counts()
    vac = np.array([d.xy for d in r.defects if d.kind == "vacancy"])
    d, _ = cKDTree(truth["vacancies"]).query(vac)
    assert (d < 1.5).sum() >= 0.6 * len(truth["vacancies"])
    assert (d < 1.5).mean() > 0.8                      # few false vacancies
    assert c["adatom"] >= len(truth["adatoms"]) - 1
    anns = r.to_annotations("img")
    assert anns and all(a.status == "proposed" for a in anns)


def test_steps_and_terraces():
    from scanbound.surface.steps import detect_steps, terraces
    rows, cols = 200, 300
    yy, xx = np.mgrid[0:rows, 0:cols]
    z = ((xx > 80 + 0.3 * yy).astype(float) + (xx > 190 + 0.2 * yy)) * 2e-10
    z += 1e-12 * np.random.default_rng(0).standard_normal((rows, cols))
    img = Image(z, x=Axis.spatial("x", cols, 0.1e-9), y=Axis.spatial("y", rows, 0.1e-9), unit="m")
    st = detect_steps(img)
    assert len(st) == 2
    assert all(abs(abs(s.height) - 2e-10) < 0.3e-10 for s in st)
    ter = terraces(img, [s.points for s in st])
    assert len(ter.terraces) == 3


# -------------------------------------------------------------------------------- assembly, stats
def test_assembly_statistics():
    from scanbound.surface.assembly import Motif, clark_evans, find_motif, neighbor_graph, pair_correlation
    rng = np.random.default_rng(1)
    objs, k = [], 0
    for _ in range(30):
        c = rng.uniform(20, 480, 2)
        t = rng.uniform(0, np.pi)
        d = np.array([np.cos(t), -np.sin(t)]) * 6
        for s, h in ((1, "L"), (-1, "R")):
            objs.append(SurfaceObject(str(k), c + s * d / 2, 0.5, "mol", orientation=np.degrees(t) % 360, handedness=h))
            k += 1
    sc = (0.1e-9, 0.1e-9)
    g = neighbor_graph(objs, sc, cutoff=0.8e-9)
    assert Counter(g.cluster_sizes())[2] >= 25
    pc = pair_correlation(objs, (500, 500), sc, r_max=5e-9, bins=25, n_sim=19)
    assert pc.g[np.argmin(abs(pc.r - 0.6e-9))] > pc.hi[np.argmin(abs(pc.r - 0.6e-9))]
    assert clark_evans(objs, (500, 500), sc)["R"] < 0.8
    hits = find_motif(Motif.from_objects(objs[:2], sc), objs, sc)
    assert len(hits) >= 25


def test_orientation_and_chirality_statistics():
    from scanbound.surface.statistics import chirality_stats, compare_groups, orientation_families, relative_orientation
    rng = np.random.default_rng(0)
    assert not orientation_families(rng.uniform(0, 180, 100), period=180).families
    a = np.concatenate([rng.normal(30, 3, 40), rng.normal(120, 3, 30), rng.uniform(0, 180, 20)]) % 180
    centres = sorted(round(f.centre) for f in orientation_families(a, period=180).families)
    assert len(centres) == 2 and abs(centres[0] - 30) < 3 and abs(centres[1] - 120) < 3
    cs = chirality_stats(["L"] * 30 + ["R"] * 10 + [None] * 5)
    assert cs["n_known"] == 40 and cs["racemic_p"] < 0.01
    assert relative_orientation(70.0, 10.0, 1, 6) == pytest.approx(0.0)
    assert compare_groups({"a": {"top": 10, "hollow": 30}, "b": {"top": 30, "hollow": 10}}).p < 0.01


# ------------------------------------------------------------------------- registration, tracking
def test_registration():
    from scipy import ndimage
    from scanbound.surface.registration import register_landmarks, register_rigid, register_translation
    rng = np.random.default_rng(0)
    a = ndimage.gaussian_filter(rng.standard_normal((256, 256)), 3)
    b = ndimage.shift(a, (2.5, -4.25))
    r = register_translation(a, b)
    np.testing.assert_allclose(r.translation, (-4.25, 2.5), atol=0.1)
    from skimage.transform import AffineTransform, warp
    t = math.radians(-7)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = np.array([128, 128.]) - R @ [128, 128] + [5.3, -3.2]
    b2 = warp(a, AffineTransform(matrix=M).inverse, preserve_range=True)
    rr = register_rigid(a, b2)
    p = np.array([[60.0, 70.0], [200.0, 180.0]])
    np.testing.assert_allclose(rr.apply(p), (M @ np.c_[p, np.ones(2)].T).T[:, :2], atol=1.0)
    q = rng.uniform(0, 256, (6, 2))
    L = register_landmarks(q, (M @ np.c_[q, np.ones(6)].T).T[:, :2], "similarity")
    assert L.error_px < 0.1


def test_tracking_events():
    from scanbound.surface.registration import Registration
    from scanbound.surface.tracking import Frame, track
    rng = np.random.default_rng(3)
    f0 = [SurfaceObject(str(i), rng.uniform(20, 230, 2), 0.3, "mol") for i in range(10)]
    f1 = [SurfaceObject(o.id + "'", o.xy + [0.1, 0.0], 0.3, "mol") for o in f0[1:]]
    f1[0].xy = f1[0].xy + [8, 0]
    f1.append(SurfaceObject("a", f0[0].xy + [3, 0], 0.3, "frag"))
    f1.append(SurfaceObject("b", f0[0].xy - [3, 0], 0.3, "frag"))
    tr = track([Frame(f0), Frame(f1, Registration.identity())], max_distance=1.5e-9, scale=(0.1e-9, 0.1e-9))
    ev = Counter(e["type"] for e in tr.events)
    assert ev["moved"] >= 1
    assert any(t["type"] == "split" for t in tr.transformations())


def test_quality_flags():
    from scanbound.surface.quality import assess
    img, _ = lattice_image(size=(200, 200), seed=3)
    fit = fit_image(img)
    assert assess(img, fit).score > 0.8
    z = np.array(img.data)
    z[80] += 5e-11
    z[120:] += 3e-11
    r = assess(Image(z, x=img.x, y=img.y, unit="m"), fit)
    codes = {f.code for f in r.issues}
    assert 80 in r.bad_rows or "streak" in codes or "line_jump" in codes


def test_hypotheses_rank_a_fragment():
    from scanbound.surface.features import class_templates, shape_descriptors, template_scores
    from scanbound.surface.hypotheses import Evidence, rank
    rng = np.random.default_rng(3)
    mols = [{"x": rng.uniform(20, 230), "y": rng.uniform(20, 230), "angle": rng.uniform(0, 180), "length": 1.0e-9,
             "height": 2.0} for _ in range(25)]
    mols.append({"x": 128, "y": 128, "angle": 30, "length": 0.3e-9, "height": 2.0})
    img, _ = lattice_image(size=(256, 256), molecules=mols, seed=1, noise=0.03)
    objs = [SurfaceObject(str(i), np.array([m["x"], m["y"]]), 0.5, "mol", orientation=m["angle"], area_px=700)
            for i, m in enumerate(mols)]
    sh = shape_descriptors(objs, img)
    ev = Evidence(objs, shapes=sh, templates=template_scores(img, objs, class_templates(img, objs)),
                  scale=img.pixel_size)
    hyps = rank("25", ev)
    assert hyps and hyps[0].name != "mol (typical)"
    assert any("missing" in h.describe() for h in hyps)


# ---------------------------------------------------------------------------- project + workflows
@pytest.fixture
def surface_project(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from scanbound.annotations import Annotation, Point, Polygon, Polyline
    from scanbound.project import Project
    img, truth = lattice_image(size=(240, 240), vacancies=0.02, adatoms=30, seed=3)
    a = np.asarray(img.data)
    u16 = ((a - a.min()) / np.ptp(a) * 65535).astype(np.uint16)
    cv2.imwrite(str(tmp_path / "s1.png"), u16)
    cv2.imwrite(str(tmp_path / "s2.png"), np.roll(u16, (2, 3), axis=(0, 1)))
    p = Project.open_folder(str(tmp_path))
    for i in p.image_ids():
        p.processed_image(i)
    ids = sorted(p.image_ids())
    for i in ids:
        p.set_calibration(i, *truth["pixel"])
    rng = np.random.default_rng(1)
    for k, xy in enumerate(truth["adatoms"]):
        p.annotations.add(Annotation(ids[0], "feature", Point(*xy), label="mol",
                                     orientation=float(rng.choice([0, 60]) + rng.normal(0, 3)),
                                     attrs={"handedness": "L" if k % 2 else "R"}))
        p.annotations.add(Annotation(ids[1], "feature", Point(xy[0] + 3, xy[1] + 2), label="mol"))
    for xy in truth["vacancies"][:5]:
        p.annotations.add(Annotation(ids[0], "feature", Point(*xy), label="vacancy"))
    p.annotations.add(Annotation(ids[0], "step_edge", Polyline([(0, 120), (240, 130)])))
    p.annotations.add(Annotation(ids[0], "region", Polygon([(0, 0), (30, 0), (30, 30), (0, 30)]), label="exclude"))
    return p, ids, truth


def test_store_and_workflows(surface_project, tmp_path):
    from scanbound.project import Project
    from scanbound.surface import workflow as wf
    from scanbound.surface.store import SurfaceStore, remap_image_ids
    from scanbound.surface.structure import preset
    p, (i1, i2), truth = surface_project
    st = SurfaceStore(p)
    fit = wf.run_lattice_fit(st, i1)
    assert fit.measurements()["a1"].value == pytest.approx(A, rel=3e-3)
    wf.run_lattice_fit(st, i2)
    m = preset("fcc(111)", "Cu")
    st.set_model(m)
    from scanbound.surface.structure import align
    al = align(m, fit)[0]
    st.set_alignment(i1, m.name, al.T, al.offset, al.mirrored)
    assert {o.label for o in st.objects(i1)} == {"mol"}                 # vacancies are defects, not molecules
    assert len(st.objects(i1)) == 30
    assert len(st.defects(i1)[0]) == 5 and len(st.steps(i1)) == 1

    rep = wf.structure_explorer(st, i1)
    assert "lattice" in rep.tables and rep.tables["domains"]
    rep = wf.adsorption_map(st, i1, n_sim=49)
    occ = {r["site"]: r["count"] for r in rep.tables["site occupancy"]}
    assert sum(occ.values()) == 30 and max(occ, key=occ.get).startswith("hollow")
    assert rep.tables["chirality"] and rep.tables["proximity tests"]
    rep = wf.assembly_analysis(st, i1, n_sim=19)
    assert rep.tables["pair correlation"]
    rep = wf.defect_map(st, i1)
    assert rep.proposals and all(a.status == "proposed" for a in rep.proposals)
    assert wf.quality_check(st, i1).tables["metrics"]
    files = rep.save(str(tmp_path / "out"))
    assert any(f.endswith("defects.csv") for f in files) and any(f.endswith("report.json") for f in files)

    reg = wf.register_images(st, i1, i2)
    np.testing.assert_allclose(reg.translation, (3, 2), atol=0.2)
    rep = wf.evolution_tracker(st, [i1, i2])
    assert len(rep.extra["tracking"].tracks) == 30
    rep = wf.correlation_workbench(st, [i1, i2])
    assert rep.tables["features"]
    target = p.annotations.for_image(i1, labels=["mol"])[0].id
    rep = wf.explain_object(st, i1, target)
    assert rep.tables["hypotheses"][0]["hypothesis"].startswith("mol")

    with pytest.raises(wf.SurfaceError):
        wf.terrace_map(st, i2)

    path = p.save()
    q = Project.load(path)
    s2 = SurfaceStore(q)
    assert s2.fit(i1).n_sites == fit.n_sites
    assert s2.alignment(i1) is not None and s2.registration(i2, i1) is not None

    d = {"images": {"a": 1}, "registrations": {"a||b": 2}}
    remap_image_ids(d, {"a": "x", "b": "y"})
    assert d == {"images": {"x": 1}, "registrations": {"x||y": 2}}


# ------------------------------------------------------------------------------------- 3D builder
BENZENE = """12
benzene
C 1.39 0.00 0.00
C 0.695 1.204 0.00
C -0.695 1.204 0.00
C -1.39 0.00 0.00
C -0.695 -1.204 0.00
C 0.695 -1.204 0.00
H 2.47 0.00 0.00
H 1.235 2.139 0.00
H -1.235 2.139 0.00
H -2.47 0.00 0.00
H -1.235 -2.139 0.00
H 1.235 -2.139 0.00
"""

ROD = "4\nrod, chiral kink\nC 0 0 0\nC 1.5 0 0\nC 3.0 0 0\nC 3.8 1.2 0\n"


def test_builder_scene_export_and_state(surface_project, tmp_path):
    from scanbound.project import Project
    from scanbound.surface import workflow as wf
    from scanbound.surface.builder import (BuildSettings, MoleculeTemplate, build_scene, load_state, read_xyz,
                                           save_state)
    from scanbound.surface.store import SurfaceStore
    from scanbound.surface.structure import align, preset

    p, ids, truth = surface_project
    st = SurfaceStore(p)
    iid = ids[0]
    fit = wf.run_lattice_fit(st, iid)
    bz = MoleculeTemplate.from_xyz(BENZENE, name="benzene")
    rod = MoleculeTemplate.from_xyz(ROD, name="rod", handedness="L")
    g = rod.geometry()
    assert g[:, 2].min() == pytest.approx(0.0) and np.ptp(g[:, 2]) < 1e-6        # laid flat, on z = 0
    assert np.ptp(g[:, 0]) > np.ptp(g[:, 1])                                     # long axis along x
    n_mol = len(st.objects(iid, p.processed_image(iid)))

    # no model: one atom of the chosen species per lattice maximum
    sc = build_scene(st, iid, {"benzene": bz}, {"mol": "benzene"}, BuildSettings(species="Cu", layers=2))
    assert len(sc.placements) == n_mol and set(sc.substrate_symbols) == {"Cu"} and sc.notes

    # with an aligned fcc(111) model: layers at the model spacing, sites assigned when snapping
    m = preset("fcc(111)", "Cu")
    st.set_model(m)
    al = align(m, fit)[0]
    st.set_alignment(iid, m.name, al.T, al.offset, al.mirrored)
    sc = build_scene(st, iid, {"rod": rod}, {"mol": "rod"}, BuildSettings(layers=3, snap=True, height=2.5))
    zs = np.unique(sc.substrate_positions[:, 2].round(2))
    assert len(zs) == 3 and zs.max() == pytest.approx(0.0) and np.diff(zs) == pytest.approx([2.08, 2.08], abs=0.05)
    assert all(pl.site for pl in sc.placements)
    assert any(pl.mirrored for pl in sc.placements) and not all(pl.mirrored for pl in sc.placements)  # R annotations
    pl = sc.placements[0]
    s, P = sc.molecule(pl)
    assert P[:, 2].min() == pytest.approx(2.5)                                   # height = surface to lowest atom
    chk = {r["id"]: r for r in sc.check()}
    assert chk[pl.id]["substrate_ratio"] > 1.0 and chk[pl.id]["closest_substrate"] > 2.0
    pl.height = 0.4
    low = {r["id"]: r for r in sc.check()}[pl.id]
    assert low["clash"] and low["substrate_ratio"] < 1.0

    # manual adjustments survive a rebuild
    pl.height, pl.angle = 3.3, pl.angle + 17.0
    sc2 = build_scene(st, iid, {"rod": rod}, {"mol": "rod"}, BuildSettings(layers=2), previous=sc.placements)
    q = {x.id: x for x in sc2.placements}[pl.id]
    assert q.height == pytest.approx(3.3) and q.angle == pytest.approx(pl.angle)

    out = tmp_path / "model.xyz"
    n = sc2.write_xyz(str(out))
    syms, pos, comment = read_xyz(str(out))
    assert len(syms) == n == len(sc2.substrate_symbols) + 4 * len(sc2.placements)
    assert "Lattice=" in out.read_text().splitlines()[1] and (tmp_path / "model_placements.json").exists()
    try:
        import ase.io
        atoms = ase.io.read(str(out))
        assert len(atoms) == n and set(atoms.arrays["molecule"]) >= {-1, 0}
    except ImportError:
        pass

    save_state(st, iid, sc2, {"rod": rod}, {"mol": "rod"}, sc2.settings)
    path = p.save()
    templates, label_map, settings, places = load_state(SurfaceStore(Project.load(path)), iid)
    assert label_map == {"mol": "rod"} and templates["rod"].handedness == "L" and settings.layers == 2
    assert {x.id: x for x in places}[pl.id].height == pytest.approx(3.3)
