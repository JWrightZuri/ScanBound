"""
Following objects through a series of scans: tracks, events, motion relative to the lattice,
transformations and class transitions.

    frames = [Frame(objs0), Frame(objs1, reg_1_to_0, fit1), ...]       # registrations map each frame onto frame 0
    tr = track(frames, max_distance=0.6e-9, scale=(dx, dy))
    tr.tracks                     # object ids per frame, positions in frame-0 pixels
    tr.events                     # appeared / disappeared / moved / changed class / rotated, per frame step
    tr.transformations()          # candidate reactions: one object replaced by several (split) or several by one (merge)
    tr.transition_matrix()        # how often class A is followed by class B
    tr.msd(dt)                    # mean squared displacement and a diffusion estimate

Drift and the lattice
---------------------
Positions are compared after mapping every frame onto frame 0 with its registration.  When the
frames have lattice fits, displacements are also expressed in lattice units of frame 0 and rounded to
the nearest lattice vector ("hops"), which separates a real jump from one site to another from
residual drift.  The registration error is carried into the motion threshold: a displacement
smaller than ``min_move_sigma`` × the combined position and registration uncertainty is not called
motion.  Apparent changes of shape or class across frames may also be tip changes - check the
quality report of each frame before reading them as molecular events.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .lattice import LatticeFit
from .objects import SurfaceObject
from .registration import Registration, link_objects


@dataclass
class Frame:
    objects: list[SurfaceObject]
    registration: Optional[Registration] = None     # this frame's pixels -> frame 0 pixels
    fit: Optional[LatticeFit] = None
    name: str = ""
    time: Optional[float] = None                    # seconds (for rates and diffusion)


@dataclass
class Track:
    id: int
    entries: dict[int, int] = field(default_factory=dict)       # frame -> object index in that frame

    @property
    def frames(self) -> list[int]:
        return sorted(self.entries)

    @property
    def length(self) -> int:
        return len(self.entries)


@dataclass
class TrackingResult:
    frames: list[Frame]
    tracks: list[Track]
    events: list[dict[str, Any]]
    positions: list[np.ndarray]                     # per frame: (N, 2) positions in frame-0 pixels
    scale: tuple[float, float]
    params: dict = field(default_factory=dict)

    def track_of(self, frame: int, index: int) -> Optional[Track]:
        for t in self.tracks:
            if t.entries.get(frame) == index:
                return t
        return None

    def rows(self) -> list[dict[str, Any]]:
        out = []
        sc = np.asarray(self.scale)
        for t in self.tracks:
            for f in t.frames:
                o = self.frames[f].objects[t.entries[f]]
                p = self.positions[f][t.entries[f]]
                out.append({"track": t.id, "frame": f, "object": o.id, "label": o.label, "x_px": float(p[0]),
                            "y_px": float(p[1]), "x": float(p[0] * sc[0]), "y": float(p[1] * sc[1]),
                            "orientation": o.orientation})
        return out

    def transition_matrix(self) -> dict[str, Any]:
        """Counts and probabilities of label A (frame t) -> label B (frame t + 1) along tracks."""
        c: Counter = Counter()
        for t in self.tracks:
            fr = t.frames
            for f0, f1 in zip(fr[:-1], fr[1:]):
                if f1 != f0 + 1:
                    continue
                a = self.frames[f0].objects[t.entries[f0]].label
                b = self.frames[f1].objects[t.entries[f1]].label
                c[(str(a), str(b))] += 1
        labels = sorted({k[0] for k in c} | {k[1] for k in c})
        M = np.array([[c[(a, b)] for b in labels] for a in labels], float).reshape(len(labels), len(labels))
        P = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        return {"labels": labels, "counts": M, "probabilities": P,
                "note": "correlations along tracks under these measurement conditions, not established causes"}

    def transformations(self, radius: Optional[float] = None) -> list[dict[str, Any]]:
        """Candidate splits (one disappears, several appear nearby) and merges (several disappear,
        one appears), from consecutive frames."""
        out = []
        sc = np.asarray(self.scale)
        r = radius or self.params.get("max_distance", 1.0) * 2
        for f in range(1, len(self.frames)):
            gone = [e for e in self.events if e["frame"] == f and e["type"] == "disappeared"]
            new = [e for e in self.events if e["frame"] == f and e["type"] == "appeared"]
            for g in gone:
                near = [n for n in new if math.hypot(*((np.array(n["xy"]) - g["xy"]) * sc)) <= r]
                if len(near) >= 2:
                    out.append({"frame": f, "type": "split", "before": [g["object"]], "after": [n["object"] for n in near],
                                "before_labels": [g["label"]], "after_labels": [n["label"] for n in near]})
            for n in new:
                near = [g for g in gone if math.hypot(*((np.array(g["xy"]) - n["xy"]) * sc)) <= r]
                if len(near) >= 2:
                    out.append({"frame": f, "type": "merge", "before": [g["object"] for g in near], "after": [n["object"]],
                                "before_labels": [g["label"] for g in near], "after_labels": [n["label"]]})
            # an object linked to a product of another class, with more new objects next to it: a split
            for c in (e for e in self.events if e["frame"] == f and e["type"] == "changed class"):
                near = [n for n in new if math.hypot(*((np.array(n["xy"]) - c["xy_before"]) * sc)) <= r]
                if near:
                    out.append({"frame": f, "type": "split", "before": [c["before_object"]],
                                "after": [c["object"]] + [n["object"] for n in near],
                                "before_labels": [c["before"]], "after_labels": [c["label"]] + [n["label"] for n in near]})
            for g in gone:
                near = [n for n in new if math.hypot(*((np.array(n["xy"]) - g["xy"]) * sc)) <= r]
                if len(near) == 1 and near[0]["label"] != g["label"]:
                    out.append({"frame": f, "type": "replaced", "before": [g["object"]], "after": [near[0]["object"]],
                                "before_labels": [g["label"]], "after_labels": [near[0]["label"]]})
        return out

    def msd(self, dt: Optional[float] = None, max_lag: int = 10) -> dict[str, Any]:
        """Mean squared displacement against frame lag (physical units²) and D = MSD / (4 τ) from the
        first lag (needs ``dt``, seconds per frame, or frame times)."""
        sc = np.asarray(self.scale)
        lags, msd, n = [], [], []
        for lag in range(1, max_lag + 1):
            d2 = []
            for t in self.tracks:
                for f in t.frames:
                    if f + lag in t.entries:
                        p0 = self.positions[f][t.entries[f]] * sc
                        p1 = self.positions[f + lag][t.entries[f + lag]] * sc
                        d2.append(float(((p1 - p0) ** 2).sum()))
            if d2:
                lags.append(lag)
                msd.append(float(np.mean(d2)))
                n.append(len(d2))
        out: dict[str, Any] = {"lag": lags, "msd": msd, "n": n}
        if dt and msd:
            out["D"] = msd[0] / (4 * dt)
        return out

    def residence(self) -> list[dict[str, Any]]:
        """For each track: frames spent without moving (consecutive 'stayed' steps)."""
        moved = {(e["frame"], e["track"]) for e in self.events if e["type"] == "moved"}
        out = []
        for t in self.tracks:
            run, runs = 1, []
            fr = t.frames
            for f0, f1 in zip(fr[:-1], fr[1:]):
                if f1 == f0 + 1 and (f1, t.id) not in moved:
                    run += 1
                else:
                    runs.append(run)
                    run = 1
            runs.append(run)
            out.append({"track": t.id, "frames": t.length, "longest_stay": max(runs), "stays": runs})
        return out


def track(frames: Sequence[Frame], max_distance: float, scale: tuple[float, float] = (1.0, 1.0),
          min_move_sigma: float = 3.0, rotate_threshold: float = 20.0, mol_fold: int = 1) -> TrackingResult:
    """
    Link objects frame to frame (Hungarian assignment within ``max_distance``, physical units, on
    positions mapped to frame 0).  Events per step: ``appeared``, ``disappeared``, ``moved`` (beyond
    ``min_move_sigma`` uncertainties; with lattice fits also the hop in lattice vectors),
    ``changed class`` and ``rotated`` (by more than ``rotate_threshold`` degrees, modulo the
    molecule's symmetry).
    """
    sc = np.asarray(scale, float)
    pos = []
    for fr in frames:
        P = np.array([o.xy for o in fr.objects], float).reshape(-1, 2)
        pos.append(fr.registration.apply(P) if fr.registration is not None else P)
    tracks: list[Track] = []
    current: dict[int, Track] = {}
    for i in range(len(frames[0].objects) if frames else 0):
        t = Track(len(tracks), {0: i})
        tracks.append(t)
        current[i] = t
    events: list[dict[str, Any]] = []
    period = 360.0 / max(1, mol_fold)
    fit0 = frames[0].fit if frames else None
    for f in range(1, len(frames)):
        a, b = frames[f - 1], frames[f]

        class _O:  # positions already in frame-0 pixels
            def __init__(self, xy):  # noqa: ANN001
                self.xy = xy
        # link in physical units: scale pixel positions so the gate is isotropic
        la = [_O(p * sc) for p in pos[f - 1]]
        lb = [_O(p * sc) for p in pos[f]]
        L = link_objects(la, lb, None, max_distance)
        reg_err = 0.0
        for fr in (a, b):
            if fr.registration is not None and np.isfinite(fr.registration.error_px):
                reg_err = math.hypot(reg_err, fr.registration.error_px)
        nxt: dict[int, Track] = {}
        for i, j, d in L["pairs"]:
            t = current.get(i)
            if t is None:
                t = Track(len(tracks), {f - 1: i})
                tracks.append(t)
            t.entries[f] = j
            nxt[j] = t
            oa, ob = a.objects[i], b.objects[j]
            sig = math.hypot(math.hypot(oa.sigma_px, ob.sigma_px), reg_err) * float(np.sqrt(sc[0] * sc[1]))
            if d > min_move_sigma * sig:
                ev = {"frame": f, "type": "moved", "track": t.id, "object": ob.id, "label": ob.label,
                      "distance": d, "sigma": sig, "xy": pos[f][j].tolist()}
                if fit0 is not None:
                    duv = np.linalg.solve(fit0.A, (pos[f][j] - pos[f - 1][i]) * sc)
                    ev["du"], ev["dv"] = float(duv[0]), float(duv[1])
                    ev["hop"] = [int(round(duv[0])), int(round(duv[1]))]
                events.append(ev)
            if oa.label != ob.label:
                events.append({"frame": f, "type": "changed class", "track": t.id, "object": ob.id, "label": ob.label,
                               "before": oa.label, "before_object": oa.id, "xy": pos[f][j].tolist(),
                               "xy_before": pos[f - 1][i].tolist()})
            if oa.orientation is not None and ob.orientation is not None:
                dd = (ob.orientation - oa.orientation) % period
                dd = min(dd, period - dd)
                if dd > rotate_threshold:
                    events.append({"frame": f, "type": "rotated", "track": t.id, "object": ob.id, "label": ob.label,
                                   "angle": dd, "xy": pos[f][j].tolist()})
        for i in L["only_a"]:
            t = current.get(i)
            events.append({"frame": f, "type": "disappeared", "track": t.id if t else -1, "object": a.objects[i].id,
                           "label": a.objects[i].label, "xy": pos[f - 1][i].tolist()})
        for j in L["only_b"]:
            t = Track(len(tracks), {f: j})
            tracks.append(t)
            nxt[j] = t
            events.append({"frame": f, "type": "appeared", "track": t.id, "object": b.objects[j].id,
                           "label": b.objects[j].label, "xy": pos[f][j].tolist()})
        current = nxt
    return TrackingResult(list(frames), tracks, events, pos, (float(sc[0]), float(sc[1])),
                          {"max_distance": max_distance, "min_move_sigma": min_move_sigma,
                           "rotate_threshold": rotate_threshold})


def compare_before_after(before: Sequence[SurfaceObject], after: Sequence[SurfaceObject], reg: Registration,
                         scale: tuple[float, float], max_distance: float) -> dict[str, Any]:
    """Two scans around a manipulation or reaction: unchanged, moved, disappeared, appeared and
    candidate transformations, as one event record."""
    tr = track([Frame(list(before)), Frame(list(after), reg)], max_distance, scale)
    return {"events": tr.events, "transformations": tr.transformations(),
            "unchanged": sum(1 for t in tr.tracks if t.length == 2) - sum(1 for e in tr.events if e["type"] == "moved"),
            "summary": Counter(e["type"] for e in tr.events)}
