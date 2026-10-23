"""
scanbound.labeling
==================

Rules that turn measured objects (:func:`scanbound.measure.measure_objects`) into labels.

    labels = assign_labels(feats, config)          # {object id: label or None}

Assignment modes (``config.mode``)
----------------------------------
single    every object gets ``single_label``.
rules     ordered rules ``{"label", "conditions": [{"feature", "min", "max"}]}``; the first rule
          whose conditions all hold wins, other objects get ``fallback`` (None = no ROI).
bins      one feature split by ascending ``thresholds`` into classes, ``labels[i]`` per class.
clusters  k-means on the chosen ``features`` (standardised, optionally log-scaled); groups are
          numbered by increasing value of the first feature, ``labels[i]`` per group.

A label of None means "skip": no ROI is created for the object.  ``config`` is either a
:class:`LabelingConfig` or the plain dict saved by the labelling panel (same keys).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from .errors import ConfigError
from .measure import FEATURES, SCALE_FEATURES

LabelMode = Literal["single", "rules", "bins", "clusters"]
LABEL_MODES: tuple[str, ...] = ("single", "rules", "bins", "clusters")


@dataclass
class Condition:
    feature: str
    min: Optional[float] = None
    max: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "min": self.min, "max": self.max}


@dataclass
class Rule:
    label: Optional[str]
    conditions: list[Condition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "conditions": [c.to_dict() for c in self.conditions]}


@dataclass
class BinsConfig:
    feature: str = "area"
    thresholds: list[float] = field(default_factory=list)
    labels: list[Optional[str]] = field(default_factory=lambda: [None])
    log: bool = True


@dataclass
class ClustersConfig:
    features: list[str] = field(default_factory=lambda: ["area", "aspect_ratio"])
    k: int = 2
    labels: list[Optional[str]] = field(default_factory=lambda: [None, None])
    log: bool = True


def _opt_float(v: Any, where: str) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected a number or null, got {v!r}") from None


def _check_feature(key: str, where: str) -> str:
    if key not in FEATURES:
        raise ConfigError(f"{where}: unknown feature {key!r}; expected one of {', '.join(FEATURES)}")
    return key


@dataclass
class LabelingConfig:
    """How detected objects get their label.  Round-trips the panel's ``labeling`` dict."""

    mode: LabelMode = "single"
    single_label: Optional[str] = None
    rules: list[Rule] = field(default_factory=list)
    fallback: Optional[str] = None
    bins: BinsConfig = field(default_factory=BinsConfig)
    clusters: ClustersConfig = field(default_factory=ClustersConfig)

    @property
    def needs_features(self) -> bool:
        return self.mode != "single"

    @classmethod
    def single(cls, label: Optional[str]) -> "LabelingConfig":
        return cls(single_label=label, bins=BinsConfig(labels=[label]), clusters=ClustersConfig(labels=[label, label]))

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "LabelingConfig":
        d = dict(d or {})
        mode = d.get("mode", "single")
        if mode not in LABEL_MODES:
            raise ConfigError(f"labeling.mode: unknown mode {mode!r}; expected one of {', '.join(LABEL_MODES)}")
        rules = []
        for r_i, r in enumerate(d.get("rules") or []):
            conds = [Condition(_check_feature(str(c.get("feature", "area")), f"labeling.rules[{r_i}]"),
                               _opt_float(c.get("min"), f"labeling.rules[{r_i}].min"),
                               _opt_float(c.get("max"), f"labeling.rules[{r_i}].max"))
                     for c in (r.get("conditions") or [])]
            rules.append(Rule(r.get("label"), conds))
        b = dict(d.get("bins") or {})
        bins = BinsConfig(_check_feature(str(b.get("feature", "area")), "labeling.bins.feature"),
                          sorted(float(t) for t in (b.get("thresholds") or [])),
                          list(b.get("labels") or [None]), bool(b.get("log", True)))
        c = dict(d.get("clusters") or {})
        k = int(c.get("k", 2))
        if k < 1:
            raise ConfigError("labeling.clusters.k must be >= 1")
        clusters = ClustersConfig([_check_feature(str(f), "labeling.clusters.features")
                                   for f in (c.get("features") or ["area", "aspect_ratio"])],
                                  k, list(c.get("labels") or [None] * k), bool(c.get("log", True)))
        return cls(mode, d.get("single_label"), rules, d.get("fallback"), bins, clusters)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "single_label": self.single_label,
                "rules": [r.to_dict() for r in self.rules], "fallback": self.fallback,
                "bins": {"feature": self.bins.feature, "thresholds": list(self.bins.thresholds),
                         "labels": list(self.bins.labels), "log": self.bins.log},
                "clusters": {"features": list(self.clusters.features), "k": self.clusters.k,
                             "labels": list(self.clusters.labels), "log": self.clusters.log}}


def default_config(label: Optional[str] = None) -> dict[str, Any]:
    """The panel's default ``labeling`` dict (legacy helper)."""
    return LabelingConfig.single(label).to_dict()



def _passes(row, cond):
    v = row.get(cond.get("feature"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    lo, hi = cond.get("min"), cond.get("max")
    return (lo is None or v >= lo) and (hi is None or v <= hi)


def bin_index(values, thresholds):
    """Class index per value: 0 below thresholds[0], ..., len(thresholds) above the last."""
    return np.searchsorted(np.sort(np.asarray(thresholds, float)), np.asarray(values, float), side="right")


def suggest_thresholds(values, n_classes, method="quantiles", log=False):
    """n_classes - 1 ascending thresholds: equal-count quantiles or natural breaks (1D k-means)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n_classes = int(max(2, n_classes))
    if v.size < n_classes:
        return []
    t = np.log(np.clip(v, 1e-12, None)) if log else v
    if method == "natural breaks":
        centres = cluster_values(t[:, None], n_classes)[1].ravel()
        centres = np.sort(centres)
        cuts = (centres[:-1] + centres[1:]) / 2
    else:
        cuts = np.quantile(t, np.linspace(0, 1, n_classes + 1)[1:-1])
    cuts = np.exp(cuts) if log else cuts
    return [float(c) for c in np.unique(cuts)]


def cluster_values(x, k, seed=0):
    """k-means: (labels, centres).  x is (n, d), already scaled."""
    from sklearn.cluster import KMeans
    k = int(min(max(1, k), len(x)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    return km.labels_, km.cluster_centers_


def cluster_groups(features: pd.DataFrame, feature_keys, k, log=True):
    """
    Group index per object (Series aligned with ``features``), numbered by increasing mean of the
    first feature so the numbering is stable between runs.
    """
    keys = [f for f in feature_keys if f in features.columns]
    if not keys or features.empty:
        return pd.Series(np.zeros(len(features), int), index=features.index)
    x = features[keys].to_numpy(float)
    ok = np.isfinite(x).all(axis=1)
    groups = np.full(len(features), -1, int)
    if ok.sum() == 0:
        return pd.Series(groups, index=features.index)
    xs = x[ok].copy()
    for j, key in enumerate(keys):
        if log and key in SCALE_FEATURES:
            xs[:, j] = np.log(np.clip(xs[:, j], 1e-12, None))
    sd = xs.std(axis=0)
    xs = (xs - xs.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    raw, _ = cluster_values(xs, k)
    order = np.argsort([x[ok][raw == g, 0].mean() for g in range(raw.max() + 1)])
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    groups[ok] = rank[raw]
    return pd.Series(groups, index=features.index)


def assign_groups(features: pd.DataFrame, config: "LabelingConfig | Mapping[str, Any]"):
    """(group per object, list of labels per group) for the current mode."""
    if isinstance(config, LabelingConfig):
        config = config.to_dict()
    mode = config.get("mode", "single")
    idx = features.index
    if mode == "rules":
        rules = config.get("rules") or []
        groups = pd.Series(len(rules), index=idx)          # len(rules) = fallback group
        for oid, row in features.iterrows():
            r = row.to_dict()
            for g, rule in enumerate(rules):
                conds = rule.get("conditions") or []
                if all(_passes(r, c) for c in conds):
                    groups[oid] = g
                    break
        return groups, [r.get("label") for r in rules] + [config.get("fallback")]
    if mode == "bins":
        b = config.get("bins") or {}
        thresholds = sorted(b.get("thresholds") or [])
        labels = list(b.get("labels") or [])
        labels += [labels[-1] if labels else None] * (len(thresholds) + 1 - len(labels))
        col = b.get("feature", "area")
        values = features[col] if col in features else pd.Series(np.nan, index=idx)
        groups = pd.Series(bin_index(values.fillna(-np.inf), thresholds), index=idx)
        return groups, labels[:len(thresholds) + 1]
    if mode == "clusters":
        c = config.get("clusters") or {}
        k = int(c.get("k", 2))
        groups = cluster_groups(features, c.get("features") or [], k, bool(c.get("log", True)))
        labels = list(c.get("labels") or [])
        labels += [labels[-1] if labels else None] * (k - len(labels))
        return groups, labels[:k]
    return pd.Series(0, index=idx), [config.get("single_label")]


def assign_labels(features: pd.DataFrame, config: "LabelingConfig | Mapping[str, Any]") -> dict[int, Optional[str]]:
    """{object id: label or None}."""
    groups, labels = assign_groups(features, config)
    out = {}
    for oid, g in groups.items():
        out[oid] = labels[g] if 0 <= g < len(labels) else None
    return out
