"""
A project: a dataset folder plus everything done to it.

The project file (JSON, ``scanbound_project.json`` by default) stores *references* to the raw
data (relative paths, sizes, modification times, optional SHA-256), the per-image processing
chain, display settings and manual calibrations, the label set, every annotation with its
review status and origin, and a log of detector / model runs.  Pixel data are never copied into
it: images are regenerated from the raw files and the recorded chains.

    proj = Project.open_folder("D:/STM/2026-07-16")       # lists the data files
    img_id = proj.image_ids()[0]                           # "0001.sxm::Z"
    raw = proj.raw_image(img_id)                           # Image as read (metres)
    view = proj.processed_image(img_id)                    # with its processing chain
    proj.annotations.add(...)
    proj.save()                                            # scanbound_project.json in the folder
    proj = Project.load("D:/STM/2026-07-16/scanbound_project.json")
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .annotations import AnnotationStore, LabelSet
from .annotations.legacy import legacy_display_transform, load_csv, to_frame
from .display import DisplaySettings
from .errors import ScanBoundError
from .io import DataFile, list_data_files, open_data
from .model import Calibration, Image, file_digest
from .model.provenance import now_iso, software_version
from .processing import ProcessingChain

PROJECT_FORMAT = "scanbound_project"
PROJECT_VERSION = 1
DEFAULT_NAME = "scanbound_project.json"
# project files from older versions are upgraded when they are read: MIGRATIONS[v] turns a version v
# file (a dict) into version v + 1.  The first save over such a file keeps the original next to it as
# <name>.v<v>.bak.json, so an update never loses a project.  Newer files than PROJECT_VERSION are refused.
MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


class ProjectError(ScanBoundError):
    pass


def migrate_project_data(data: dict) -> dict:
    """A project file's contents in the current format (PROJECT_VERSION)."""
    version = data.get("version")
    while isinstance(version, int) and version < PROJECT_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise ProjectError(f"project file version {version!r} can no longer be read")
        data = dict(step(dict(data)), version=version + 1)
        version += 1
    return data


def _safe_name(text: str) -> str:
    """The old scan browser's file-name cleaning (``<stem>_<channel>.jpg``)."""
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_") or "channel"


@dataclass
class FileEntry:
    path: str                       # relative to the project root, '/' separated
    kind: str = ""                  # scan / spectrum / image ('' until opened)
    format: str = ""
    size: Optional[int] = None
    mtime: Optional[float] = None
    sha256: Optional[str] = None
    error: str = ""                 # why it could not be opened (shown in the browser)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, d: dict) -> "FileEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ImageEntry:
    id: str
    file: str
    channel: str
    chain: ProcessingChain = field(default_factory=ProcessingChain)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    calibration: Optional[dict] = None   # manual override: dx, dy (m), source, detail, rel_uncertainty
    notes: str = ""
    reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "file": self.file, "channel": self.channel,
                             "chain": self.chain.to_dict(), "display": self.display.to_dict()}
        if self.calibration:
            d["calibration"] = self.calibration
        if self.notes:
            d["notes"] = self.notes
        if self.reviewed:
            d["reviewed"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ImageEntry":
        return cls(str(d["id"]), str(d["file"]), str(d["channel"]), ProcessingChain.from_dict(d.get("chain")),
                   DisplaySettings.from_dict(d.get("display")), d.get("calibration"), str(d.get("notes", "")),
                   bool(d.get("reviewed", False)))


@dataclass
class RunRecord:
    """A detector / model run, training or evaluation, logged so proposals can be traced back."""

    id: str
    kind: str                       # "detect" / "predict" / "train" / "evaluate"
    name: str
    config: dict = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    created: str = field(default_factory=now_iso)
    software: str = field(default_factory=software_version)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class _LRU(OrderedDict):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.lock = threading.RLock()

    def get_or(self, key: Any, make: Callable[[], Any]) -> Any:
        with self.lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        value = make()
        with self.lock:
            self[key] = value
            while len(self) > self.size:
                self.popitem(last=False)
        return value


class Project:
    def __init__(self, root: str, path: Optional[str] = None):
        self.root = os.path.abspath(root)
        self.path = path
        self.files: dict[str, FileEntry] = {}
        self.images: dict[str, ImageEntry] = {}
        self.labels = LabelSet()
        self.annotations = AnnotationStore()
        self.runs: list[RunRecord] = []
        self.settings: dict[str, Any] = {}
        self.surface: dict[str, Any] = {}     # surface analysis state (lattice fits, models ...), see scanbound.surface.store
        self.dirty = False
        self.loaded_version = PROJECT_VERSION   # the file's format version when it was read
        self._files_cache = _LRU(24)
        self._image_cache = _LRU(48)
        self.annotations.subscribe(lambda ev: self._mark_dirty() if ev.source != "load" else None)

    def _mark_dirty(self) -> None:
        self.dirty = True

    # ================================================================== files
    @classmethod
    def open_folder(cls, folder: str, recursive: bool = False, **scan_options: Any) -> "Project":
        """A new project on a folder (or the project saved in it, if there is one).  ``scan_options``
        (limit, progress, cancelled) go to :func:`list_data_files`."""
        saved = os.path.join(folder, DEFAULT_NAME)
        if os.path.isfile(saved):
            p = cls.load(saved)
            p.scan(recursive, **scan_options)
            return p
        p = cls(folder)
        p.scan(recursive, **scan_options)
        p.dirty = False
        return p

    def rel(self, path: str) -> str:
        r = os.path.relpath(os.path.abspath(path), self.root)
        return r.replace(os.sep, "/")

    def abs(self, rel: str) -> str:
        if os.sep == "/":                      # a path written on Windows ("sub\\scan.sxm"), opened on macOS / Linux
            rel = rel.replace("\\", "/")
        return os.path.normpath(os.path.join(self.root, rel))

    def scan(self, recursive: bool = False, **scan_options: Any) -> list[str]:
        """Add data files of the root folder that are not listed yet; returns the new ones."""
        new = []
        skip = {DEFAULT_NAME}
        for p in list_data_files(self.root, recursive, **scan_options):
            r = self.rel(p)
            if os.path.basename(r) in skip or r in self.files:
                continue
            st = os.stat(p)
            self.files[r] = FileEntry(r, size=st.st_size, mtime=st.st_mtime)
            new.append(r)
        return new

    def add_files(self, paths: Iterable[str]) -> list[str]:
        out = []
        for p in paths:
            r = self.rel(p)
            if r not in self.files:
                st = os.stat(p)
                self.files[r] = FileEntry(r, size=st.st_size, mtime=st.st_mtime)
                out.append(r)
        if out:
            self._mark_dirty()
        return out

    def data_file(self, rel: str) -> DataFile:
        """Open (cached) a file of the project.  Records its kind / format or the error."""
        entry = self.files.get(rel)
        if entry is None:
            raise ProjectError(f"{rel} is not part of the project")

        def make() -> DataFile:
            try:
                f = open_data(self.abs(rel))
            except Exception as exc:
                entry.error = str(exc)
                raise
            entry.kind, entry.format, entry.error = f.kind, f.format, ""
            return f
        return self._files_cache.get_or(rel, make)

    def changed_files(self) -> list[str]:
        """Files whose size / modification time differ from when they were added (or are gone)."""
        out = []
        for r, e in self.files.items():
            p = self.abs(r)
            try:
                st = os.stat(p)
            except OSError:
                out.append(r)
                continue
            if e.size is not None and (st.st_size != e.size or (e.mtime and abs(st.st_mtime - e.mtime) > 1e-3)):
                out.append(r)
        return out

    def hash_file(self, rel: str) -> str:
        e = self.files[rel]
        e.sha256 = file_digest(self.abs(rel))
        self._mark_dirty()
        return e.sha256

    # ================================================================== images
    @staticmethod
    def make_image_id(rel: str, channel: str) -> str:
        return f"{rel}::{channel}"

    @staticmethod
    def split_image_id(image_id: str) -> tuple[str, str]:
        rel, _, ch = image_id.rpartition("::")
        return rel, ch

    def image_ids(self, kinds: tuple[str, ...] = ("scan", "image")) -> list[str]:
        """One id per image file (its current channel); unopened files use their default channel."""
        out = []
        for r, e in self.files.items():
            if e.kind and e.kind not in kinds:
                continue
            ids = [i for i, im in self.images.items() if im.file == r]
            if ids:
                out.extend(ids)
            elif not r.lower().endswith(".dat"):
                out.append(self.make_image_id(r, ""))
        return out

    def entry(self, image_id: str, create: bool = True) -> ImageEntry:
        """The settings of an image; ``"file::"`` (empty channel) resolves to the default channel."""
        if image_id in self.images:
            return self.images[image_id]
        rel, ch = self.split_image_id(image_id)
        if not ch:
            ch = self.data_file(rel).default_channel
            image_id = self.make_image_id(rel, ch)
            if image_id in self.images:
                return self.images[image_id]
        if not create:
            raise KeyError(image_id)
        default_chain = ProcessingChain.from_dict(self.settings.get("default_chain")) \
            if self.settings.get("default_chain") else ProcessingChain()
        e = ImageEntry(image_id, rel, ch, default_chain)
        self.images[image_id] = e
        return e

    def raw_image(self, image_id: str) -> Image:
        e = self.entry(image_id)
        img = self.data_file(e.file).image(e.channel)
        img = Image(img.data, x=img.x, y=img.y, quantity=img.quantity, unit=img.unit, id=e.id, source=img.source,
                    metadata=img.metadata, history=img.history, frame=img.frame, raw_dtype=img.raw_dtype)
        if e.calibration:
            c = e.calibration
            img = img.with_calibration(c.get("dx"), c.get("dy"), unit=c.get("unit", "m"),
                                       calibration=Calibration(c.get("source", "manual"), c.get("detail", ""),
                                                               float(c.get("rel_uncertainty", 0.0))))
        return img

    def _key(self, e: ImageEntry) -> str:
        return json.dumps([e.id, e.chain.to_dict(), e.calibration], sort_keys=True)

    def processed_image(self, image_id: str) -> Image:
        e = self.entry(image_id)
        return self._image_cache.get_or(self._key(e), lambda: e.chain.apply(self.raw_image(e.id), strict=False))

    def set_chain(self, image_id: str, chain: ProcessingChain) -> None:
        self.entry(image_id).chain = chain
        self._mark_dirty()

    def set_display(self, image_id: str, display: DisplaySettings) -> None:
        self.entry(image_id).display = display
        self._mark_dirty()

    def set_calibration(self, image_id: str, dx: Optional[float], dy: Optional[float] = None,
                        rel_uncertainty: float = 0.0, detail: str = "entered manually") -> None:
        """Manual pixel size in metres (``dx=None`` restores the file's calibration)."""
        e = self.entry(image_id)
        e.calibration = None if dx is None else {"dx": float(dx), "dy": float(dy if dy is not None else dx),
                                                  "unit": "m", "source": "manual", "detail": detail,
                                                  "rel_uncertainty": float(rel_uncertainty)}
        self._mark_dirty()

    def file_name(self, image_id: str) -> str:
        return os.path.basename(self.split_image_id(image_id)[0])

    def spectra_files(self) -> list[str]:
        return [r for r, e in self.files.items() if e.kind == "spectrum" or (not e.kind and r.lower().endswith(".dat"))]

    # ================================================================== legacy annotations
    def legacy_resolver(self, csv_dir: Optional[str] = None) -> Callable[[str], Optional[tuple[str, Optional[np.ndarray]]]]:
        """Maps file names of an old annotations.csv to images of this project (see annotations.legacy)."""
        by_name: dict[str, str] = {}
        stems: dict[str, str] = {}
        for r in self.files:
            by_name.setdefault(os.path.basename(r), r)
            stems.setdefault(os.path.splitext(os.path.basename(r))[0], r)
        legacy_sizes = self._legacy_sizes(csv_dir)

        # the old app ran on Windows: names may carry a Windows folder and differ in case from the files
        by_lower = {k.lower(): v for k, v in reversed(list(by_name.items()))}
        stems_lower = {k.lower(): k for k in reversed(list(stems))}

        def resolve(name: str) -> Optional[tuple[str, Optional[np.ndarray]]]:
            name = name.replace("\\", "/").rsplit("/", 1)[-1]
            if name not in by_name and name.lower() in by_lower:
                name = os.path.basename(by_lower[name.lower()])
            stem = os.path.splitext(name)[0]
            if not any(stem.startswith(s + "_") for s in stems):
                low = stem.lower()
                for s_low, s in sorted(stems_lower.items(), key=lambda kv: len(kv[0]), reverse=True):
                    if low.startswith(s_low + "_"):
                        stem = s + stem[len(s):]
                        break
            # 1) the annotated JPG is itself part of the project
            if name in by_name:
                rel = by_name[name]
                try:
                    return self.make_image_id(rel, self.data_file(rel).default_channel), None
                except Exception:  # noqa: BLE001
                    return None
            # 2) "<scan stem>_<channel>.jpg" exported from a scan of the project
            for s in sorted(stems, key=len, reverse=True):
                if not stem.startswith(s + "_"):
                    continue
                rel = stems[s]
                try:
                    f = self.data_file(rel)
                except Exception:  # noqa: BLE001
                    continue
                want = stem[len(s) + 1:]
                for ch in f.channel_keys():
                    if _safe_name(ch) == want:
                        img = f.image(ch)
                        size = legacy_sizes.get(name) or (img.shape[1], img.shape[0])
                        m = legacy_display_transform(img.shape, np.all(img.valid, axis=1) if img.valid.size else
                                                     np.zeros(0, bool), img.pixel_size, size)
                        if m is not None and np.allclose(m, np.eye(3)):
                            m = None
                        return self.make_image_id(rel, ch), m
            return None
        return resolve

    def _legacy_sizes(self, csv_dir: Optional[str]) -> dict[str, tuple[int, int]]:
        """(width, height) of old JPG exports, from a metadata.csv or the JPG files themselves."""
        import csv as _csv
        out: dict[str, tuple[int, int]] = {}
        dirs = [d for d in {csv_dir, os.path.join(self.root, "ScanBound_image_output"),
                            os.path.join(self.root, "saved_output")} if d and os.path.isdir(d)]
        for d in dirs:
            meta = os.path.join(d, "metadata.csv")
            if os.path.isfile(meta):
                try:
                    with open(meta, newline="", encoding="utf-8-sig") as fh:
                        for row in _csv.DictReader(fh):
                            dims = re.findall(r"\d+", row.get("Dimensions", ""))
                            if len(dims) >= 2 and row.get("Filename"):
                                # the old export wrote "width x height" of the QImage
                                out.setdefault(row["Filename"], (int(dims[0]), int(dims[1])))
                except (OSError, ValueError):
                    pass
            for name in os.listdir(d):
                if name.lower().endswith((".jpg", ".jpeg", ".png")) and name not in out:
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(os.path.join(d, name)) as im:
                            out[name] = im.size
                    except Exception:  # noqa: BLE001
                        pass
        return out

    def import_legacy_annotations(self, csv_path: str, status: str = "manual") -> tuple[int, list[str]]:
        anns, problems = load_csv(csv_path, self.legacy_resolver(os.path.dirname(csv_path)), status)
        for a in anns:
            self.entry(a.image_id)
        self.labels.merge(a.label for a in anns)
        self.annotations.add(anns, f"Import {len(anns)} annotation(s) from {os.path.basename(csv_path)}")
        return len(anns), problems

    def export_legacy_annotations(self, csv_path: str, image_ids: Optional[Iterable[str]] = None) -> int:
        ids = set(image_ids) if image_ids is not None else None
        items = [a for a in self.annotations if ids is None or a.image_id in ids]
        df = to_frame(items, self.file_name)
        df.to_csv(csv_path, index=False)
        return len(df)

    # ================================================================== runs
    def log_run(self, kind: str, name: str, config: dict, images: list[str], summary: Optional[dict] = None
                ) -> RunRecord:
        from .annotations import new_id
        run = RunRecord(new_id(), kind, name, config, list(images), dict(summary or {}))
        self.runs.append(run)
        self._mark_dirty()
        return run

    def run(self, run_id: str) -> Optional[RunRecord]:
        return next((r for r in self.runs if r.id == run_id), None)

    # ================================================================== persistence
    def to_dict(self) -> dict[str, Any]:
        return {"format": PROJECT_FORMAT, "version": PROJECT_VERSION, "software": software_version(),
                "saved": now_iso(), "root": ".", "files": [e.to_dict() for e in self.files.values()],
                "images": [e.to_dict() for e in self.images.values()], "labels": self.labels.to_dict(),
                "annotations": self.annotations.to_list(), "runs": [r.to_dict() for r in self.runs],
                "settings": self.settings, **({"surface": self.surface} if self.surface else {})}

    def save(self, path: Optional[str] = None) -> str:
        """Write the project file atomically (temporary file + rename)."""
        path = os.path.abspath(path or self.path or os.path.join(self.root, DEFAULT_NAME))
        new_root = os.path.dirname(path)
        if os.path.normcase(new_root) != os.path.normcase(self.root):
            self._rebase(new_root)
        data = self.to_dict()
        self._keep_original(path)
        fd, tmp = tempfile.mkstemp(prefix=".scanbound_", suffix=".json", dir=new_root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1, allow_nan=False, default=_json_default)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self.path = path
        self.dirty = False
        return path

    def _keep_original(self, path: str) -> None:
        """Before the first save over a file in an older format, copy it to <name>.v<version>.bak.json."""
        if self.loaded_version >= PROJECT_VERSION or not os.path.isfile(path):
            return
        backup = f"{os.path.splitext(path)[0]}.v{self.loaded_version}.bak.json"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        self.loaded_version = PROJECT_VERSION

    def _rebase(self, new_root: str) -> None:
        """Keep file references valid when the project file is saved in another folder."""
        mapping = {r: os.path.relpath(self.abs(r), new_root).replace(os.sep, "/") for r in self.files}
        self.root = os.path.abspath(new_root)
        self.files = {mapping[r]: FileEntry(**{**e.__dict__, "path": mapping[r]}) for r, e in self.files.items()}
        id_map = {}
        new_images = {}
        for iid, e in self.images.items():
            new_id = self.make_image_id(mapping.get(e.file, e.file), e.channel)
            id_map[iid] = new_id
            e.id, e.file = new_id, mapping.get(e.file, e.file)
            new_images[new_id] = e
        self.images = new_images
        if self.surface:
            from .surface.store import remap_image_ids
            remap_image_ids(self.surface, id_map)
        self._files_cache.clear()
        self._image_cache.clear()
        if id_map:
            items = [a if a.image_id not in id_map else a.evolve(image_id=id_map[a.image_id], modified=a.modified)
                     for a in self.annotations]
            self.annotations.load(items, "Rebase project")

    @classmethod
    def from_dict(cls, data: dict, root: str, path: Optional[str] = None) -> "Project":
        if not isinstance(data, dict) or data.get("format") != PROJECT_FORMAT:
            raise ProjectError("not a ScanBound project file")
        version = data.get("version")
        if not isinstance(version, int) or version > PROJECT_VERSION:
            raise ProjectError(f"project file version {version!r} is newer than this ScanBound "
                               f"(reads up to {PROJECT_VERSION})")
        data = migrate_project_data(data)
        p = cls(root, path)
        p.loaded_version = version
        p.files = {e["path"]: FileEntry.from_dict(e) for e in data.get("files", [])}
        p.images = {e["id"]: ImageEntry.from_dict(e) for e in data.get("images", [])}
        p.labels = LabelSet.from_dict(data.get("labels"))
        p.annotations.load(data.get("annotations", []))
        p.runs = [RunRecord.from_dict(r) for r in data.get("runs", [])]
        p.settings = dict(data.get("settings") or {})
        p.surface = dict(data.get("surface") or {})
        p.dirty = False
        return p

    @classmethod
    def load(cls, path: str) -> "Project":
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ProjectError(f"{path}: not valid JSON ({exc})") from exc
        return cls.from_dict(data, os.path.dirname(os.path.abspath(path)), os.path.abspath(path))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
