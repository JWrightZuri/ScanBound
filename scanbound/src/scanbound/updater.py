"""
Updates from GitHub Releases: is a newer ScanBound out, what changed, and installing it.

A release is a GitHub release on :data:`UPDATE_REPO` whose tag is the version (``v0.2.0``), whose
notes say what changed, and which carries the Windows installer (``ScanBound-Setup-0.2.0.exe``) as
an asset.  ``packaging/make_release.ps1`` makes one.  Drafts and pre-releases are never offered.

How an update is installed depends on how ScanBound was installed (:func:`detect_install`):

``installer``  the Windows installer: the new installer is downloaded, checked against the size and
               SHA-256 GitHub reports, and run over the old install once the app has closed
``source``     the scanbound folder with ``run_scanbound.bat`` / ``.sh``: the release's source is
               downloaded and staged, and :mod:`scanbound._update_apply` swaps it in after the app has
               closed, keeping a backup and putting the old files back if anything fails
``git``        a git checkout (a developer copy): only told about the release, update with ``git pull``
``other``      installed with pip somewhere else: only told about the release

Updating only ever replaces program files.  Projects (``scanbound_project.json`` next to the data),
the data files, ``~/.scanbound`` (settings, tutorial copy, crash log) and Qt's settings are never
touched; project files are read by newer versions and upgraded in place with a backup of the old file
(see :mod:`scanbound.project`).

Qt-free; the dialogs are in :mod:`scanbound.app.updates`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import __version__
from .errors import ScanBoundError

UPDATE_REPO = os.environ.get("SCANBOUND_UPDATE_REPO", "JWrightZuri/ScanBound")
API = os.environ.get("SCANBOUND_UPDATE_API", "https://api.github.com")      # overridable for testing
HOME = os.path.join(os.path.expanduser("~"), ".scanbound")
UPDATES_DIR = os.path.join(HOME, "updates")
STATE_FILE = os.path.join(HOME, "updater.json")
INSTALLER_PATTERN = re.compile(r"^ScanBound-Setup-.*\.exe$", re.IGNORECASE)
USER_AGENT = f"ScanBound/{__version__} (+https://github.com/{UPDATE_REPO})"
KEEP_BACKUPS = 2                    # backups of replaced program files kept in UPDATES_DIR


class UpdateError(ScanBoundError):
    pass


# ====================================================================== versions
_VERSION = re.compile(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-.]?(a|b|rc|dev)\.?(\d*))?$", re.IGNORECASE)
_PRE_RANK = {"dev": 0, "a": 1, "b": 2, "rc": 3}


def parse_version(text: str) -> Optional[tuple]:
    """``"v1.2.3"`` -> a tuple that sorts like the versions do (pre-releases before the release), or None."""
    m = _VERSION.match(str(text).strip())
    if not m:
        return None
    major, minor, patch, pre, num = m.groups()
    rank = (_PRE_RANK[pre.lower()], int(num or 0)) if pre else (9, 0)
    return (int(major), int(minor or 0), int(patch or 0)) + rank


def is_newer(candidate: str, current: str = __version__) -> bool:
    a, b = parse_version(candidate), parse_version(current)
    return a is not None and b is not None and a > b


# ====================================================================== releases
@dataclass
class Asset:
    name: str
    url: str
    size: int = 0
    sha256: str = ""                 # GitHub's digest of the uploaded file, when it reports one


@dataclass
class Release:
    version: str                     # "0.2.0"
    tag: str                         # "v0.2.0"
    name: str = ""
    notes: str = ""                  # Markdown, as written on GitHub
    page: str = ""                   # the release's web page
    published: str = ""
    source_zip: str = ""
    assets: list[Asset] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict) -> "Release":
        tag = str(data.get("tag_name") or "")
        assets = []
        for a in data.get("assets") or []:
            digest = str(a.get("digest") or "")
            assets.append(Asset(str(a.get("name", "")), str(a.get("browser_download_url", "")), int(a.get("size") or 0),
                                digest[7:].lower() if digest.startswith("sha256:") else ""))
        return cls(tag[1:] if tag[:1] in "vV" else tag, tag, str(data.get("name") or tag),
                   str(data.get("body") or "").strip(), str(data.get("html_url") or ""),
                   str(data.get("published_at") or ""), str(data.get("zipball_url") or ""), assets)

    @property
    def installer(self) -> Optional[Asset]:
        return next((a for a in self.assets if INSTALLER_PATTERN.match(a.name)), None)

    def change_list(self, limit: int = 12) -> list[str]:
        """The notes' bullet points (the quick list the update prompt shows), or their first lines."""
        items: list[str] = []
        bullet = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
        in_item = False
        for raw in self.notes.splitlines():
            if bullet.match(raw):
                items.append(bullet.sub("", raw).strip())
                in_item = True
            elif in_item and raw[:1].isspace() and raw.strip():
                items[-1] += " " + raw.strip()           # a wrapped bullet point
            else:
                in_item = False
        if not items:
            items = [ln.strip() for ln in self.notes.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        return items[:limit] + ([f"… and {len(items) - limit} more"] if len(items) > limit else [])


def _get(url: str, timeout: float, accept: str = "application/vnd.github+json"):  # noqa: ANN202
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT,
                                               "X-GitHub-Api-Version": "2022-11-28"})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - fixed https URLs


def latest_release(repo: str = UPDATE_REPO, timeout: float = 8.0) -> Optional[Release]:
    """The newest published, non-pre-release release, or None when the repository has none."""
    try:
        with _get(f"{API}/repos/{repo}/releases/latest", timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise UpdateError(f"GitHub answered {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise UpdateError(f"could not reach GitHub: {getattr(exc, 'reason', exc)}") from exc
    release = Release.from_api(data)
    return release if parse_version(release.version) else None


def check_for_update(current: str = __version__, repo: str = UPDATE_REPO, timeout: float = 8.0) -> Optional[Release]:
    """The latest release if it is newer than ``current``, else None.  Raises UpdateError when offline."""
    release = latest_release(repo, timeout)
    return release if release is not None and is_newer(release.version, current) else None


# ====================================================================== state (~/.scanbound/updater.json)
def load_state() -> dict[str, Any]:
    try:
        with open(STATE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(**changes: Any) -> dict[str, Any]:
    """Merge ``changes`` into the state file (None removes a key), written atomically."""
    state = load_state()
    for k, v in changes.items():
        if v is None:
            state.pop(k, None)
        else:
            state[k] = v
    os.makedirs(HOME, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".updater_", suffix=".json", dir=HOME)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, STATE_FILE)
    return state


def check_on_startup() -> bool:
    return bool(load_state().get("check_on_startup", True)) and not os.environ.get("SCANBOUND_NO_UPDATE_CHECK")


def should_prompt(release: Release) -> bool:
    """False for a version the user chose to skip."""
    return load_state().get("skipped_version") != release.version


# ====================================================================== how this copy was installed
@dataclass
class Install:
    kind: str                        # installer / source / git / other
    root: str                        # install folder (installer) or the scanbound folder (source, git)
    python: str                      # a console Python for helper processes

    @property
    def can_update(self) -> bool:
        return self.kind in ("installer", "source") and (self.kind != "installer" or sys.platform == "win32")

    def explain(self) -> str:
        if self.kind == "git":
            return (f"This copy of ScanBound is a git checkout ({self.root}), so it is updated with git: "
                    "run  git pull  in that folder.")
        if self.kind == "other":
            return "This copy of ScanBound was installed with pip, so update it with pip, or download the release."
        if self.kind == "installer" and sys.platform != "win32":
            return "Download the new version from the release page."
        return ""


def _console_python() -> str:
    head, name = os.path.split(sys.executable)
    if name.lower() == "pythonw.exe" and os.path.exists(os.path.join(head, "python.exe")):
        return os.path.join(head, "python.exe")
    return sys.executable


def detect_install(package_dir: Optional[str] = None) -> Install:
    package_dir = os.path.abspath(package_dir or os.path.dirname(__file__))
    python = _console_python()
    # installer:  <app>\ScanBound.exe  and  <app>\runtime\  (the Python this runs in)
    app = os.path.dirname(os.path.abspath(sys.prefix))
    if os.path.basename(os.path.normpath(sys.prefix)).lower() == "runtime" and \
            os.path.isfile(os.path.join(app, "ScanBound.exe")):
        return Install("installer", app, python)
    # source: <root>/src/scanbound with <root>/pyproject.toml
    src = os.path.dirname(package_dir)
    root = os.path.dirname(src)
    if os.path.basename(src) == "src" and os.path.isfile(os.path.join(root, "pyproject.toml")):
        for d in (root, os.path.dirname(root)):
            if os.path.exists(os.path.join(d, ".git")):
                return Install("git", d, python)
        return Install("source", root, python)
    return Install("other", package_dir, python)


# ====================================================================== downloading
Progress = Callable[[int, int], None]         # (bytes so far, total or 0)


class Cancelled(UpdateError):
    pass


def download(url: str, dest: str, size: int = 0, sha256: str = "", progress: Optional[Progress] = None,
             cancelled: Callable[[], bool] = lambda: False, timeout: float = 30.0) -> str:
    """Download to ``dest`` (through ``dest.part``, renamed only when complete and verified)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    digest = hashlib.sha256()
    done = 0
    try:
        with _get(url, timeout, accept="application/octet-stream") as r, open(part, "wb") as fh:
            total = size or int(r.headers.get("Content-Length") or 0)
            while True:
                if cancelled():
                    raise Cancelled("download cancelled")
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
        if size and done != size:
            raise UpdateError(f"the download is incomplete ({done:,} of {size:,} bytes)")
        if sha256 and digest.hexdigest() != sha256.lower():
            raise UpdateError("the download does not match its SHA-256 checksum")
        os.replace(part, dest)
    except (urllib.error.URLError, OSError) as exc:
        raise UpdateError(f"download failed: {getattr(exc, 'reason', exc)}") from exc
    finally:
        if os.path.exists(part):
            os.unlink(part)
    return dest


# ====================================================================== installer updates (Windows)
def download_installer(release: Release, progress: Optional[Progress] = None,
                       cancelled: Callable[[], bool] = lambda: False) -> str:
    asset = release.installer
    if asset is None:
        raise UpdateError(f"release {release.tag} has no Windows installer")
    return download(asset.url, os.path.join(UPDATES_DIR, asset.name), asset.size, asset.sha256, progress, cancelled)


def launch_installer(setup: str, release: Release) -> None:
    """Start the downloaded installer; the caller then quits.  It closes ScanBound if it is still running,
    installs over the old version in the same folder (the projects and settings are elsewhere and are
    not touched) and starts the new version."""
    _remember_yolo()
    save_state(installing=release.version)
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([setup, "/SILENT", "/SP-", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS",
                      "/RELAUNCH=1"], close_fds=True, creationflags=flags)


def _remember_yolo() -> None:
    """The installer replaces the bundled Python, so YOLO (installed into it later) goes; offer it again."""
    import importlib.util
    if importlib.util.find_spec("ultralytics") is not None:
        save_state(reinstall_yolo=True)


# ====================================================================== source updates
def _safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    dest = os.path.abspath(dest)
    for info in zf.infolist():
        target = os.path.abspath(os.path.join(dest, info.filename))
        if os.path.commonpath([dest, target]) != dest:
            raise UpdateError(f"unsafe path in the download: {info.filename}")
    zf.extractall(dest)


def _version_in(package_dir: str) -> str:
    with open(os.path.join(package_dir, "__init__.py"), encoding="utf-8") as fh:
        m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', fh.read(), re.MULTILINE)
    return m.group(1) if m else ""


def find_app_folder(extracted: str) -> str:
    """The folder in a source download that holds ``pyproject.toml`` and ``src/scanbound``."""
    for here, dirs, files in os.walk(extracted):
        if "pyproject.toml" in files and os.path.isfile(os.path.join(here, "src", "scanbound", "__init__.py")):
            return here
        if here.count(os.sep) - extracted.count(os.sep) >= 3:
            dirs[:] = []
    raise UpdateError("the download does not contain ScanBound")


def stage_source_update(release: Release, progress: Optional[Progress] = None,
                        cancelled: Callable[[], bool] = lambda: False) -> str:
    """Download and unpack the release's source; returns the staged scanbound folder, checked to be that version."""
    if not release.source_zip:
        raise UpdateError(f"release {release.tag} has no source download")
    stage = os.path.join(UPDATES_DIR, f"staged-{release.version}")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    archive = download(release.source_zip, os.path.join(UPDATES_DIR, f"scanbound-{release.version}-source.zip"),
                       progress=progress, cancelled=cancelled, timeout=60)
    try:
        with zipfile.ZipFile(archive) as zf:
            if zf.testzip() is not None:
                raise UpdateError("the download is damaged")
            _safe_extract(zf, stage)
    except zipfile.BadZipFile as exc:
        raise UpdateError("the download is not a zip file") from exc
    finally:
        os.unlink(archive)
    folder = find_app_folder(stage)
    found = _version_in(os.path.join(folder, "src", "scanbound"))
    if parse_version(found) != parse_version(release.version):
        raise UpdateError(f"the download is version {found or '?'}, expected {release.version}")
    return folder


def launch_source_update(install: Install, staged: str, release: Release, relaunch: bool = True) -> None:
    """Start :mod:`scanbound._update_apply` in its own process; the caller then quits.  The helper is copied
    out of the program folder first because it replaces that folder."""
    os.makedirs(UPDATES_DIR, exist_ok=True)
    helper = os.path.join(UPDATES_DIR, "apply_update.py")
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "_update_apply.py"), helper)
    _prune_backups()
    save_state(installing=release.version)
    args = [install.python, "-I", helper, "--pid", str(os.getpid()), "--target", install.root, "--staged", staged,
            "--backups", UPDATES_DIR, "--from-version", __version__, "--to-version", release.version,
            "--log", os.path.join(UPDATES_DIR, "update.log"), "--state", STATE_FILE,
            "--cleanup", os.path.join(UPDATES_DIR, f"staged-{release.version}")]
    if relaunch:
        args.append("--relaunch")
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)       # Windows: the helper shows its progress
    subprocess.Popen(args, cwd=UPDATES_DIR, close_fds=True, creationflags=flags,
                     start_new_session=sys.platform != "win32")


def _prune_backups() -> None:
    if not os.path.isdir(UPDATES_DIR):
        return
    old = sorted((d for d in os.listdir(UPDATES_DIR) if d.startswith("backup-")),
                 key=lambda d: os.path.getmtime(os.path.join(UPDATES_DIR, d)))
    for d in old[:max(0, len(old) - (KEEP_BACKUPS - 1))]:
        shutil.rmtree(os.path.join(UPDATES_DIR, d), ignore_errors=True)


# ====================================================================== after an update
def finish_pending() -> dict[str, Any]:
    """Called at start-up: what the last update did.  Returns ``{"updated": version}`` when this run is
    the version an update installed, ``{"failed": message}`` when it did not get there, and
    ``"reinstall_yolo": True`` when YOLO was installed before the update and is gone now."""
    import importlib.util
    state = load_state()
    out: dict[str, Any] = {}
    target = state.get("installing")
    if target:
        if parse_version(target) == parse_version(__version__):
            out["updated"] = target
        elif state.get("update_error"):
            out["failed"] = str(state["update_error"])
        save_state(installing=None, update_error=None)
    if state.get("reinstall_yolo"):
        if importlib.util.find_spec("ultralytics") is None:
            out["reinstall_yolo"] = True
        else:
            save_state(reinstall_yolo=None)
    return out
