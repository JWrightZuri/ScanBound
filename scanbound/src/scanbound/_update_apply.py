"""
Puts a staged source update in place, run by :func:`scanbound.updater.launch_source_update` in its own
process once the app has closed.  Standard library only: it is copied out of the program folder and
run from there, because it replaces that folder.

    python -I apply_update.py --pid 1234 --target <scanbound folder> --staged <new scanbound folder>
                              --backups ~/.scanbound/updates --from-version 0.1.0 --to-version 0.2.0 [--relaunch]

What it replaces, and nothing else:

* ``src/scanbound``: the whole package (so modules removed in the new version go too)
* the files at the top of the new scanbound folder (``pyproject.toml``, ``README.md``, launchers ...)
* the files in its other folders (``tests``, ``packaging``), one by one; files only the old copy has
  stay, so build caches such as ``packaging/windows/build`` survive

``.venv``, ``dist`` and everything outside the scanbound folder (projects, data, ``~/.scanbound``)
are never touched.  Every replaced file is moved to ``backup-<old version>-<time>`` first; if any
step fails, or the new version does not import, the old files are put back.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

PACKAGE = os.path.join("src", "scanbound")
NEVER = {".venv", "venv", "dist", "build", ".git", "__pycache__"}
EXTRAS = "[gui,clusters,formats]"            # what run_scanbound.bat / .sh install


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


LOG = ""


def process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)       # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x00000102  # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_exit(pid: int, timeout: float = 120.0) -> bool:
    end = time.monotonic() + timeout
    while process_alive(pid):
        if time.monotonic() > end:
            return False
        time.sleep(0.3)
    time.sleep(0.5)          # let Windows release the files
    return True


def _without_version(text: str) -> str:
    return re.sub(r'(?m)^version\s*=.*$', "", text).strip()


def deps_changed(old: str, new: str) -> bool:
    try:
        with open(old, encoding="utf-8") as a, open(new, encoding="utf-8") as b:
            return _without_version(a.read()) != _without_version(b.read())
    except OSError:
        return True


class Journal:
    """What was moved where, to undo it."""

    def __init__(self, backup: str, target: str):
        self.backup, self.target = backup, target
        self.moved: list[tuple[str, str]] = []      # (path in target, its backup)
        self.created: list[str] = []                # paths that did not exist before

    def set_aside(self, rel: str) -> None:
        path = os.path.join(self.target, rel)
        if os.path.lexists(path):
            dest = os.path.join(self.backup, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(path, dest)
            self.moved.append((path, dest))
        else:
            self.created.append(path)

    def rollback(self) -> None:
        for path in reversed(self.created):
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.lexists(path):
                os.unlink(path)
        for path, dest in reversed(self.moved):
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.lexists(path):
                os.unlink(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.move(dest, path)


def replace(staged: str, target: str, journal: Journal) -> int:
    """Copy the staged release over the target; returns the number of files written."""
    n = 0
    journal.set_aside(PACKAGE)
    shutil.copytree(os.path.join(staged, PACKAGE), os.path.join(target, PACKAGE),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    n += sum(len(f) for _, _, f in os.walk(os.path.join(target, PACKAGE)))
    for here, dirs, files in os.walk(staged):
        rel_dir = os.path.relpath(here, staged)
        dirs[:] = [d for d in dirs if d not in NEVER and os.path.normpath(os.path.join(rel_dir, d)) != PACKAGE]
        for name in files:
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if rel.startswith(PACKAGE + os.sep) or name.endswith(".pyc"):
                continue
            dest = os.path.join(target, rel)
            if os.path.isdir(dest):
                raise RuntimeError(f"{dest} is a folder in this copy and a file in the update")
            journal.set_aside(rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(os.path.join(here, name), dest)
            n += 1
    return n


def check_import(python: str, target: str, version: str) -> None:
    code = ("import scanbound, scanbound.project, scanbound.app.mainwindow; "
            "print(scanbound.__version__); print(scanbound.__file__)")
    env = dict(os.environ, PYTHONNOUSERSITE="1", QT_QPA_PLATFORM="offscreen")
    env.pop("PYTHONPATH", None)
    out = subprocess.run([python, "-c", code], cwd=target, env=env, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError("the new version does not start:\n" + (out.stderr or out.stdout)[-2000:])
    found, where = (out.stdout.strip().splitlines() + ["", ""])[:2]
    if found != version:
        raise RuntimeError(f"after the update Python imports ScanBound {found} from {where}, expected {version}")


def save_state(path: str, **changes: object) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    for k, v in changes.items():
        if v is None:
            state.pop(k, None)
        else:
            state[k] = v
    fd, tmp = tempfile.mkstemp(prefix=".updater_", suffix=".json", dir=os.path.dirname(path))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--staged", required=True)
    ap.add_argument("--backups", required=True)
    ap.add_argument("--from-version", required=True)
    ap.add_argument("--to-version", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--state", default="")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cleanup", default="", help="the staging folder, removed after a successful update")
    ap.add_argument("--relaunch", action="store_true")
    ap.add_argument("--wait", type=float, default=120.0)
    a = ap.parse_args(argv)
    LOG = a.log
    state = a.state or os.path.join(os.path.dirname(os.path.abspath(a.backups)), "updater.json")
    target, staged = os.path.abspath(a.target), os.path.abspath(a.staged)
    log(f"Updating ScanBound {a.from_version} -> {a.to_version} in {target}")

    ok = False
    try:
        if not os.path.isfile(os.path.join(staged, PACKAGE, "__init__.py")):
            raise RuntimeError(f"{staged} does not hold a ScanBound release")
        if not os.path.isfile(os.path.join(target, PACKAGE, "__init__.py")):
            raise RuntimeError(f"{target} is not a ScanBound folder")
        log("Waiting for ScanBound to close ...")
        if not wait_for_exit(a.pid, a.wait):
            raise RuntimeError("ScanBound did not close, so nothing was changed")
        stamp = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
        journal = Journal(os.path.join(a.backups, f"backup-{a.from_version}-{stamp}"), target)
        reinstall = deps_changed(os.path.join(target, "pyproject.toml"), os.path.join(staged, "pyproject.toml"))
        try:
            n = replace(staged, target, journal)
            log(f"Replaced {n} program files; the old ones are in {journal.backup}")
            if reinstall:
                log("The dependencies changed: installing them (this can take a few minutes) ...")
                r = subprocess.run([a.python, "-m", "pip", "install", "--disable-pip-version-check",
                                    "-e", target + EXTRAS], cwd=target)
                if r.returncode != 0:
                    raise RuntimeError(f"pip install failed (exit code {r.returncode})")
            log("Checking that the new version starts ...")
            check_import(a.python, target, a.to_version)
        except BaseException:
            log("The update failed: putting the old version back")
            journal.rollback()
            raise
        if a.cleanup:
            shutil.rmtree(a.cleanup, ignore_errors=True)
        ok = True
        log(f"ScanBound {a.to_version} is installed")
    except BaseException as exc:  # noqa: BLE001 - reported to the user and to the app's next start
        log(f"ERROR: {exc}")
        try:
            save_state(state, update_error=str(exc))
        except OSError:
            pass
    if a.relaunch:
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen([a.python, "-m", "scanbound.app"], cwd=target, creationflags=flags)
    if not ok and sys.stdin is not None and sys.stdin.isatty():
        try:
            input("The update did not finish and the old version was kept. Press Enter to close.")
        except EOFError:
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
