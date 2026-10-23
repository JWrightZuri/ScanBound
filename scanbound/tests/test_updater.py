"""The updater: versions, release notes, staging a source update, and the helper that puts it in place
(including putting the old files back when something fails).  No network: downloads are faked."""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import pytest

from scanbound import __version__, updater
from scanbound import _update_apply as apply
from scanbound.project import PROJECT_VERSION, Project


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Keep ~/.scanbound/updater.json and the updates folder inside the test."""
    h = tmp_path / "home"
    monkeypatch.setattr(updater, "HOME", str(h))
    monkeypatch.setattr(updater, "UPDATES_DIR", str(h / "updates"))
    monkeypatch.setattr(updater, "STATE_FILE", str(h / "updater.json"))
    monkeypatch.delenv("SCANBOUND_NO_UPDATE_CHECK", raising=False)
    return h


def test_versions():
    assert updater.parse_version("v1.2.3") == (1, 2, 3, 9, 0)
    assert updater.is_newer("v0.2.0", "0.1.0")
    assert updater.is_newer("0.10.0", "0.9.9")
    assert updater.is_newer("1.0.0", "1.0.0rc2")
    assert not updater.is_newer("1.0.0rc2", "1.0.0")
    assert not updater.is_newer("0.1.0", "0.1.0")
    assert not updater.is_newer("nightly", "0.1.0")


API_RELEASE = {
    "tag_name": "v9.0.0", "name": "ScanBound 9.0.0", "html_url": "https://github.com/x/y/releases/tag/v9.0.0",
    "zipball_url": "https://api.github.com/repos/x/y/zipball/v9.0.0", "published_at": "2026-10-23T12:00:00Z",
    "body": "## What's new\r\n\r\n- Faster lattice fits\r\n- Fixed the crash on empty images\r\n* Autosave\r\n",
    "assets": [{"name": "ScanBound-Setup-9.0.0.exe", "browser_download_url": "https://example/setup.exe",
                "size": 123, "digest": "sha256:ABCDEF"}],
}


def test_release_from_api():
    r = updater.Release.from_api(API_RELEASE)
    assert (r.version, r.tag) == ("9.0.0", "v9.0.0")
    assert r.change_list() == ["Faster lattice fits", "Fixed the crash on empty images", "Autosave"]
    assert r.installer.name == "ScanBound-Setup-9.0.0.exe" and r.installer.sha256 == "abcdef"
    assert updater.Release.from_api({**API_RELEASE, "body": "Just one line"}).change_list() == ["Just one line"]


def test_check_and_skip(monkeypatch):
    monkeypatch.setattr(updater, "latest_release", lambda repo, timeout: updater.Release.from_api(API_RELEASE))
    r = updater.check_for_update("0.1.0")
    assert r is not None and updater.should_prompt(r)
    updater.save_state(skipped_version="9.0.0")
    assert not updater.should_prompt(r)
    assert updater.check_for_update("9.0.0") is None
    assert updater.check_on_startup()
    updater.save_state(check_on_startup=False)
    assert not updater.check_on_startup()


def test_download_checks_size_and_hash(tmp_path, monkeypatch):
    payload = b"x" * 1000

    class Resp(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(updater, "_get", lambda url, timeout, accept="": Resp(payload))
    import hashlib
    good = hashlib.sha256(payload).hexdigest()
    dest = str(tmp_path / "f.bin")
    assert updater.download("u", dest, 1000, good) == dest and open(dest, "rb").read() == payload
    os.unlink(dest)
    with pytest.raises(updater.UpdateError):
        updater.download("u", dest, 1000, "0" * 64)
    with pytest.raises(updater.UpdateError):
        updater.download("u", dest, 999)
    assert not os.path.exists(dest) and not os.path.exists(dest + ".part")


# ---------------------------------------------------------------------- source updates
def make_app(folder, version: str, extra: dict[str, str] | None = None) -> str:
    """A minimal scanbound folder: pyproject, a package that imports, a launcher."""
    files = {
        "pyproject.toml": f'[project]\nname = "scanbound"\nversion = "{version}"\ndependencies = []\n',
        "run_scanbound.sh": "echo old\n",
        "src/scanbound/__init__.py": f'__version__ = "{version}"\n',
        "src/scanbound/project.py": "",
        "src/scanbound/app/__init__.py": "",
        "src/scanbound/app/mainwindow.py": "",
        **(extra or {}),
    }
    for rel, text in files.items():
        p = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
    return str(folder)


def zip_release(folder, version, extra=None) -> bytes:
    src = make_app(folder / "src_tree" / "JWrightZuri-ScanBound-abc123" / "scanbound", version, extra)
    buf = io.BytesIO()
    base = os.path.dirname(os.path.dirname(src))
    with zipfile.ZipFile(buf, "w") as zf:
        for here, _, files in os.walk(base):
            for f in files:
                p = os.path.join(here, f)
                zf.write(p, os.path.relpath(p, base))
    return buf.getvalue()


def test_stage_source_update(tmp_path, monkeypatch):
    data = zip_release(tmp_path, "9.0.0")

    def fake_download(url, dest, size=0, sha256="", progress=None, cancelled=lambda: False, timeout=30):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(data)
        return dest
    monkeypatch.setattr(updater, "download", fake_download)
    staged = updater.stage_source_update(updater.Release.from_api(API_RELEASE))
    assert os.path.isfile(os.path.join(staged, "src", "scanbound", "__init__.py"))
    assert os.path.basename(staged) == "scanbound"
    with pytest.raises(updater.UpdateError, match="expected 9.1.0"):
        updater.stage_source_update(updater.Release.from_api({**API_RELEASE, "tag_name": "v9.1.0"}))


def test_detect_install(tmp_path):
    app = make_app(tmp_path / "plain" / "scanbound", "0.1.0")
    assert updater.detect_install(os.path.join(app, "src", "scanbound")).kind == "source"
    gitted = make_app(tmp_path / "repo" / "scanbound", "0.1.0")
    os.makedirs(tmp_path / "repo" / ".git")
    inst = updater.detect_install(os.path.join(gitted, "src", "scanbound"))
    assert inst.kind == "git" and not inst.can_update and "git pull" in inst.explain()
    assert updater.detect_install(str(tmp_path)).kind == "other"


def run_apply(tmp_path, target, staged, to_version, python=sys.executable):
    return apply.main(["--pid", str(2 ** 22 + 12345), "--target", target, "--staged", staged,
                       "--backups", str(tmp_path / "home" / "updates"), "--from-version", "0.1.0",
                       "--to-version", to_version, "--state", str(tmp_path / "home" / "updater.json"),
                       "--python", python, "--wait", "5"])


def test_apply_replaces_program_files_only(tmp_path, monkeypatch):
    target = make_app(tmp_path / "install" / "scanbound", "0.1.0", {
        "src/scanbound/removed_module.py": "",
        ".venv/keep.txt": "venv", "dist/ScanBound-Setup-0.1.0.exe": "exe",
        "packaging/windows/build/cache.bin": "cache", "packaging/windows/scanbound.iss": "old",
    })
    os.makedirs(tmp_path / "home", exist_ok=True)
    outside = tmp_path / "install" / "my_project" / "scanbound_project.json"
    os.makedirs(outside.parent)
    outside.write_text("{}")
    staged = make_app(tmp_path / "staged" / "scanbound", "0.2.0", {
        "src/scanbound/new_module.py": "", "packaging/windows/scanbound.iss": "new", "CHANGELOG.md": "- new",
    })
    monkeypatch.setattr(apply, "check_import", lambda python, target, version: None)
    assert run_apply(tmp_path, target, staged, "0.2.0") == 0
    t = lambda rel: os.path.join(target, rel)  # noqa: E731
    assert '"0.2.0"' in open(t("src/scanbound/__init__.py")).read()
    assert not os.path.exists(t("src/scanbound/removed_module.py")) and os.path.exists(t("src/scanbound/new_module.py"))
    assert open(t("packaging/windows/scanbound.iss")).read() == "new" and os.path.exists(t("CHANGELOG.md"))
    for kept in (".venv/keep.txt", "dist/ScanBound-Setup-0.1.0.exe", "packaging/windows/build/cache.bin"):
        assert os.path.exists(t(kept)), kept
    assert outside.read_text() == "{}"
    backups = [d for d in os.listdir(tmp_path / "home" / "updates") if d.startswith("backup-0.1.0-")]
    assert len(backups) == 1
    assert os.path.exists(tmp_path / "home" / "updates" / backups[0] / "src" / "scanbound" / "removed_module.py")


def test_apply_puts_the_old_version_back_on_failure(tmp_path, monkeypatch):
    target = make_app(tmp_path / "install" / "scanbound", "0.1.0", {"src/scanbound/old.py": "old"})
    os.makedirs(tmp_path / "home", exist_ok=True)
    staged = make_app(tmp_path / "staged" / "scanbound", "0.2.0", {"README.md": "new readme"})

    def broken(python, target, version):
        raise RuntimeError("the new version does not start")
    monkeypatch.setattr(apply, "check_import", broken)
    assert run_apply(tmp_path, target, staged, "0.2.0") == 1
    assert '"0.1.0"' in open(os.path.join(target, "src/scanbound/__init__.py")).read()
    assert os.path.exists(os.path.join(target, "src/scanbound/old.py"))
    assert not os.path.exists(os.path.join(target, "README.md"))
    state = json.load(open(tmp_path / "home" / "updater.json"))
    assert "does not start" in state["update_error"]


def test_import_check_catches_the_wrong_version(tmp_path):
    """check_import starts a separate Python: here it finds this checkout's ScanBound, not the version asked for."""
    with pytest.raises(RuntimeError, match="expected 99.0.0"):
        apply.check_import(sys.executable, str(tmp_path), "99.0.0")
    apply.check_import(sys.executable, str(tmp_path), __version__)


def test_finish_pending(monkeypatch):
    updater.save_state(installing=__version__)
    assert updater.finish_pending() == {"updated": __version__}
    updater.save_state(installing="99.0.0", update_error="pip failed")
    assert updater.finish_pending() == {"failed": "pip failed"}
    assert "installing" not in updater.load_state()


# ---------------------------------------------------------------------- project files across versions
def test_old_project_files_are_upgraded_with_a_backup(tmp_path, monkeypatch):
    from scanbound import project as proj
    p = Project(str(tmp_path))
    path = p.save()
    data = json.load(open(path))
    old = dict(data, version=PROJECT_VERSION, marker="old")
    json.dump(old, open(path, "w"))
    # pretend the current format is one newer, with a migration from the file's version
    monkeypatch.setattr(proj, "PROJECT_VERSION", PROJECT_VERSION + 1)
    monkeypatch.setitem(proj.MIGRATIONS, PROJECT_VERSION, lambda d: dict(d, migrated=True))
    assert proj.migrate_project_data(old)["migrated"] and proj.migrate_project_data(old)["version"] == PROJECT_VERSION + 1
    loaded = Project.load(path)
    assert loaded.loaded_version == PROJECT_VERSION
    loaded.save()
    backup = os.path.join(str(tmp_path), f"scanbound_project.v{PROJECT_VERSION}.bak.json")
    assert json.load(open(backup))["marker"] == "old"
    assert json.load(open(path))["version"] == PROJECT_VERSION + 1
    loaded.save()                                   # later saves leave the backup alone
    assert json.load(open(backup))["marker"] == "old"


def test_newer_project_files_are_refused(tmp_path):
    from scanbound.project import ProjectError
    p = Project(str(tmp_path))
    path = p.save()
    data = json.load(open(path))
    json.dump(dict(data, version=PROJECT_VERSION + 1), open(path, "w"))
    with pytest.raises(ProjectError, match="newer"):
        Project.load(path)
    assert json.load(open(path))["version"] == PROJECT_VERSION + 1       # and left as it was


# ---------------------------------------------------------------------- the dialog
def test_update_dialog(qapp_offscreen):
    from scanbound.app.updates import UpdateDialog
    r = updater.Release.from_api(API_RELEASE)
    dlg = UpdateDialog(r, updater.Install("source", "/x", sys.executable))
    assert dlg.b_install.text() == "Install and restart"
    dlg._done("skip")
    assert updater.load_state()["skipped_version"] == "9.0.0" and not updater.should_prompt(r)
    dlg = UpdateDialog(r, updater.Install("git", "/x", sys.executable))
    assert dlg.b_install.text() == "Open release page"


@pytest.fixture
def qapp_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
