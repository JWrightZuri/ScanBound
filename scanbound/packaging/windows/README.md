# Windows installer

`build_installer.ps1` builds `dist\ScanBound-Setup-<version>.exe`, a single file people can download and
run. The wizard lets them pick an install folder (default `%LOCALAPPDATA%\Programs\ScanBound`, no
administrator rights needed), shows a progress bar, and adds Start menu and optional desktop shortcuts.
Uninstall it from Windows Settings → Apps.

## Building

One-time setup: `winget install JRSoftware.InnoSetup`. The C# compiler for the launcher ships with Windows.

```
powershell -ExecutionPolicy Bypass -File packaging\windows\build_installer.ps1
```

Add `-SkipRuntime` to reuse the staged Python after changing only the launcher or the installer
script. The version comes from `pyproject.toml`, the icon from `src\scanbound\app\resources\sb_logo.png`.

## What gets installed

```
ScanBound\
  ScanBound.exe     launcher (ScanBound.cs): runs runtime\pythonw.exe -m scanbound.app
  runtime\          python.org's Python 3.12 (the nuget.org "python" package) with ScanBound
                    and its gui, clusters and formats extras installed in Lib\site-packages
```

The app runs from a real, private Python rather than a frozen bundle, so **Machine learning → Install
YOLO…** works the same as in the developer setup: it runs `runtime\python.exe -m pip install
ultralytics` into the install folder. YOLO and PyTorch are not in the installer. Upgrading
replaces the runtime folder, so YOLO has to be installed again after an upgrade.

If the app exits with an error, ScanBound.exe shows its last messages and saves them to
`%LOCALAPPDATA%\ScanBound\last_crash.txt`.

## Running from source / VS Code

`run_scanbound.bat` in the scanbound folder is unchanged: it builds `.venv` with an editable
install on first run and starts the app with a console for debug messages. The VS Code launch
configuration uses that `.venv`.
