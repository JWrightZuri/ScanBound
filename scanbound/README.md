# scanbound: the ScanBound detection core

`scanbound` is the part of ScanBound that finds objects in images: preprocessing, thresholding,
cleaning up the mask, building ROIs, measuring and labelling objects, and measuring nanoribbons.
It doesn't import PyQt, matplotlib or tkinter, and it has no global state. The GUI, scripts and
the command line all share the same `Detector`.

It sits next to `core/`, which now imports its detection code from here. Tests check that it gives
the same results as the original `core/`: the same masks, ROI points and feature tables, and the same labels.

## Install

From the ScanBound folder, using the project's Python environment:

```bash
pip install -e scanbound            # the detection core; add [gui] for the app, [dev] for pytest + mypy
```

## Run the desktop app

The app runs on Windows 10/11, macOS 12 or later (Intel and Apple silicon) and Linux (x86_64 and
aarch64) with Python 3.10 or newer. The launcher scripts in this folder create a Python environment in
`scanbound/.venv` on the first run, install everything into it (this takes a few minutes), and then
start the app. Later runs start straight away. Add `--update` to reinstall the dependencies, and add a
data folder or a `project.json` to open it straight away.

**Windows**: double-click `run_scanbound.bat`, or in a terminal:

```bat
run_scanbound.bat
```

**macOS**: install Python from [python.org](https://www.python.org/downloads/) or with
`brew install python@3.12`. Then, in Terminal:

```bash
cd path/to/ScanBound/scanbound
bash run_scanbound.sh
```

After that you can double-click `run_scanbound.command` in the Finder. If macOS says the file is not
executable, run `chmod +x run_scanbound.command run_scanbound.sh` once. If it says the file is from an
unidentified developer, right-click it and choose Open.

**Linux**: Python and the Qt system libraries come from the package manager. On Debian or Ubuntu:

```bash
sudo apt install python3 python3-venv libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 \
                 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libdbus-1-3
```

On Fedora: `sudo dnf install python3 mesa-libEGL libxkbcommon-x11 xcb-util-cursor xcb-util-wm xcb-util-keysyms`.
Then start it:

```bash
cd path/to/ScanBound/scanbound
bash run_scanbound.sh
```

**By hand**, in any environment of your own: `pip install -e "scanbound[gui,clusters,formats]"`, then
start `python -m scanbound.app` or `scanbound-app`. YOLO detection and training also need
Ultralytics YOLO and PyTorch (about 0.7 GB), which the standard install leaves out: use
**Machine learning → Install YOLO…** in the app, or `pip install -e "scanbound[ml]"`. Until then the
YOLO commands are shown but disabled.

### When something goes wrong

- **"Could not load the Qt platform plugin xcb"** (Linux): a system library is missing. Install the
  packages above. To find the missing one, run `QT_DEBUG_PLUGINS=1 bash run_scanbound.sh`.
- **Wayland** (Linux): the app runs natively on Wayland. If windows or the 3D view behave oddly, start
  it through XWayland with `QT_QPA_PLATFORM=xcb bash run_scanbound.sh`.
- **3D view is blank or reports an OpenGL error**: choose a different OpenGL context with the
  `SCANBOUND_OPENGL` variable. `core` is the OpenGL 4.1 core profile (the newest one macOS offers),
  `legacy` is OpenGL 2.1, and `software` is a software renderer (Windows and Linux).
  For example: `SCANBOUND_OPENGL=core bash run_scanbound.sh`, or on Windows
  `set SCANBOUND_OPENGL=software` and then `run_scanbound.bat`. The rest of the app does not use OpenGL.
- **Crashes** are logged in `~/.scanbound/crash.log` (on Windows: `%USERPROFILE%\.scanbound\crash.log`).
- Settings (window layout, theme, recent folder) are kept where Qt keeps them on each system: the
  registry on Windows, `~/Library/Preferences/com.scanbound.ScanBound.plist` on macOS, and
  `~/.config/ScanBound/ScanBound.conf` on Linux.

Projects are portable between systems. File references in `project.json` are relative to the project
and use `/`, and older files with Windows `\` separators open on macOS and Linux too. Annotations
imported from the old app's CSV files are matched by file name, ignoring the folder and upper or lower
case. (Linux file names are case-sensitive, and Windows and macOS usually are not.)

## Updates

When ScanBound starts it asks GitHub (in the background, silently when offline) whether a newer
release is out on [JWrightZuri/ScanBound](https://github.com/JWrightZuri/ScanBound/releases). If one
is, it shows the version's list of changes with **Install and restart**, **Skip this version** and
**Not now**. Updating is always optional; **Help → Check for updates…** checks at any time, and the
prompt has a box to stop the start-up check.

- **Installed with the Windows installer**: the new installer is downloaded, checked against the size
  and SHA-256 that GitHub reports, and run over the old version, which then starts again. The
  installer replaces the bundled Python, so if YOLO was installed the app offers to install it again.
- **Run from this folder** (`run_scanbound.bat` / `.sh`): the release's source is downloaded and
  checked, and after the app closes a helper replaces `src/scanbound` and the other program files,
  reinstalls the dependencies if they changed, checks that the new version imports, and restarts it.
  If any step fails, the old files are put back. The replaced files are kept in
  `~/.scanbound/updates/backup-<version>-<time>` and the log in `~/.scanbound/updates/update.log`.
- **A git checkout** (a developer copy): the prompt only tells you about the release; update with `git pull`.

An update replaces program files only. Projects (`scanbound_project.json` next to the data), the data
files, `~/.scanbound` (crash log, tutorial copy, updater state), `.venv` and Qt's settings are never
touched. Project files carry a format version: a newer ScanBound reads older files and upgrades them,
and the first save keeps the original next to it as `scanbound_project.v<N>.bak.json`. An older
ScanBound refuses a newer file instead of overwriting it.

To publish an update, raise the version in `pyproject.toml` and `src/scanbound/__init__.py`, add a
`## <version>` section to `CHANGELOG.md`, commit and push, then run
`powershell -ExecutionPolicy Bypass -File packaging\make_release.ps1`. It builds the installer and
creates the GitHub release that running copies then offer.

## Use it from Python

```python
from scanbound import Detector, DetectionConfig

cfg = DetectionConfig.load("data/user_data/image_adapt_params.json")   # saved by the dialog
det = Detector(cfg)

result = det.detect_file("scan_0001.sxm")          # any format microscopy_io supports
result = det.detect(image_array, px_scale=0.1)     # or a numpy image, nm per pixel

result.mask                 # bool mask from the pipeline
result.objects              # [SegObject]: id, roi_type, points, bbox, area
result.features             # DataFrame: area, length, width, aspect_ratio, orientation ...
result.labels               # {object id: label or None}
result.labeled_objects()    # [(SegObject, label)], the ROIs to add to the labelling page
result.annotation_rows()    # rows in the annotations_df layout
result.overlay()            # RGB uint8 preview

for item in det.detect_many(paths, workers=4):     # batch over several processes
    print(item.path, item.result.n_objects if item.ok else item.error)
```

Nanoribbons:

```python
det = Detector(DetectionConfig.nanoribbons())
r = det.detect_file("gnr.png", px_scale=0.1)
r.ribbons_table()           # length, width, angle, end points per skeleton segment (px and nm)
```

## Use it from the command line

```bash
scanbound detect -c data/user_data/image_adapt_params.json -o results/ scans/ --workers 4 --overlays
scanbound detect --preset nanoribbons --px-scale 0.1 -o results/ image.png
scanbound steps                       # every step and parameter, with ranges
scanbound check-config my_params.json # validate a saved pipeline
```

The `detect` command writes `annotations.csv` (the labelling page's format), `features.csv`,
`ribbons.csv` (when ribbons are measured), `summary.json`, and one `*_overlay.png` per image if you
pass `--overlays`. If one file fails, the other files are still processed. The failure is logged
and the command exits with code 1.

## Layout

```
scanbound/
  pyproject.toml
  src/scanbound/
    steps/registry.py   Param, StepSpec, StepRegistry: step descriptions with validation (no Qt)
    steps/adapt.py      gray -> mask steps (ADAPT_STEPS) and the default and nanoribbon presets
    steps/spm.py        SPM steps in physical units (SPM_STEPS)
    steps/fourier.py    2D FFT filter maths
    pipeline.py         Pipeline, PipelineCache, PipelineResult, and run_pipeline() (legacy signature)
    roi.py              mask -> RectROI / PolygonROI / LineSegmentROI / PointROI per object
    measure.py          shape features per object
    labeling.py         LabelingConfig (typed) and single / rules / bins / clusters assignment
    ribbons.py          skeleton segments: ribbon length, width and angle
    config.py           DetectionConfig: reads and writes the dialog's JSON and validates it
    detect.py           Detector, Detection, batch runs
    render.py           overlays without Qt
    io/                 microscopy_io file formats and load_image()
    export.py           CSV / JSON / PNG output
    cli.py              the `scanbound` command
    updater.py          new releases on GitHub, downloading and staging them (_update_apply.py installs them)
    errors.py           ScanBoundError > ConfigError (UnknownStepError, ParamError), PipelineError
  tests/                pytest suite, including the comparisons against core/
  run_scanbound.bat     app launchers: Windows, macOS / Linux, macOS Finder double-click
  run_scanbound.sh
  run_scanbound.command
```

Each layer only imports the layers listed above it. For example, `steps` knows nothing about
ROIs, and `detect` is the only module that ties everything together.

## Where the old code went

| Old (`core/`)                                    | New (`scanbound`)                     | Notes |
|--------------------------------------------------|---------------------------------------|-------|
| `image_adapt_dialog.py`: `f_*`, `Param`, `StepSpec`, `STEP_REGISTRY`, `DEFAULT_PIPELINE`, `run_pipeline`, `render_objects`, `to_float_gray` | `steps/adapt.py`, `steps/registry.py`, `pipeline.py`, `render.py`, `_image.py` | Same keys, parameters and results. Adds a new `rescale` step (a fixed intensity window). |
| `spm_processing.py`: `s_*`, `SPM_STEPS`          | `steps/spm.py`                        | Same results |
| `fourier_filter.py`: `build_mask`, `fourier_filter`, `find_peaks`, ... | `steps/fourier.py`     | Same results |
| `roi_extraction.py`                              | `roi.py`                              | Same results. Adds `RoiSettings.validate()` and `ROI_PARAMS` (the dialog's ROI editors). |
| `object_labeling.py`                             | `measure.py` and `labeling.py`        | Same results. Config is a typed `LabelingConfig`, and plain dicts are still accepted. |
| `GNR_segment.py`, `GNR_segmenter_standalone.py`  | `ribbons.py` and the `nanoribbons` preset | About 3.5x faster on a 1024² image, and also measures width. No `print`s or plotting. |
| `microscopy_io.py`                               | `io/microscopy.py`                    | Copied unchanged |
| `first_pass_segment*.py`, `second_pass_segment*.py`, `filtering_tester.py`, `feature_parameters.py`, `sharpen_test.py` | dropped | You decided to drop the old molecule and chirality segmenters in favour of the Image Adapt pipeline (`steps/adapt.py`). They also used tkinter or matplotlib windows. |
| Qt widgets (`StepCard`, `ParamControl`, dialogs, panels) | stay in `core/` | UI |
| `yolo_*.py`, `roi_analysis.py`                   | stay in `core/` for now               | Already Qt-free. They could move in a later step. |

## How core/ uses it (switched on 21 Oct 2026)

`core/` now imports the detection code from this package, so there is only one copy of it:

- `core/image_adapt_dialog.py`, `core/spm_processing.py` and `core/fourier_filter.py` keep their
  Qt widgets and import the steps, registry, pipeline runner and renderer from `scanbound`. The
  old names (`STEP_REGISTRY`, `run_pipeline`, `f_threshold`, `OBJECT_PARAMS` ...) still work.
- `core/roi_extraction.py`, `core/object_labeling.py` and `core/microscopy_io.py` are now
  re-exports of `scanbound.roi`, `scanbound.measure` + `scanbound.labeling`, and `scanbound.io.microscopy`.
- `core/_scanbound.py` finds the package: it uses an installed `scanbound`, and otherwise adds
  `scanbound/src` to the import path, so the app runs with or without `pip install -e scanbound`.
- The step menu has one more entry, **Intensity window** (`rescale`). Everything else behaves as before.
- The original `core/` files are in `backups/core_before_scanbound.zip`.

New steps now go in `scanbound/src/scanbound/steps/` and show up in the dialogs automatically.
Batch actions in the pages could call `Detector.detect_many` next, in place of their own loops.

## Checking it

```bash
cd scanbound
pytest            # comparison tests: point SCANBOUND_LEGACY_ROOT at a folder holding the
                  # original core/ (from the backup zip) and data/; by default they use ../core,
                  # which now checks that the core/ shims still work
mypy src
```

The GUI tests run without a screen (`QT_QPA_PLATFORM=offscreen`, set by the tests themselves), so
they also run on a Linux server or in CI. To include the 3D views' OpenGL on Linux without a display,
run them under a virtual X server: `QT_QPA_PLATFORM=xcb xvfb-run -a pytest`.
