# ScanBound

ScanBound is a scanning probe microscopy (STM / AFM) analysis tool: browse and process SPM data,
annotate and detect nanostructures, measure them, and analyse surfaces. It runs on Windows, macOS
and Linux.

The app is in the [`scanbound`](scanbound) folder. See [scanbound/README.md](scanbound/README.md) for
installing, running, troubleshooting and the Python API, and [scanbound/CHANGELOG.md](scanbound/CHANGELOG.md)
for what changed in each version.

## Install

- **Windows**: download `ScanBound-Setup-<version>.exe` from the
  [latest release](https://github.com/JWrightZuri/ScanBound/releases/latest) and run it. No
  administrator rights are needed.
- **From source** (Windows, macOS, Linux, Python 3.10 or newer): clone or download this repository,
  then run `scanbound\run_scanbound.bat` on Windows or `bash scanbound/run_scanbound.sh` on macOS and
  Linux. The first run sets up its own Python environment.

## Updates

ScanBound checks for a new release when it starts and shows what changed. Installing it is optional,
and an update never changes your projects, data or settings. **Help → Check for updates…** checks at
any time.
