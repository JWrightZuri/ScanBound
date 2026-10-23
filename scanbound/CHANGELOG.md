# Changes

Each version's section is the release note on GitHub, and its list is what the update prompt in the
app shows. Keep the items short. `packaging/make_release.ps1` publishes the section for the version in
`pyproject.toml`.

## 0.1.0

- The rebuilt ScanBound app (PyQt6): browse SPM data, annotate, detect, measure and analyse surfaces
- Windows installer, plus launchers for Windows, macOS and Linux
- Checks GitHub for new versions at start-up and installs them on request; your projects, data and
  settings are never changed by an update
