"""
The ScanBound desktop app (PyQt6), a thin layer over the Qt-free model.

    session      Session: the open project, current image, selection; turns model changes into Qt signals
    canvas       ImageCanvas: shows one image (NaN transparent, true pixel aspect) and its annotations
    layer        AnnotationLayer: draws the AnnotationStore, updated incrementally from store events
    tools        drawing / editing tools (pure interaction logic on top of the store)
    docks/       browser, labels, review (switched off, see features.py), processing & display, info, measurements, spectra, log
    dialogs/     detection, YOLO, dataset export, training, evaluation, analysis
    widgets/     widgets carried over from the old app
    mainwindow   MainWindow: menus, toolbars and docks

Run with ``python -m scanbound.app [folder or project.json]``.
"""


def main(argv=None) -> int:
    from .mainwindow import run
    return run(argv)
