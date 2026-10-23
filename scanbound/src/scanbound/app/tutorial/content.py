"""
The tutorials, as data.  To add one: write its steps with the helpers of ``engine`` and append it to
TUTORIALS; Help → Tutorials… lists it.  Targets find widgets by what the user sees (menu titles,
button texts, panel names), so the widgets themselves need no changes.
"""

from __future__ import annotations


from .engine import (Ctx, Step, Tutorial, button, dock, find_button, group, in_window, later, menu, raise_dock, tool,
                     window)

ADAPT = "ImageAdaptDialog"


# ---------------------------------------------------------------------------------------------------- helpers
def _image_open(c: Ctx) -> bool:
    return bool(c.session.current_id) and c.win.canvas.image is not None


def _anns(c: Ctx, manual: bool = True) -> list:
    return [a for a in c.session.annotations() if a.kind == "feature" and (a.origin.method == "manual") == manual]


def _proposals(c: Ctx, status: str = "proposed") -> list:
    return [a for a in c.session.annotations() if a.status == status and a.origin.method != "manual"]


def _remember(key: str, fn):  # noqa: ANN001, ANN202 - a Step.enter that stores fn(c) in the memo
    def enter(c: Ctx) -> None:
        c.memo[key] = fn(c)
    return enter


def _open_practice(c: Ctx) -> None:
    from .sample import practice_folder

    def go() -> None:
        if c.session.project.dirty and not c.win._confirm_discard():
            return
        c.win._open_path(practice_folder())
    later(go)


def _processing_tab(c: Ctx) -> None:
    raise_dock(c, "Display")
    c.win.processing.setCurrentWidget(c.win.processing.processing)


def _plane_on(c: Ctx) -> None:
    _processing_tab(c)
    panel = c.win.processing.processing.panel
    card = next((k for k in panel.cards if k.spec.key == "plane"), None)
    if card is None:
        panel.add_step("plane")
    else:
        card.enable_cb.setChecked(True)


def _add_label(c: Ctx) -> None:
    s = c.session
    name = "molecule"
    if name not in s.project.labels:
        s.project.labels.add(name)
        s.labels_edited()
    s.set_current_label(name)


def _draw_one(c: Ctx) -> None:
    """Draw a rectangle around a feature: the brightest spot of the image, or its centre."""
    import numpy as np
    from scipy import ndimage
    c.win.canvas.set_tool("rect")
    img = c.session.image()
    data = np.nan_to_num(np.asarray(img.data, float)) if img is not None else None
    if data is None or data.ndim != 2:
        return
    h, w = data.shape
    smooth = ndimage.gaussian_filter(data - ndimage.uniform_filter(data, max(h, w) // 8), 3)
    margin = max(h, w) // 12
    inner = smooth[margin:h - margin, margin:w - margin]
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    x, y, r = x + margin, y + margin, max(h, w) / 24
    t = c.win.canvas.tool
    t.press(x - r, y - r)
    t.move(x + r, y + r)
    t.release(x + r, y + r)
    c.win.canvas.layer.refresh()


def _delete_last(c: Ctx) -> None:
    anns = _anns(c)
    if anns:
        c.session.store.remove([anns[-1].id], description="Delete annotation")


def _tune(c: Ctx) -> None:
    if window(ADAPT) is None:
        later(c.win.tune_adapt)


def _apply_adapt(c: Ctx) -> None:
    dlg = window(ADAPT)
    if dlg is not None:
        dlg.accept()


def _adapt_saved(c: Ctx) -> bool:
    from ..dialogs.detect import ADAPT_KEY
    return window(ADAPT) is None and bool(c.session.project.settings.get(ADAPT_KEY))


def _adapt_part(name: str):  # noqa: ANN202
    return in_window(ADAPT, lambda d: getattr(d, name))


def _surface(c: Ctx):  # noqa: ANN202
    return c.win.surface


def _analysis(_c: Ctx):  # noqa: ANN202
    return window("RoiAnalysisDialog")


def _builder(_c: Ctx):  # noqa: ANN202
    return window("Builder3DWindow")


def _figure(_c: Ctx):  # noqa: ANN202
    return window("FigureDialog")


def _open_analysis(c: Ctx, keys: tuple = ()) -> None:
    from PyQt6.QtCore import Qt
    c.win.show_analysis()
    dlg = window("RoiAnalysisDialog")
    if dlg is None or not keys:
        return
    lst = dlg.analysis_list
    for k in range(lst.count()):
        it = lst.item(k)
        on = it.data(Qt.ItemDataRole.UserRole) in keys
        it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        if on:
            lst.setCurrentItem(it)


def _analysis_item(key: str):  # noqa: ANN202
    from PyQt6.QtCore import Qt

    def find(c: Ctx):  # noqa: ANN202
        dlg = window("RoiAnalysisDialog")
        if dlg is None:
            return None
        lst = dlg.analysis_list
        for k in range(lst.count()):
            if lst.item(k).data(Qt.ItemDataRole.UserRole) == key:
                return lst.viewport(), lst.visualItemRect(lst.item(k))
        return lst
    return find


# ---------------------------------------------------------------------------------------------------- shared
OPEN_IMAGE = Step(
    "Open an image",
    "ScanBound works on a <b>data folder</b>: every scan in it (.sxm, .sm4, .mtrx, .gwy, .ibw, TIFF, PNG …) "
    "is listed in the <b>Files</b> panel, and your annotations are saved next to the data in "
    "<i>scanbound_project.json</i>. The data files themselves are never changed.<br><br>"
    "Open one of your folders with <b>File → Open data folder…</b> (Ctrl+O), or press <b>Open the practice "
    "image</b> to follow along on an example scan: 9AP molecules on PdGa(111).",
    target=menu("File"),
    done=_image_open,
    show=_open_practice, show_text="Open the practice image",
    task="open a data folder (File → Open data folder…).")


# ---------------------------------------------------------------------------------------------------- basics
GETTING_STARTED = Tutorial(
    "getting_started", "Getting started",
    "Open a scan, process it, make a label, draw and delete a shape, find features with the adapt "
    "pipeline and correct what it found.",
    minutes=8, needs_image=False,
    steps=[
        Step("Welcome to ScanBound",
             "This tour walks you through one complete pass over an image, using the real program: the "
             "highlighted part of the window is where to look, and each step ticks itself off when you have "
             "done it.<br><br>"
             "If you would rather watch, every step has a button that does it for you. You can drag this card "
             "anywhere, and reopen the tour at any time from <b>Help → Tutorials…</b> (F1)."),
        OPEN_IMAGE,
        Step("Browse your files",
             "The <b>Files</b> panel lists every scan in the folder, with a thumbnail and the number of "
             "annotations. Click a file to show it; <b>Page Up / Page Down</b> step through them. The filter box "
             "and the ★ (favourites) narrow the list.<br><br>"
             "On the image: the <b>wheel</b> zooms, <b>middle-drag</b> (or Space + drag) pans and <b>F</b> fits "
             "the image to the window.",
             target=dock("Files"), enter=lambda c: raise_dock(c, "Files")),
        Step("Adjust the display",
             "The <b>Display</b> tab sets how the image looks: colormap, levels and gamma. This only changes "
             "the picture, never the data.",
             target=dock("Display"),
             enter=lambda c: (raise_dock(c, "Display"), c.win.processing.setCurrentIndex(0))),
        Step("Process the image",
             "Raw scans are usually tilted. The <b>Processing</b> tab holds a chain of correction steps that run "
             "top to bottom; each image keeps its own chain, saved with the project.<br><br>"
             "Tick <b>Plane level</b> to remove the tilt. <b>＋ Add step</b> adds more (align rows, remove scars, "
             "smoothing, Fourier filter …), and <b>Apply to all images</b> copies the chain to every scan.",
             target=lambda c: c.win.processing.processing.panel, enter=_processing_tab,
             done=lambda c: bool(c.session.entry and c.session.entry.chain.active),
             show=_plane_on, task="tick Plane level."),
        Step("Create a label",
             "Labels are the names of the things you annotate (molecule, island, defect …). Open the "
             "<b>Labels</b> panel and press <b>＋</b> to add one; call it e.g. <i>molecule</i>.<br><br>"
             "Clicking a label makes it the label for new shapes. The first nine labels have number keys: "
             "<b>1–9</b> label the selected shapes, <b>0</b> removes the label.",
             target=button("＋", lambda c: c.win.labels),
             enter=lambda c: (raise_dock(c, "Labels"), c.memo.update(labels=len(c.session.project.labels.names))),
             done=lambda c: len(c.session.project.labels.names) > c.memo.get("labels", 0)
             or (c.memo.get("labels", 0) > 0 and c.session.current_label is not None),
             show=_add_label, task="add a label with ＋ in the Labels panel."),
        Step("Pick a drawing tool",
             "The tool bar on the left holds the drawing and editing tools; each has a one-key shortcut. "
             "Choose the <b>Rectangle</b> tool (B) or the <b>Polygon</b> tool (G).",
             target=lambda c: c.win.tool_toolbar,
             done=lambda c: c.win.canvas.tool.name in ("rect", "ellipse", "polygon"),
             show=lambda c: c.win.canvas.set_tool("rect"), show_text="Choose the rectangle",
             task="choose Rectangle (B) or Polygon (G)."),
        Step("Draw a shape",
             "Draw around one feature on the image:<br>"
             "• <b>Rectangle</b>: press, drag and release.<br>"
             "• <b>Polygon</b>: click the corners, then click the first point, double-click or press Enter.<br><br>"
             "The shape gets the current label and its colour. The status bar at the bottom always shows how "
             "the current tool works.",
             target=lambda c: c.win.canvas,
             enter=_remember("drawn", lambda c: len(_anns(c))),
             done=lambda c: len(_anns(c)) > c.memo.get("drawn", 0),
             show=_draw_one, show_text="Draw one for me", task="draw a shape around a feature."),
        Step("Edit or remove a shape",
             "Switch to <b>Select / edit</b> (S) and click a shape to select it: drag it to move, drag its "
             "handles to reshape, arrow keys nudge it.<br><br>"
             "To remove it, select it and press <b>Delete</b> (or right-click → Delete). With a drawing tool "
             "you can also hover a shape and press Delete. Made a mistake? <b>Ctrl+Z</b> undoes it.",
             target=lambda c: c.win.canvas,
             enter=lambda c: (c.memo.update(before=len(_anns(c))), c.win.canvas.set_tool("select")),
             done=lambda c: len(_anns(c)) < c.memo.get("before", 0),
             show=_delete_last, show_text="Delete it for me", task="select the shape and press Delete."),
        Step("Find features automatically",
             "Drawing every feature by hand is slow. The <b>adapt pipeline</b> finds them for you with a chain "
             "of image steps you tune on this image.<br><br>Open <b>Detect → Tune adapt pipeline…</b>.",
             target=menu("Detect"),
             done=lambda c: window(ADAPT) is not None or _adapt_saved(c),
             show=_tune, show_text="Open it for me", task="open Detect → Tune adapt pipeline…"),
        Step("The pipeline steps",
             "Each card is one step and they run top to bottom: smoothing, background removal, a "
             "<b>threshold</b> that separates features from the surface, and a size filter that drops specks. "
             "Tick a card to switch it on; move its sliders and the preview above updates live.<br><br>"
             "<b>＋ Add step</b> adds more (morphology, fill holes, CLAHE, skeletonize …); <b>Save…</b> keeps a "
             "pipeline for other projects.",
             target=_adapt_part("tabs"), show=_tune, show_text="Open the dialog"),
        Step("Check the preview",
             "The preview shows what will be detected. <b>View</b> switches between the result, the original "
             "and both side by side; <b>⤢ Large preview</b> enlarges it.<br><br>"
             "Aim for one coloured object per feature. If neighbours merge, change the threshold or add a "
             "<i>Morphology</i> step (opening); if specks remain, raise the minimum size of the size filter.",
             target=_adapt_part("preview"), show=_tune, show_text="Open the dialog"),
        Step("Objects and labels",
             "The <b>Labels</b> tab of this dialog gives the found objects a label: one label for all, or "
             "rules on size and shape so that, for example, round islands and long rods get different labels.",
             target=in_window(ADAPT, lambda d: d.tabs.tabBar()), show=_tune, show_text="Open the dialog"),
        Step("Save the pipeline",
             "Press <b>Apply</b>: the settings are stored in the project and used every time you run the "
             "pipeline.",
             target=in_window(ADAPT, lambda d: find_button(d, "Apply")),
             done=_adapt_saved, show=_apply_adapt, show_text="Apply for me", task="press Apply."),
        Step("Run the pipeline",
             "Now run it: <b>Detect → Run adapt pipeline on this image</b> (Ctrl+D). The same menu runs it on "
             "every image at once.<br><br>"
             "What it finds is added to the image as annotations with the pipeline's labels.",
             target=menu("Detect"),
             done=lambda c: bool(_proposals(c, "accepted")),
             show=lambda c: later(lambda: c.win.run_adapt(False)), show_text="Run it for me",
             task="run Detect → Run adapt pipeline on this image."),
        Step("Correct the results",
             "Fix anything the pipeline got wrong the same way as your own shapes: with <b>Select</b> (V) "
             "click a shape and drag it or its corners, press a <b>digit</b> to give it another label, or "
             "<b>Del</b> to remove it. Draw anything it missed with the drawing tools.<br><br>"
             "Running the pipeline again replaces the results you have not touched; your edits stay.",
             target=tool("select")),
        Step("You're done",
             "That is the whole loop: open, process, label, draw, detect, correct. Save with <b>Ctrl+S</b>; "
             "everything goes into the project file next to the data.<br><br>"
             "<b>Help → Tutorials…</b> (F1) has this tour again and tours of the advanced tools: measuring, "
             "surface and lattice analysis, the 3D model builder, nanoribbons and publication figures."),
    ])


MEASURE = Tutorial(
    "measure", "Measure and analyse",
    "Distances and height profiles, the Measure panel, and the ROI analysis window with its statistics "
    "and plots.",
    level="Advanced", minutes=5,
    steps=[
        Step("Measure a distance",
             "Choose <b>Measure / profile</b> (M) and drag across the image. The length appears on the line, in "
             "nm when the image is calibrated (<b>Process → Set pixel size…</b> if it is not).",
             target=tool("measure"), done=lambda c: c.win.canvas.tool.name == "measure",
             show=lambda c: c.win.canvas.set_tool("measure"), show_text="Choose it", task="choose Measure (M)."),
        Step("Height profiles",
             "<b>View → Line profiles…</b> (Ctrl+P) shows the height along every measure line and updates while "
             "you move them: step heights, apparent heights and periodicities.",
             target=menu("View"), show=lambda c: c.win.show_profiles(), show_text="Open the profiles"),
        Step("The Measure panel",
             "Select one or more shapes and the <b>Measure</b> panel lists their measurements (area, length, "
             "width, orientation …), each with its uncertainty. Hover a value to see how it was calculated. "
             "<b>Copy</b> copies them; <b>Export all…</b> writes every annotated image as CSV.",
             target=dock("Measure"), enter=lambda c: raise_dock(c, "Measure")),
        Step("ROI analysis",
             "<b>Analyze → ROI analysis…</b> collects statistics over this image or all images: label "
             "composition, coverage, size distributions, orientations and the nanoribbon analyses.",
             target=menu("Analyze"), done=lambda c: _analysis(c) is not None,
             show=lambda c: _open_analysis(c), show_text="Open it for me", task="open Analyze → ROI analysis…"),
        Step("Choose what to analyse",
             "Pick the images (current or all), tick the labels and the analyses. Selecting an analysis shows "
             "its description and parameters below the list.",
             target=lambda c: _analysis(c) and _analysis(c).analysis_list,
             show=lambda c: _open_analysis(c), show_text="Open the window"),
        Step("Run and export",
             "<b>Run analysis</b> fills one tab per analysis with plots and tables; <b>Export results…</b> writes "
             "them all (PNG/SVG plots and CSV tables). <b>Analyze → Export all measurements…</b> exports the raw "
             "per-shape table.",
             target=lambda c: _analysis(c) and _analysis(c).btn_run,
             show=lambda c: (_analysis(c) or _open_analysis(c)) and _analysis(c) and _analysis(c).run(),
             show_text="Run it"),
    ])


SURFACE = Tutorial(
    "surface", "Surface and lattice analysis",
    "Fit the substrate lattice, calibrate from it, align a structure model and run the surface analyses "
    "(adsorption sites, assembly, defects, step edges, scan quality).",
    level="Advanced", minutes=7,
    steps=[
        Step("The Surface panel",
             "Atomically resolved images can be analysed against the substrate lattice. Everything for that is "
             "in the <b>Surface</b> panel (and the <b>Surface</b> menu).",
             target=dock("Surface"), enter=lambda c: raise_dock(c, "Surface")),
        Step("Find the lattice spacing",
             "Press <b>Estimate</b> to read the lattice spacing from the Fourier transform, or type it. Choose "
             "whether the atoms show as <i>bright</i> or <i>dark</i> sites.",
             target=group("Substrate lattice", _surface), enter=lambda c: raise_dock(c, "Surface"),
             show=lambda c: c.win.surface.estimate_spacing(), show_text="Estimate it"),
        Step("Preview the sites",
             "Tick <b>Preview sites</b> and move the <b>Threshold</b>: green circles are the sites the fit will "
             "use, grey ones are too weak. Aim for most atoms green and no noise picked up.",
             target=lambda c: c.win.surface.preview,
             done=lambda c: c.win.surface.preview.isChecked(),
             show=lambda c: c.win.surface.preview.setChecked(True), show_text="Tick it",
             task="tick Preview sites."),
        Step("Fit the lattice",
             "<b>Fit lattice</b> (Ctrl+L) indexes the sites and fits origin and lattice vectors with "
             "uncertainties; the result appears below the button and on the image. Draw an <b>Analysis region</b> "
             "(J) labelled <i>exclude</i> over areas to leave out.",
             target=lambda c: c.win.surface.fit_btn,
             show=lambda c: c.win.surface.fit_lattice(), show_text="Fit it"),
        Step("Calibrate from the lattice",
             "<b>Calibrate…</b> sets the pixel size from a known lattice constant (Au(111), Cu(111), graphene …), "
             "correcting piezo calibration errors for every measurement on this image.",
             target=button("Calibrate", _surface)),
        Step("Show it on the image",
             "Choose what is drawn: fitted sites (green → red with distance from the ideal lattice), the ideal "
             "lattice, the unit cell, the aligned model; the map shows local strain or the domains.",
             target=group("Show on the image", _surface)),
        Step("Align a structure model",
             "A <b>structure model</b> (a preset such as fcc(111), or an imported .xyz / JSON) tells which "
             "position in the cell is top, bridge or hollow. Choose the alignment and which model position sits "
             "on the image maxima, then <b>Use</b>.",
             target=group("Structure model", _surface)),
        Step("Which annotations are molecules",
             "Tell the analyses which labels are molecules, which are defects and which are left- / "
             "right-handed, and the molecule's rotational symmetry.",
             target=group("Molecules", _surface)),
        Step("Run the analyses",
             "Each button writes a report: <b>Structure</b> (symmetry, strain, domains), <b>Adsorption sites</b>, "
             "<b>Assembly</b> (neighbours, g(r), order), <b>Defects</b>, <b>Step edges</b>, "
             "<b>Terraces</b> and <b>Scan quality</b>. <b>Explain selected</b> ranks explanations for "
             "one odd feature.",
             target=group("Analyse this image", _surface)),
        Step("Several images",
             "The <b>Surface</b> menu also registers images onto each other (other bias, ncAFM, later frames), "
             "correlates molecules across channels and tracks them through a time series.",
             target=menu("Surface")),
    ])


BUILDER = Tutorial(
    "builder3d", "3D model builder",
    "Place molecule geometries (.xyz) on your annotated molecules, on a substrate from the lattice fit, "
    "adjust them in 3D and export the model.",
    level="Advanced", minutes=5,
    steps=[
        Step("Molecule geometries",
             "The builder places real molecules on your annotations. In the <b>Surface</b> panel's "
             "<b>3D model (.xyz)</b> box, <b>Add .xyz…</b> loads one or more geometries (XYZ or extended XYZ, "
             "Å); they are saved in the project.",
             target=group("3D model (.xyz)", _surface), enter=lambda c: raise_dock(c, "Surface")),
        Step("Label → molecule",
             "Below the list, choose which molecule goes on each label: every annotation with that label "
             "becomes one molecule, at its position and orientation (set orientations with the Orientation "
             "tool (O) or Detect → Estimate orientations).",
             target=group("3D model (.xyz)", _surface)),
        Step("Build the model",
             "<b>Build 3D model (.xyz)…</b> opens the 3D window. The substrate comes from the lattice fit and "
             "the aligned structure model when there is one.",
             target=button("Build 3D model", _surface),
             done=lambda c: _builder(c) is not None,
             show=lambda c: c.win.surface.build_model(), show_text="Open it for me",
             task="press Build 3D model (.xyz)…"),
        Step("Substrate and defaults",
             "For each molecule file choose <i>Lay flat</i>, an angle offset and which face is down. Under "
             "<b>Substrate and defaults</b> set the layers, extent and default height, and whether molecules "
             "snap to their adsorption site; <b>Rebuild</b> applies the changes.",
             target=lambda c: find_button(_builder(c), "Rebuild") or _builder(c)),
        Step("Adjust single molecules",
             "Click molecules in the 3D view to select them, then change their height, angle, tilt, shift or "
             "mirror them in <b>Selected molecules</b>. <b>Top view</b> / <b>Side view</b> help to check "
             "overlaps.",
             target=lambda c: find_button(_builder(c), "Top view") or _builder(c)),
        Step("Export",
             "<b>Export .xyz…</b> writes the whole model (substrate and molecules) for DFT, ASE or a renderer.",
             target=lambda c: find_button(_builder(c), "Export") or _builder(c)),
    ])


RIBBONS = Tutorial(
    "ribbons", "Shape skeletons: length, branching, herringbone",
    "Outline or paint any elongated objects (nanoribbons, chains, islands), split them into segments, measure "
    "length, width and branching, and their alignment to the Au(111) herringbone.",
    level="Advanced", minutes=7,
    steps=[
        Step("Shapes first",
             "The skeleton analyses start from <b>area ROIs</b> of any label: polygons, rectangles, ellipses or "
             "painted masks, drawn by hand or found by the adapt pipeline (line align, an intensity window and an "
             "adaptive threshold work well on Au(111)).",
             target=tool("polygon")),
        Step("Split into segments",
             "<b>Analyze → Split shapes into centre lines</b> skeletonises the selected shapes (or all of them) "
             "and adds one <b>centre line</b> per segment, split at junctions and sharp bends. Correct or delete "
             "them like any shape; the skeleton analyses use these centre lines.",
             target=menu("Analyze")),
        Step("Centre lines by hand",
             "You can also draw or correct centre lines with the <b>Centre line</b> tool (C): click along the "
             "middle of an object, double-click to finish. <b>Detect → Centre lines → outlines…</b> turns them "
             "into outlines of a given width.",
             target=tool("centre_line")),
        Step("Mark the herringbone",
             "For the alignment, the herringbone direction is needed. Let <b>Analyze → Detect herringbone "
             "direction</b> measure it on the bare gold (it adds axes in every herringbone domain), or drag the "
             "<b>Herringbone axis</b> tool (W) along the ridges.",
             target=tool("herringbone")),
        Step("Open the analyses",
             "The skeleton statistics are in <b>Analyze → ROI analysis…</b>.",
             target=menu("Analyze"), done=lambda c: _analysis(c) is not None,
             show=lambda c: _open_analysis(c, ("skeleton", "alignment")), show_text="Open it for me",
             task="open Analyze → ROI analysis…"),
        Step("Length, width, branching",
             "<b>Skeleton: length, width, branching</b> measures every segment (length, width, straightness) "
             "and counts branch, fusion and crossing points per shape, with histograms and a map. Its parameters "
             "(kink angle, straightness tolerance, spur length) appear below the list when it is selected.",
             target=_analysis_item("skeleton"),
             show=lambda c: _open_analysis(c, ("skeleton", "alignment")), show_text="Select it"),
        Step("Alignment to the herringbone",
             "<b>Alignment to the herringbone</b> gives the angle of every segment to the herringbone ridges of "
             "its domain (marked or measured), as a histogram, a rose diagram and a table.",
             target=_analysis_item("alignment"),
             show=lambda c: _open_analysis(c, ("skeleton", "alignment")), show_text="Select it"),
        Step("Run them",
             "<b>Run analysis</b> fills a tab per analysis; <b>Export results…</b> saves plots and tables.",
             target=lambda c: _analysis(c) and _analysis(c).btn_run),
    ])


PUBLICATION = Tutorial(
    "publication", "Publication figures and export",
    "Lay out an image for print with scale bar and colour bar, copy it into slides, and export images and "
    "measurements.",
    level="Advanced", minutes=4,
    steps=[
        Step("Frame the view",
             "The figure shows what the canvas shows: zoom and pan to the area you want, and set the colormap "
             "and levels in the <b>Display</b> tab. <b>View</b> switches the scale bar, colour bar and label "
             "names on the canvas.",
             target=dock("Display"), enter=lambda c: (raise_dock(c, "Display"), c.win.processing.setCurrentIndex(0))),
        Step("Publication figure",
             "<b>File → Publication figure…</b> (Ctrl+Shift+E) lays the image out for print.",
             target=menu("File"), done=lambda c: _figure(c) is not None,
             show=lambda c: later(c.win.publication_figure), show_text="Open it for me",
             task="open File → Publication figure…"),
        Step("Size, fonts, bars",
             "Set the width (single or double column, or mm), resolution, font and size, the scale bar and "
             "colour bar, a panel letter, and whether annotations are drawn. The preview is what is saved.",
             target=lambda c: _figure(c)),
        Step("Save or copy",
             "<b>Save…</b> writes PNG, TIFF, PDF or SVG at the chosen resolution; <b>Copy to clipboard</b> pastes "
             "straight into slides. Later, <b>File → Copy figure</b> (Ctrl+Shift+C) copies with the same style "
             "without opening the dialog.",
             target=lambda c: find_button(_figure(c), "Save") or _figure(c)),
        Step("Exporting data",
             "<b>File → Export image…</b> / <b>Export all images…</b> write the processed images; "
             "<b>Analyze → Export all measurements…</b> writes every shape with its measurements as CSV; "
             "<b>Machine learning → Export dataset</b> writes a training set.",
             target=menu("File"), enter=lambda c: (_figure(c) is not None and _figure(c).reject())),
    ])


TUTORIALS: list[Tutorial] = [GETTING_STARTED, MEASURE, SURFACE, BUILDER, RIBBONS, PUBLICATION]


def by_key(key: str):  # noqa: ANN201
    return next((t for t in TUTORIALS if t.key == key), None)


__all__ = ["TUTORIALS", "OPEN_IMAGE", "by_key"]
