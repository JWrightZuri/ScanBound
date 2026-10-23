"""
Crystal-surface analysis: lattices, adsorption geometry, defects, assembly, registration, tracking.

Modules (import what you need; nothing here needs Qt):

``lattice``         FFT peaks, site detection, lattice fit with covariance, Bravais candidates, calibration
``distortion``      local deformation gradient: strain, rotation, dilatation maps
``domains``         rotation and translational domains, boundaries
``superstructure``  periodicities beyond the lattice, matrix and Wood notation, moiré formulas
``structure``       surface models (presets, ASE import, JSON), sites, alignment to a fitted lattice
``objects``         molecules from annotations (position, orientation, handedness, uncertainty)
``adsorption``      site assignment with probabilities, positions in the cell, occupancy
``statistics``      orientation families, chirality, group comparisons, randomized references
``defects``         vacancies, adatoms, complexes, environments, line defects
``steps``           step edges and terraces
``assembly``        neighbour graph, g(r), Clark–Evans, bond order, chains, motif search
``registration``    image registration and object linking between images
``features``        shape descriptors, cross-channel features, clustering, templates
``tracking``        following objects through frames: moves, rotations, transformations, diffusion
``quality``         scan artefacts and tip-state checks
``hypotheses``      ranking candidate explanations of one object with their evidence
``builder``         3D models: molecule .xyz files on the annotations, on a substrate slab; XYZ export
``store``           where all of this lives in a project
``workflow``        the integrated analyses the GUI runs, each returning a Report
``synthetic``       synthetic lattice images with known truth (tests, demonstrations)
"""
