"""Switches for parts of the app that are kept in the code but turned off.

REVIEW: the proposal review workflow (Review panel, accept / reject keys, "Mark image reviewed",
the "Review adapt results" option and the review state in the Files panel).  While it is off,
detector, model and surface results are added as accepted annotations straight away (any proposals
already in a project are accepted when it is opened); labels and outlines are corrected by editing
them.  Set it to True to bring the workflow back as it was.
"""

REVIEW = False
