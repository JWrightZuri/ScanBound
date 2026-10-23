"""
Interactive tutorials: a step card next to the real widgets and a spotlight on the one to use.

    engine     Step, Tutorial, Guide (the runner) and target helpers (menu, dock, button, …)
    overlay    Spotlight (dims the window around the target) and StepCard (text and buttons)
    content    the tutorials themselves, as data: TUTORIALS
    chooser    Help → Tutorials… dialog and the first-launch welcome
    sample     the practice image the getting-started tour opens

A tutorial is a list of Steps.  Each step names a target (a function of the main window that finds
the widget, menu or rectangle to point at), the text, and optionally a ``done`` check that ticks the
step off when the user has done it and a ``show`` action that does it for them.  New tutorials only
need a new entry in ``content.TUTORIALS``; nothing in the widgets knows about the tutorials.
"""

from .content import TUTORIALS
from .engine import Guide, Step, Tutorial

__all__ = ["Guide", "Step", "Tutorial", "TUTORIALS"]
