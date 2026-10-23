"""Exceptions raised by the detection core."""

from __future__ import annotations


class ScanBoundError(Exception):
    """Base class of every error raised by :mod:`scanbound`."""


class ConfigError(ScanBoundError, ValueError):
    """A configuration (pipeline, ROI or labelling settings) is invalid."""


class UnknownStepError(ConfigError, KeyError):
    """A pipeline names a step that is not in the registry."""

    def __str__(self) -> str:          # KeyError would quote the message
        return str(self.args[0]) if self.args else ""


class ParamError(ConfigError):
    """A step parameter has the wrong type or an invalid value."""


class PipelineError(ScanBoundError, RuntimeError):
    """A pipeline step failed while running in strict mode."""

    def __init__(self, index: int, step: str, message: str):
        super().__init__(f"step {index} ({step}): {message}")
        self.index = index
        self.step = step
        self.message = message
