"""Individual steganalysis detectors.

Each detector exposes ``analyse_plane(plane, ...) -> Result`` where *plane*
is a 2-D uint8 array, and the returned :class:`Result` carries an
estimated embedding rate plus enough diagnostics to judge it.
"""

from .result import Result

__all__ = ["Result"]
