"""Shared result container for a single detector on a single plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Result:
    method: str
    rate: float                    # estimated embedding rate; NaN if not measurable
    n_obs: int = 0                 # histogram bins / groups entering the estimate
    p_uniform: float = float("nan")  # Westfeld-style uniformity p-value
    note: str = ""
    details: dict = field(default_factory=dict)
    # True when the estimator's own model has stopped tracking the payload:
    # SPA's (Y-X)/P grows once enough bits are embedded, and chi-square's
    # smooth-envelope predictor disappears once the histogram flattens.  A
    # saturated estimate must not constrain the cross-method consensus.
    model_saturated: bool = False

    @property
    def valid(self) -> bool:
        """Whether the estimate has any support behind it.

        The bar is deliberately low (4 histogram pairs): on a small, smooth
        JPEG the coefficient histogram may genuinely have only a handful of
        usable pairs, and refusing all of them would report "no estimate"
        rather than "an estimate with wide error bars".  Callers threshold
        confidence on ``n_obs`` themselves.
        """
        return not math.isnan(self.rate) and self.n_obs >= 4

    @property
    def full_plane(self) -> bool:
        """Whether the image looks full enough that the estimate vanished.

        When the whole LSB plane is payload the histogram pair differences
        go to zero, the smooth-envelope predictor loses its signal, and the
        quantitative estimate cannot be computed -- but the classic
        uniformity test is at its strongest.  Checking this lets a caller
        fall back on ``p_uniform`` instead of reading a missing number as
        "nothing here".
        """
        return (not self.valid) and not math.isnan(self.p_uniform) \
            and self.p_uniform > 0.5
