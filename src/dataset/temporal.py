"""Pure temporal-encoding inputs for the P1a encoder (methodology 2.2)."""

from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86_400
# Unix epoch (1970-01-01) was a Thursday => day-of-week index offset.
_EPOCH_DOW_OFFSET = 4


def time_of_day_frac(t_unix: np.ndarray) -> np.ndarray:
    """Fraction of the day in [0, 1)."""
    return (np.asarray(t_unix, dtype=np.float64) % SECONDS_PER_DAY) / SECONDS_PER_DAY


def hour_of_day(t_unix: np.ndarray) -> np.ndarray:
    """Integer hour 0..23 (UTC, matching the parser's timegm assumption)."""
    return ((np.asarray(t_unix, dtype=np.int64) % SECONDS_PER_DAY) // 3600).astype(np.int64)


def day_of_week(t_unix: np.ndarray) -> np.ndarray:
    """Integer day-of-week 0..6 (0 = Monday)."""
    days = np.asarray(t_unix, dtype=np.int64) // SECONDS_PER_DAY
    return ((days + _EPOCH_DOW_OFFSET) % 7).astype(np.int64)
