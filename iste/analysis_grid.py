"""T27 -- Phase 3 fixed hourly analysis-timestamp grid (T0..T72).

Pure timestamp-grid generation only. This module contains no particle/KDE/scientific
logic -- it produces the fixed set of historical query timestamps that T28+ (and the
already-existing T14 `iste.particle_query.query_particle_slice`) evaluate. It never reruns
or touches Stage III transport (`iste.step`/`iste.trajectory`/`iste.ensemble_run` are not
imported here), and it never interpolates particle state itself.

Per README.md/CLAUDE.md: the physical Stage III integration timestep is a completely
separate parameter (see `iste.ensemble_run.run_ensemble`'s own `dt_seconds`/`n_steps`) and
is not read, assumed, or constrained by this function. If the physical checkpoint spacing
is coarser or finer than the analysis interval, a caller queries this grid's timestamps via
the existing T14 `query_particle_slice`, which already interpolates between checkpoints --
no second interpolation implementation exists here.
"""

from __future__ import annotations

import numpy as np


def generate_analysis_grid(
    t_obs,
    window_hours: float = 72.0,
    interval_hours: float = 1.0,
) -> np.ndarray:
    """Return the fixed backward analysis timestamp grid T0..T_{window_hours}.

    T0 = t_obs (the observed slick, k=0). T_k = t_obs - k * interval_hours hours, for
    k = 0, 1, ..., window_hours / interval_hours (both endpoints included).

    Parameters
    ----------
    t_obs : np.datetime64-convertible
        The observation time, T0.
    window_hours : float
        Total backward window length in hours (default 72.0, i.e. the locked 3-day Phase 3
        window). Not hard-coded as a constant elsewhere -- always an explicit parameter.
    interval_hours : float
        Analysis interval in hours (default 1.0, the locked Phase 3 resolution).

    Returns
    -------
    np.ndarray of datetime64[ns], strictly descending (backward in time), length
    `round(window_hours / interval_hours) + 1`, with `grid[0] == t_obs` and
    `grid[-1] == t_obs - window_hours hours`.

    Raises
    ------
    ValueError
        If `window_hours` is not (to floating-point tolerance) an exact multiple of
        `interval_hours`, or if either is not strictly positive.
    """
    if window_hours <= 0:
        raise ValueError(f"window_hours must be > 0, got {window_hours}")
    if interval_hours <= 0:
        raise ValueError(f"interval_hours must be > 0, got {interval_hours}")

    n_steps_float = window_hours / interval_hours
    n_steps = round(n_steps_float)
    if abs(n_steps_float - n_steps) > 1e-9:
        raise ValueError(
            f"window_hours ({window_hours}) must be an exact multiple of interval_hours "
            f"({interval_hours}); got {n_steps_float} steps, which is not an integer"
        )

    t_obs = np.datetime64(t_obs, "ns")
    interval_ns = np.timedelta64(int(round(interval_hours * 3600 * 1e9)), "ns")
    offsets = np.arange(n_steps + 1, dtype=np.int64)
    return t_obs - offsets * interval_ns
