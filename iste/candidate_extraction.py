"""T17 -- Phase 2 candidate peak and candidate-core extraction.

Implements exactly `README.md` Sec. 16 at one historical timestamp `t_s`, working from a
T16 Phase 2 support map (a list of `iste.support_features.SupportFeatures`, one per point
of an explicit regular `lat_grid` x `lon_grid`):

    1. identify local maxima of the support heat map;
    2. suppress near-duplicate maxima;
    3. retain multiple spatial modes;
    4. attach timestamp t_s to each candidate;
    5. define a high-support candidate core around each maximum.

"The entire KDE is not automatically treated as the source" (`README.md` Sec. 16) -- this
module extracts a small set of discrete candidate peaks/cores from the support map; it
does not itself do candidate ranking, core-particle sampling, forward propagation, AIS,
likelihood evaluation, posterior inference, or source attribution, and it never touches
Stage III's `c_hat*` (this module has no knowledge of it at all -- only of T16's `D`/`W`/
`V`/`Support` features).

**Local-maximum detection** reuses `scipy.ndimage.maximum_filter` (an existing project
dependency, not a hand-rolled peak-finder) over the support values reshaped onto the
caller-supplied regular grid.

**Near-duplicate suppression** is a greedy, support-descending, non-maximum suppression:
process candidate peaks from highest support to lowest, keeping a peak only if it is
farther than `min_separation_m` (an explicit, configurable metres threshold) from every
peak already kept. Distances are computed via `iste.kdtree_neighborhood.
project_to_local_metres` (T15's own distance-safe local projection) -- never raw lat/lon
degree differences.

**Candidate cores** are, in this prototype, a fixed-radius circle (`core_radius_m`,
explicit/configurable) around each retained peak -- the simplest, smallest-complexity
core definition satisfying "define an explicit configurable high-support candidate core"
(`README.md` Sec. 16). A more elaborate contour/support-threshold-based core is a
possible future refinement, not implemented here (see the module's known limitations in
`progress.md`).

**Provenance for later core sampling (T18/T19)** is preserved by attaching each
candidate's own `SupportFeatures` (`peak_support_features`) -- which itself carries the
full T15 `NeighborhoodResult` (`particle_id`/`observation_pixel_id`/`forcing_member_id`
of every particle that contributed to the peak's own evaluated neighborhood) -- to the
returned `Candidate`. A later core-sampling task with access to the same
`ParticleNeighborhoodIndex` can re-query at `core_region`'s own radius if that differs
from whatever radius/k the support map was originally evaluated with.

No particle sampling, forward transport, AIS, likelihood, or posterior inference is
implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from iste.kdtree_neighborhood import project_to_local_metres
from iste.support_features import SupportFeatures, compute_support_map


@dataclass(frozen=True)
class CandidateExtractionConfig:
    """PROTOTYPE / CONFIGURATION -- every field here is an explicit Phase 2 parameter
    (`README.md` Sec. 26: "candidate peak threshold", "candidate core definition",
    "candidate merge/separation threshold"), not a physically validated constant.

    - `peak_neighborhood_cells`: odd integer `>= 3`, the side length (in GRID CELLS) of
      the square neighborhood compared for local-maximum detection (default `3` ->
      immediate 8-connected neighbors).
    - `min_support`: a cell must exceed this support value to be considered a candidate
      peak at all (default `-inf`, i.e. no explicit floor beyond the "must have at least
      one contributing particle" check below).
    - `min_separation_m`: minimum metres between two retained peaks (non-maximum
      suppression); must be `>= 0`. `0.0` (the default) suppresses nothing beyond exact
      coincident peaks.
    - `core_radius_m`: the fixed radius (metres) defining every candidate's
      `core_region`; must be `> 0`.
    """

    peak_neighborhood_cells: int = 3
    min_support: float = float("-inf")
    min_separation_m: float = 0.0
    core_radius_m: float = 1000.0

    def __post_init__(self) -> None:
        if self.peak_neighborhood_cells < 3 or self.peak_neighborhood_cells % 2 == 0:
            raise ValueError(
                f"peak_neighborhood_cells must be an odd integer >= 3; got {self.peak_neighborhood_cells!r}"
            )
        if np.isnan(self.min_support):
            raise ValueError("min_support must not be NaN (-inf, the 'no explicit floor' default, is fine)")
        if not np.isfinite(self.min_separation_m) or self.min_separation_m < 0.0:
            raise ValueError(f"min_separation_m must be finite and >= 0; got {self.min_separation_m!r}")
        if not np.isfinite(self.core_radius_m) or self.core_radius_m <= 0.0:
            raise ValueError(f"core_radius_m must be finite and > 0; got {self.core_radius_m!r}")


@dataclass(frozen=True)
class CoreRegion:
    """An explicit, simple high-support candidate core: a fixed-radius circle around the
    candidate peak. PROTOTYPE definition, not a claim of physical/statistical
    optimality -- `README.md` leaves the exact core definition as an open Phase 2
    configuration choice."""

    center_lat: float
    center_lon: float
    radius_m: float


@dataclass
class Candidate:
    """One Phase 2 candidate source region at one historical timestamp -- a screening
    representation (`README.md` Sec. 16), never itself the formal source.

    `x_peak`/`y_peak` follow `README.md`'s own `(x, y)` naming (`x` = eastward = the
    peak's longitude, `y` = northward = the peak's latitude)."""

    candidate_id: str
    x_peak: float
    y_peak: float
    t_s: np.datetime64
    core_region: CoreRegion
    support_score: float
    density_score: float
    weighted_support_score: float
    pixel_diversity_score: float
    peak_support_features: SupportFeatures


def _metres_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance-safe metre distance between two lat/lon points, reusing T15's own local
    projection (`project_to_local_metres`) rather than a new/duplicate distance formula."""
    x, y = project_to_local_metres(lat2, lon2, lat_ref=lat1, lon_ref=lon1)
    return float(np.sqrt(x[0] ** 2 + y[0] ** 2))


def find_candidate_peaks(
    support_map: list[SupportFeatures],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    t_s,
    config: CandidateExtractionConfig | None = None,
) -> list[Candidate]:
    """Extract candidate peaks/cores from a T16 Phase 2 support map at one historical
    timestamp `t_s`.

    Parameters
    ----------
    support_map : list[SupportFeatures]
        T16's `compute_support_map` output, evaluated at every point of the regular grid
        `lat_grid x lon_grid`, in row-major order (`support_map[i*len(lon_grid) + j]`
        corresponds to `(lat_grid[i], lon_grid[j])`) -- exactly the order
        `iste.support_features.compute_support_map` produces when given a
        `numpy.meshgrid(lon_grid, lat_grid, indexing="xy")`-style flattened query set
        (see `find_candidate_peaks_from_index` below, or this module's own tests, for a
        worked construction).
    lat_grid, lon_grid : array-like
        The 1-D coordinate arrays defining the regular grid `support_map` was evaluated
        on. Must satisfy `len(support_map) == len(lat_grid) * len(lon_grid)`.
    t_s : datetime64-convertible
        The historical timestamp every returned candidate is attached to.
    config : CandidateExtractionConfig, optional
        Defaults to `CandidateExtractionConfig()`.

    Returns
    -------
    list[Candidate]
        Zero or more candidates, one per retained (non-suppressed) local maximum, sorted
        by descending `support_score`. Returns `[]` for an empty/all-invalid/all-empty
        support map -- no candidate is ever fabricated from a point with zero
        contributing particles (`density_score == 0`), regardless of `min_support`.
    """
    if config is None:
        config = CandidateExtractionConfig()

    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    n_lat, n_lon = len(lat_grid), len(lon_grid)

    if len(support_map) != n_lat * n_lon:
        raise ValueError(
            f"support_map has {len(support_map)} entries but lat_grid x lon_grid "
            f"implies {n_lat} x {n_lon} = {n_lat * n_lon}"
        )
    if n_lat == 0 or n_lon == 0:
        return []

    t_s = np.datetime64(t_s, "ns")

    support_grid = np.array([sf.support for sf in support_map], dtype=np.float64).reshape(n_lat, n_lon)
    density_grid = np.array([sf.density for sf in support_map], dtype=np.float64).reshape(n_lat, n_lon)

    local_max = maximum_filter(support_grid, size=config.peak_neighborhood_cells, mode="nearest")
    is_peak = (
        (support_grid == local_max)
        & np.isfinite(support_grid)
        & (support_grid > config.min_support)
        & (density_grid > 0.0)  # never a candidate from an empty (no-particle) neighborhood
    )

    peak_rows, peak_cols = np.nonzero(is_peak)
    raw_peaks = []
    for r, c in zip(peak_rows.tolist(), peak_cols.tolist()):
        idx = r * n_lon + c
        sf = support_map[idx]
        raw_peaks.append((sf.support, r, c, sf))

    # Stable sort by descending support -- ties preserve np.nonzero's deterministic
    # (row-major) scan order, so repeated runs on the same input are always identical.
    raw_peaks.sort(key=lambda item: -item[0])

    kept: list[tuple[float, int, int, SupportFeatures]] = []
    for support, r, c, sf in raw_peaks:
        lat, lon = lat_grid[r], lon_grid[c]
        too_close = any(
            _metres_between(lat, lon, lat_grid[kr], lon_grid[kc]) < config.min_separation_m
            for _, kr, kc, _ in kept
        )
        if not too_close:
            kept.append((support, r, c, sf))

    candidates: list[Candidate] = []
    t_s_str = np.datetime_as_string(t_s, unit="s")
    for i, (support, r, c, sf) in enumerate(kept):
        lat, lon = float(lat_grid[r]), float(lon_grid[c])
        candidates.append(
            Candidate(
                candidate_id=f"{t_s_str}_cand{i:03d}",
                x_peak=lon,
                y_peak=lat,
                t_s=t_s,
                core_region=CoreRegion(center_lat=lat, center_lon=lon, radius_m=config.core_radius_m),
                support_score=sf.support,
                density_score=float(sf.density),
                weighted_support_score=sf.weighted_support,
                pixel_diversity_score=float(sf.pixel_diversity),
                peak_support_features=sf,
            )
        )
    return candidates


def find_candidate_peaks_from_index(
    index,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    t_s,
    *,
    radius_m: float | None = None,
    k: int | None = None,
    support_config=None,
    combine_fn=None,
    extraction_config: CandidateExtractionConfig | None = None,
) -> list[Candidate]:
    """Convenience wrapper: build the T16 support map over `lat_grid x lon_grid` via
    `iste.support_features.compute_support_map` (using T15's `index` directly), then run
    `find_candidate_peaks` on it. Provided for integration convenience only -- the core
    peak/core-extraction LOGIC lives entirely in `find_candidate_peaks`, which works from
    an already-computed support map alone, per this task's own scope."""
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)  # row-major: shape (n_lat, n_lon)
    kwargs = {}
    if combine_fn is not None:
        kwargs["combine_fn"] = combine_fn
    support_map = compute_support_map(
        index, lat_mesh.ravel(), lon_mesh.ravel(), radius_m=radius_m, k=k, config=support_config, **kwargs
    )
    return find_candidate_peaks(support_map, lat_grid, lon_grid, t_s, config=extraction_config)
