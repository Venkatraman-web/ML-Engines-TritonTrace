"""TASK AIS-4 -- aggregate AIS vessel evidence from AIS-3's raw matches
(TRANSPARENT EVIDENCE METRICS ONLY -- no ranking, no AIS score, no source choice).

Converts AIS-3's raw (candidate, AIS observation) match pairs into per-vessel and
per-checkpoint DESCRIPTIVE evidence tables. **This module does not rank vessels, does
not compute a combined/weighted AIS score, does not choose a source vessel, and is
never imported by `iste.phase3_ais` (T35) or `iste.phase3_composite` (T36).** Every
number here is a plain descriptive statistic (count, fraction, min/median/percentile,
run-length) of the already-computed AIS-2/AIS-3 outputs -- nothing is combined into a
single figure of merit, per this task's own explicit scope.

**Exact timestamp matches are the PRIMARY evidence basis** (this task's own
instruction); `±1h`/`±2h` summaries are computed as SEPARATE, clearly namespaced
column groups from the same underlying AIS-3 match table -- never blended together.

**Observation-count bias (this task's own instruction 3):** raw match counts
(`exact_n_within_*km`) are always reported ALONGSIDE a count-normalized fraction
(`exact_frac_within_*km = exact_n_within_*km / exact_n_matches`) so a vessel is never
made to look like stronger evidence purely because it has more AIS observations. The
per-checkpoint COVERAGE fraction (`fraction_candidate_timestamps_covered_*`) is
likewise normalized by the FIXED number of candidate timestamps (not by the vessel's
own observation count), so missing AIS reports reduce the numerator naturally without
any explicit penalty term -- missing AIS is reported as missing information, never as
negative evidence (this task's own instruction 4).

**No AIS position is ever interpolated.** Every distance/evidence figure here is
derived exclusively from already-matched real AIS-3 rows (themselves built from real,
already-recorded AIS-2 observations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from iste.ais_matching import DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM, DEFAULT_DISTANCE_BINS_KM, haversine_distance_km

# Explicit, documented, configurable -- never hidden constants.
DEFAULT_N_CIRCULAR_SHIFTS = "exhaustive"  # see `run_circular_shift_null`: exhaustive means every non-zero shift on the vessel's own checkpoint grid is tested, not a random Monte Carlo subsample -- fully deterministic/reproducible, no random seed needed.


def _tolerance_slice(matches_df: pd.DataFrame, level: str) -> pd.DataFrame:
    """`level` in {'exact', 'within_1h', 'within_2h'} -- returns the matching subset of
    `matches_df`. `within_1h`/`within_2h` are NOT exclusive of `exact` (a within-1h
    match set already contains every exact match) -- this mirrors AIS-3's own explicit
    `is_exact`/`is_within_1h`/`is_within_2h` boolean columns, reused unchanged here."""
    col = {"exact": "is_exact", "within_1h": "is_within_1h", "within_2h": "is_within_2h"}[level]
    return matches_df[matches_df[col]]


def _longest_consecutive_run(sorted_t_s: np.ndarray) -> tuple[int, np.datetime64 | None, np.datetime64 | None]:
    """Longest run of CONSECUTIVE (exactly 1h-apart) entries in a sorted, de-duplicated
    array of candidate timestamps -- and the [first, last] timestamp of that run.
    Returns (0, None, None) for an empty input."""
    if len(sorted_t_s) == 0:
        return 0, None, None
    if len(sorted_t_s) == 1:
        return 1, sorted_t_s[0], sorted_t_s[0]
    deltas_h = np.diff(sorted_t_s).astype("timedelta64[s]").astype(np.float64) / 3600.0
    best_run, best_start, best_end = 1, 0, 0
    run, run_start = 1, 0
    for i, d in enumerate(deltas_h):
        if abs(d - 1.0) < 1e-6:
            run += 1
        else:
            run, run_start = 1, i + 1
        if run > best_run:
            best_run, best_start, best_end = run, run_start, run_start + run - 1
    return best_run, sorted_t_s[best_start], sorted_t_s[best_end]


def _one_tolerance_summary(vm: pd.DataFrame, n_candidate_timestamps: int, *, distance_bins_km, close_encounter_km: float) -> dict:
    """All descriptive metrics for ONE vessel at ONE tolerance level (item 1/2). `vm`
    is already filtered to that tolerance level's matching subset for one vessel."""
    prefix_out: dict = {}
    n_matches = len(vm)
    prefix_out["n_matches"] = n_matches
    distinct_cand_ts = np.array(sorted(pd.to_datetime(vm["candidate_t_s"]).unique())) if n_matches else np.array([], dtype="datetime64[ns]")
    prefix_out["n_candidate_timestamps_covered"] = len(distinct_cand_ts)
    prefix_out["fraction_candidate_timestamps_covered"] = len(distinct_cand_ts) / n_candidate_timestamps

    if n_matches == 0:
        prefix_out.update(dict(
            min_distance_km=float("nan"), median_distance_km=float("nan"), p90_distance_km=float("nan"),
            longest_consecutive_close_encounter_checkpoints=0,
            first_close_encounter_t_s=None, last_close_encounter_t_s=None,
            best_encounter_ais_t_s=None, best_encounter_candidate_t_s=None, best_encounter_distance_km=float("nan"),
            best_encounter_candidate_id=None, best_encounter_candidate_support=float("nan"), best_encounter_candidate_cluster_id=None,
        ))
        for b in distance_bins_km:
            prefix_out[f"n_within_{b:g}km"] = 0
            prefix_out[f"frac_within_{b:g}km"] = float("nan")
        return prefix_out

    dist = vm["distance_km"].to_numpy()
    prefix_out["min_distance_km"] = float(np.min(dist))
    prefix_out["median_distance_km"] = float(np.median(dist))
    prefix_out["p90_distance_km"] = float(np.percentile(dist, 90))

    for b in distance_bins_km:
        n_in = int(np.sum(dist <= b))
        prefix_out[f"n_within_{b:g}km"] = n_in
        prefix_out[f"frac_within_{b:g}km"] = n_in / n_matches  # count-normalized: prevents observation-count bias (item 3)

    close = vm[vm["distance_km"] <= close_encounter_km]
    close_t_s = np.array(sorted(pd.to_datetime(close["candidate_t_s"]).unique())) if len(close) else np.array([], dtype="datetime64[ns]")
    run_len, run_first, run_last = _longest_consecutive_run(close_t_s)
    prefix_out["longest_consecutive_close_encounter_checkpoints"] = run_len if len(close_t_s) else 0
    prefix_out["first_close_encounter_t_s"] = close_t_s[0] if len(close_t_s) else None
    prefix_out["last_close_encounter_t_s"] = close_t_s[-1] if len(close_t_s) else None

    best = vm.loc[vm["distance_km"].idxmin()]
    prefix_out["best_encounter_ais_t_s"] = best["ais_t_s"]
    prefix_out["best_encounter_candidate_t_s"] = best["candidate_t_s"]
    prefix_out["best_encounter_distance_km"] = float(best["distance_km"])
    prefix_out["best_encounter_candidate_id"] = best["candidate_id"]
    prefix_out["best_encounter_candidate_support"] = float(best["candidate_support"])
    prefix_out["best_encounter_candidate_cluster_id"] = best["candidate_cluster_id"]
    return prefix_out


def compute_vessel_evidence(
    matches_df: pd.DataFrame,
    ais_df: pd.DataFrame,
    gap_summary_df: pd.DataFrame,
    n_candidate_timestamps: int,
    *,
    distance_bins_km=DEFAULT_DISTANCE_BINS_KM,
    close_encounter_km: float = DEFAULT_CLOSE_ENCOUNTER_DISTANCE_KM,
) -> pd.DataFrame:
    """TASK AIS-4 items 1-4: one row per vessel present in `ais_df` (the AIS-2 event-
    window+AOI dataset -- every vessel included, even with zero matches, never
    silently excluded), with THREE separate, clearly namespaced (`exact_*`,
    `within_1h_*`, `within_2h_*`) column groups, plus AIS-2 data-quality fields
    (`n_observations_total`, `median_interval_hours`, `max_interval_hours`,
    `n_gaps_over_threshold`) preserved unchanged from the AIS-2 output."""
    all_vessels = ais_df[["mmsi", "vessel_name", "vessel_type", "imo"]].drop_duplicates(subset="mmsi")
    gap_by_mmsi = gap_summary_df.set_index("mmsi")

    rows = []
    for _, v in all_vessels.iterrows():
        mmsi = int(v["mmsi"])
        row: dict = dict(mmsi=mmsi, vessel_name=v["vessel_name"], vessel_type=v["vessel_type"], imo=int(v["imo"]))

        vm_all = matches_df[matches_df["mmsi"] == mmsi]
        for level, tag in [("exact", "exact"), ("within_1h", "within_1h"), ("within_2h", "within_2h")]:
            vm = _tolerance_slice(vm_all, level)
            sub = _one_tolerance_summary(vm, n_candidate_timestamps, distance_bins_km=distance_bins_km, close_encounter_km=close_encounter_km)
            for k, val in sub.items():
                row[f"{tag}_{k}"] = val

        if mmsi in gap_by_mmsi.index:
            g = gap_by_mmsi.loc[mmsi]
            row["n_observations_total"] = int(g["n_observations"])
            row["median_interval_hours"] = g["median_interval_hours"]
            row["max_interval_hours"] = g["max_interval_hours"]
            row["n_gaps_over_threshold"] = int(g["n_gaps_over_threshold"])
        else:
            row["n_observations_total"] = int((ais_df["mmsi"] == mmsi).sum())
            row["median_interval_hours"] = float("nan")
            row["max_interval_hours"] = float("nan")
            row["n_gaps_over_threshold"] = 0

        row["data_availability_note"] = (
            f"{row['n_observations_total']} AIS observations total; "
            f"{row['exact_n_candidate_timestamps_covered']}/{n_candidate_timestamps} candidate timestamps have an exact-time AIS report. "
            "Unmatched candidate timestamps reflect NO AIS report at that instant -- missing information, not evidence against proximity."
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("exact_min_distance_km", na_position="last").reset_index(drop=True)


def compute_checkpoint_proximity(
    candidates_df: pd.DataFrame, matches_df: pd.DataFrame
) -> pd.DataFrame:
    """One row per Phase 3 candidate CHECKPOINT (distinct `t_s`) -- the complementary
    view to `compute_vessel_evidence`'s per-vessel table. Uses EXACT matches only (the
    primary basis, per this task's own instruction), so this table shows, at every
    checkpoint, how many vessels actually reported an AIS position at that exact
    instant and how close the single best (candidate, vessel) pair was."""
    exact = matches_df[matches_df["is_exact"]]
    rows = []
    for t_s, cg in candidates_df.groupby("t_s"):
        em = exact[exact["candidate_t_s"] == t_s]
        row = dict(
            t_s=t_s, n_candidates_at_checkpoint=len(cg),
            n_vessels_with_exact_observation=int(em["mmsi"].nunique()),
        )
        if len(em):
            best = em.loc[em["distance_km"].idxmin()]
            row["min_distance_km"] = float(best["distance_km"])
            row["best_mmsi"] = int(best["mmsi"])
            row["best_vessel_name"] = best["vessel_name"]
            row["best_candidate_id"] = best["candidate_id"]
            row["best_candidate_support"] = float(best["candidate_support"])
            row["best_candidate_cluster_id"] = best["candidate_cluster_id"]
        else:
            row["min_distance_km"] = float("nan")
            row["best_mmsi"] = None
            row["best_vessel_name"] = None
            row["best_candidate_id"] = None
            row["best_candidate_support"] = float("nan")
            row["best_candidate_cluster_id"] = None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("t_s").reset_index(drop=True)


@dataclass(frozen=True)
class CircularShiftNullResult:
    """TASK AIS-4 item 5: one vessel's empirical-null diagnostic result. Purely
    descriptive -- `empirical_p_value` is NOT a source probability (explicitly
    prohibited by this task); it only answers "how unusual is this vessel's REAL
    (unshifted) minimum synchronized distance, compared with every OTHER possible
    circular time-alignment of this SAME vessel's own real trajectory against the
    real candidate timeline." See module docstring / `run_circular_shift_null`'s own
    docstring for the method's documented limitations."""

    mmsi: int
    real_min_distance_km: float
    n_shifts_tested: int
    null_min_distances_km: np.ndarray
    null_median_km: float
    null_p10_km: float
    empirical_p_value: float  # fraction of null replicates with min_distance <= real_min_distance


def run_circular_shift_null(
    mmsi: int,
    candidates_df: pd.DataFrame,
    ais_df: pd.DataFrame,
    candidate_t_s_grid: list,
) -> CircularShiftNullResult | None:
    """TASK AIS-4 item 5 -- circular time-shift permutation null.

    **Method:** builds a length-`len(candidate_t_s_grid)` array of the vessel's own
    EXACT-timestamp position at each candidate checkpoint (`None` where the vessel has
    no exact-time AIS report there -- never interpolated). The REAL statistic is the
    minimum great-circle distance, over all checkpoints where the vessel has a real
    position, to the closest candidate AT THAT SAME checkpoint. The NULL distribution
    is built by circularly shifting this same real position array by every one of the
    `n-1` non-zero offsets (an EXHAUSTIVE permutation, not a random Monte Carlo sample
    -- fully deterministic and reproducible, no seed needed) and recomputing the same
    statistic against the SAME real candidate positions at each checkpoint. This
    preserves the vessel's own real trajectory shape/autocorrelation exactly, testing
    only whether its SPECIFIC real temporal alignment to the candidate timeline is
    unusual compared to other equally-plausible phase alignments of that same
    trajectory.

    **Documented limitations (this task's own instruction: "document its
    limitations"):**
    - Tests TEMPORAL phase only -- does not test whether the vessel's spatial ROUTE
      itself is unusual (e.g. compared to other vessels' routes).
    - The circular wrap-around treats the 72h window as a closed loop; a vessel that
      enters/exits the AOI partway through the window may have shifted replicates
      that place it outside the AOI without that being flagged specially.
    - No multiple-testing correction is applied across the (up to) 264 vessels tested
      -- a small `empirical_p_value` for any ONE vessel, found this way, is expected
      to occur somewhat often by chance across so many vessels; this diagnostic does
      NOT correct for that, and does not claim statistical significance on its own.
    - `empirical_p_value` is a descriptive comparison against this ONE null
      construction only -- it is NOT, and must never be read as, a probability that
      the vessel is the true source.

    Returns `None` if the vessel has fewer than 2 exact-timestamp observations on the
    candidate grid (a null distribution is not meaningfully definable).
    """
    grid = list(candidate_t_s_grid)
    n = len(grid)
    t_s_to_idx = {pd.Timestamp(t): i for i, t in enumerate(grid)}

    vessel_pos = [None] * n
    vais = ais_df[ais_df["mmsi"] == mmsi]
    for _, r in vais.iterrows():
        t = pd.Timestamp(r["timestamp"])
        if t in t_s_to_idx:
            vessel_pos[t_s_to_idx[t]] = (float(r["lat"]), float(r["lon"]))

    n_present = sum(1 for p in vessel_pos if p is not None)
    if n_present < 2:
        return None

    cand_by_idx: list[tuple[np.ndarray, np.ndarray]] = []
    for t in grid:
        cg = candidates_df[candidates_df["t_s"] == t]
        cand_by_idx.append((cg["lat"].to_numpy(), cg["lon"].to_numpy()))

    def _min_distance_for_shift(shift: int) -> float:
        best = np.inf
        for i in range(n):
            src_idx = (i - shift) % n
            p = vessel_pos[src_idx]
            if p is None:
                continue
            clat, clon = cand_by_idx[i]
            if len(clat) == 0:
                continue
            d = haversine_distance_km(p[0], p[1], clat, clon)
            m = float(np.min(d))
            if m < best:
                best = m
        return best

    real_min = _min_distance_for_shift(0)
    null_vals = np.array([_min_distance_for_shift(s) for s in range(1, n)])
    null_vals = null_vals[np.isfinite(null_vals)]

    if len(null_vals) == 0:
        return None

    p_value = float(np.mean(null_vals <= real_min)) if np.isfinite(real_min) else float("nan")

    return CircularShiftNullResult(
        mmsi=mmsi, real_min_distance_km=real_min, n_shifts_tested=len(null_vals),
        null_min_distances_km=null_vals, null_median_km=float(np.median(null_vals)),
        null_p10_km=float(np.percentile(null_vals, 10)), empirical_p_value=p_value,
    )
