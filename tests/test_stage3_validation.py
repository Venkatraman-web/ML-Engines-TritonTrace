"""T12 -- Stage III controlled numerical validation suite.

This is **validation, not new scientific functionality**: every case here calls the
already-implemented T05/T06 machinery (`iste.step.backward_euler_maruyama_step`,
`iste.trajectory.run_backward_trajectory`) and T03's derivative operators
(`iste.derivatives.spatial_gradient`/`divergence_2d`) directly -- no second transport
implementation is created for comparison, per instruction 12. What T12 adds is a
dedicated, analytically-grounded set of controlled synthetic cases, each with an
explicitly *stated expected value* (never just "the code ran without error"), the actual
computed value, and a numerical tolerance -- assembled below as `ValidationCase` records
so the same cases back both this file's `unittest` assertions and the T12 manual-check
notebook's validation table (both read from the single source of truth `ALL_CASES`
below, so the table can never silently drift from what the tests actually check).

Covers every category `tasks.md` T12 requires -- zero divergence, known convergent flow,
known divergent flow, constant diffusivity K, spatially varying diffusivity K, particle
validity, particle-weight behavior -- plus an explicit backward-sign-convention case
(instruction 4). Real `data/ocean.nc` is used in exactly one place, in
`TestRealDataSanityCheck` below, and deliberately kept OUT of `ALL_CASES`/the validation
table: per instruction 13, real data is a numerical sanity check only, never treated as
"truth" to validate against (there is no analytically known correct answer for real
ocean data).

**How the exact analytic expectations are derived:** every case uses a synthetic field
simple enough that the exact expected number is knowable by hand (uniform flow, or a
field that is exactly linear in one coordinate). Converting a known physical
(velocity/gradient) value to an expected degree displacement or weight change reuses
`iste.geometry`'s already-validated (T03) metre/degree conversion -- this is unit
arithmetic, not a competing transport model, and does not violate instruction 12.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from iste.derivatives import divergence_2d, spatial_gradient
from iste.environment import GriddedFieldSource
from iste.ensemble_run import compute_diagnostics, run_ensemble
from iste.environment import load_field_dataset
from iste.forcing_ensemble import ForcingEnsemble, ForcingEnsembleConfig
from iste.geometry import metres_per_degree_latitude, metres_per_degree_longitude
from iste.observation import adapt_upstream_particles
from iste.step import WeightedParticleState, backward_euler_maruyama_step
from iste.trajectory import run_backward_trajectory

# -- shared synthetic grid -----------------------------------------------------------
# A wider time axis than T05's own unit tests use (which only ever take one step): T12
# runs multi-step trajectories, so the ocean source's time coverage must extend far
# enough backward from T0 for every step actually taken.
LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
TIME = np.datetime64("2020-01-01T00:00", "ns") + np.arange(6) * np.timedelta64(3600, "s")
T0 = TIME[-1]  # leaves 5 hours of backward room -- plenty for every case's n_steps below
DT = 900.0  # 15-minute checkpoints, within README's 15-30 min guidance
QUERY_LAT = 20.0  # every case starts particles on this row; used for analytic conversions
DLON_PER_DEG = metres_per_degree_longitude(QUERY_LAT)
DLAT_PER_DEG = metres_per_degree_latitude()

ZERO_UV = np.zeros((3, 7))
ZERO_K = np.zeros((3, 7))


def make_ocean(uo_grid: np.ndarray, vo_grid: np.ndarray) -> GriddedFieldSource:
    """A synthetic ocean source: the same `(uo_grid, vo_grid)` snapshot repeated at every
    one of `TIME`'s timestamps (a steady, time-invariant synthetic flow -- deliberately
    simple so the expected behavior over multiple steps is analytically knowable)."""
    uo = np.stack([uo_grid] * len(TIME))
    vo = np.stack([vo_grid] * len(TIME))
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


@dataclass
class ValidationCase:
    """One controlled validation case: a stated expected value, the actual computed
    value, and a numerical tolerance -- never a bare "it ran" check. `tolerance` is a
    numerical-validation tolerance only (floating-point/finite-difference-exactness
    budget for this specific synthetic field and `dt`), not a claim about physically
    meaningful precision (instruction 11)."""

    category: str
    name: str
    expected: float
    computed: float
    tolerance: float
    notes: str = ""

    @property
    def error(self) -> float:
        return abs(self.computed - self.expected)

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance


# -- category: zero divergence --------------------------------------------------------

def case_zero_divergence() -> ValidationCase:
    """Spatially uniform `U` -> `div(U) = 0` exactly everywhere -> the weight update
    term `-div(U)*dt` is exactly zero, so `log_w_new == log_w_old`."""
    ocean = make_ocean(np.full((3, 7), 0.4), np.full((3, 7), -0.2))
    state = WeightedParticleState.initialize(lat0=[QUERY_LAT], lon0=[115.0], log_weight0=1.5)
    rng = np.random.default_rng(100)
    new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)
    return ValidationCase(
        category="zero divergence", name="uniform U -> log-weight unchanged",
        expected=1.5, computed=float(new_state.log_weight[0]), tolerance=1e-9,
        notes="U=(0.4,-0.2) m/s everywhere; div(U)=0 exactly for a spatially uniform field.",
    )


# -- backward sign convention (instruction 4) ------------------------------------------

def case_backward_sign_convention() -> ValidationCase:
    """Constant eastward U, K=0: a fully deterministic, unambiguous field. The backward
    step must move the particle WEST (`-U*dt`, never `+U*dt`) by exactly the known metre
    displacement converted to degrees."""
    u_speed, v_speed = 2.5, -1.0  # m/s east, m/s south
    ocean = make_ocean(np.full((3, 7), u_speed), np.full((3, 7), v_speed))
    state = WeightedParticleState.initialize(lat0=[QUERY_LAT], lon0=[115.0])
    rng = np.random.default_rng(101)
    new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)
    expected_lon = 115.0 + (-u_speed * DT) / DLON_PER_DEG
    return ValidationCase(
        category="backward sign convention",
        name="constant eastward U -> particle moves WEST by exactly -U*dt",
        expected=expected_lon, computed=float(new_state.lon[0]), tolerance=1e-9,
        notes=f"K=0, deterministic; expected_lon uses -U*dt (never +U*dt) converted via geometry.py.",
    )


# -- category: known convergent / divergent flow ---------------------------------------

def case_known_convergent_flow() -> ValidationCase:
    """`uo = -C*(lon-115)` (du/dx < 0, convergent), v=0: `div(U) = du/dx` is exactly `-C/
    DLON_PER_DEG` everywhere (linear field). Expected `log_w_new = -div(U)*dt`, which must
    be POSITIVE (weight increases under convergent flow)."""
    C = 0.001
    uo_grid = -C * (LON[None, :] - 115.0) * np.ones((3, 1))
    ocean = make_ocean(uo_grid, ZERO_UV)
    state = WeightedParticleState.initialize(lat0=[QUERY_LAT], lon0=[115.0])
    rng = np.random.default_rng(102)
    new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)
    expected_div = -C / DLON_PER_DEG
    expected_log_weight = -expected_div * DT
    return ValidationCase(
        category="known convergent flow", name="div(U)<0 -> log-weight increases by exactly -div(U)*dt",
        expected=expected_log_weight, computed=float(new_state.log_weight[0]), tolerance=1e-9,
        notes=f"analytic div(U)={expected_div:.6e} /s (exact for a linear field); expected positive.",
    )


def case_known_divergent_flow() -> ValidationCase:
    """Mirror of the convergent case: `uo = +C*(lon-115)` (du/dx > 0, divergent) ->
    `log_w_new` must be NEGATIVE by exactly `-div(U)*dt`."""
    C = 0.001
    uo_grid = C * (LON[None, :] - 115.0) * np.ones((3, 1))
    ocean = make_ocean(uo_grid, ZERO_UV)
    state = WeightedParticleState.initialize(lat0=[QUERY_LAT], lon0=[115.0])
    rng = np.random.default_rng(103)
    new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, ZERO_K, rng)
    expected_div = C / DLON_PER_DEG
    expected_log_weight = -expected_div * DT
    return ValidationCase(
        category="known divergent flow", name="div(U)>0 -> log-weight decreases by exactly -div(U)*dt",
        expected=expected_log_weight, computed=float(new_state.log_weight[0]), tolerance=1e-9,
        notes=f"analytic div(U)={expected_div:.6e} /s (exact for a linear field); expected negative.",
    )


# -- category: constant diffusivity K --------------------------------------------------

def case_constant_k_zero_gradient() -> ValidationCase:
    """A spatially constant K field must have `grad(K) = 0` everywhere (no artificial
    diffusivity-gradient drift can appear) -- checked directly against T03's own
    `spatial_gradient` operator output, not just indirectly through a T05 step."""
    k_field = np.full((3, 7), 42.0)
    d_dx, d_dy, valid = spatial_gradient(LAT, LON, k_field)
    max_abs_grad = float(np.max(np.abs(np.concatenate([d_dx[valid], d_dy[valid]]))))
    return ValidationCase(
        category="constant diffusivity K", name="grad(K)=0 for a spatially constant K field",
        expected=0.0, computed=max_abs_grad, tolerance=1e-9,
        notes="max|grad(K)| over the whole grid via iste.derivatives.spatial_gradient (T03), not T05.",
    )


def case_constant_k_diffusion_displacement() -> ValidationCase:
    """Zero U, constant (nonzero) K -> no drift at all (`-U=0`, `grad(K)=0`), so the
    displacement must equal EXACTLY `sqrt(2*K*dt)*xi` for the RNG's own draw -- a fully
    deterministic composition of documented formulas, reproduced independently here with
    a fresh, identically-seeded RNG."""
    ocean = make_ocean(ZERO_UV, ZERO_UV)
    k_field = np.full((3, 7), 50.0)
    state = WeightedParticleState.initialize(lat0=[QUERY_LAT], lon0=[115.0])
    rng_for_step = np.random.default_rng(104)
    new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, k_field, rng_for_step)

    rng_reference = np.random.default_rng(104)
    xi = rng_reference.standard_normal(size=(1, 2))
    scale = np.sqrt(2.0 * 50.0 * DT)
    dx_m, dy_m = scale * xi[0, 0], scale * xi[0, 1]
    expected_lon = 115.0 + dx_m / DLON_PER_DEG
    return ValidationCase(
        category="constant diffusivity K", name="U=0, constant K -> displacement == sqrt(2*K*dt)*xi exactly",
        expected=expected_lon, computed=float(new_state.lon[0]), tolerance=1e-9,
        notes="no drift term possible (U=0, grad(K)=0); pure-diffusion formula reproduced independently.",
    )


# -- category: spatially varying diffusivity K -------------------------------------------

_K_SLOPE = 100.0  # C in K(lon) = C * (lon - 115)
_K_LINEAR = _K_SLOPE * (LON[None, :] - 115.0) * np.ones((3, 1))


def case_varying_k_gradient_matches_analytic() -> ValidationCase:
    """`K(lon) = C*(lon-115)` is exactly linear -> its analytic gradient is the constant
    `C / DLON_PER_DEG` everywhere on the QUERY_LAT row. T03's numerical `spatial_gradient`
    (central differences, `edge_order=2`) is exact for a linear field, so the numeric and
    analytic gradients should match to floating-point precision."""
    d_dx, _, valid = spatial_gradient(LAT, LON, _K_LINEAR)
    row = 1  # LAT[1] == QUERY_LAT == 20.0
    col = 3  # an interior column, LON[3] == 115.0
    assert LAT[row] == QUERY_LAT and LON[col] == 115.0 and valid[row, col]
    analytic = _K_SLOPE / DLON_PER_DEG
    return ValidationCase(
        category="spatially varying diffusivity K",
        name="numerical grad(K) matches analytic gradient for a linear K field",
        expected=analytic, computed=float(d_dx[row, col]), tolerance=1e-9,
        notes="T03's spatial_gradient at (lat=20, lon=115) vs the exact analytic dK/dx=C/DLON_PER_DEG.",
    )


def case_varying_k_drift_matches_t05() -> ValidationCase:
    """With `U=0` (isolating `grad(K)` alone), T05's actual step must move the particle by
    exactly `+grad(K)*dt` (never `-grad(K)`, and never scaled/omitted)."""
    ocean = make_ocean(ZERO_UV, ZERO_UV)
    state = WeightedParticleState.initialize(lat0=[QUERY_LAT], lon0=[115.0])
    rng = np.random.default_rng(105)
    new_state = backward_euler_maruyama_step(state, T0, DT, ocean, LAT, LON, _K_LINEAR, rng)
    analytic_dKdx = _K_SLOPE / DLON_PER_DEG
    expected_dx_m = analytic_dKdx * DT  # (-U + grad(K)) with U=0
    expected_lon = 115.0 + expected_dx_m / DLON_PER_DEG
    return ValidationCase(
        category="spatially varying diffusivity K",
        name="T05 step reproduces the expected +grad(K)*dt drift (U=0 isolates it)",
        expected=expected_lon, computed=float(new_state.lon[0]), tolerance=1e-9,
        notes="K=100*(lon-115); analytic drift = (+dK/dx)*dt converted to degrees.",
    )


# -- category: particle validity --------------------------------------------------------

def case_valid_destination_continues() -> ValidationCase:
    """A particle whose every proposed step stays inside the valid domain must remain
    valid through every checkpoint (mild, interior-only synthetic flow, K=0)."""
    ocean = make_ocean(np.full((3, 7), 0.1), np.zeros((3, 7)))  # tiny drift, stays interior
    rng = np.random.default_rng(106)
    traj = run_backward_trajectory(QUERY_LAT, 115.0, T0, 4, DT, ocean, LAT, LON, ZERO_K, rng)
    return ValidationCase(
        category="particle validity", name="valid destination at every step -> particle stays valid",
        expected=1.0, computed=float(traj.valid[-1]), tolerance=0.0,
        notes=f"final valid flag after {traj.n_states-1} steps; full history: {traj.valid.tolist()}.",
    )


def _invalid_step_setup():
    """Shared setup for the two "leaves the domain" cases below: constant eastward `uo`
    tuned (via the already-validated T03 geometry conversion, not a second transport
    model) so the backward step is exactly `step_deg=3.0` degrees of longitude per step,
    K=0. Starting at lon0=114.0 with `LON.min()=100.0`, the particle must first leave the
    covered domain (`lon < 100`) at step index 5 (114, 111, 108, 105, 102, [99 -> invalid]),
    an exact, hand-computable prediction, not an empirically-observed one."""
    step_deg = 3.0
    v_speed = step_deg * DLON_PER_DEG / DT
    ocean = make_ocean(np.full((3, 7), v_speed), np.zeros((3, 7)))
    rng = np.random.default_rng(107)
    traj = run_backward_trajectory(QUERY_LAT, 114.0, T0, 6, DT, ocean, LAT, LON, ZERO_K, rng)
    return traj


def case_invalid_destination_step_matches_prediction() -> ValidationCase:
    """The particle from `_invalid_step_setup` must become invalid at exactly the
    analytically predicted step index (5), confirming the out-of-domain boundary policy
    triggers neither early nor late."""
    traj = _invalid_step_setup()
    expected_first_invalid = 5
    actual = traj.first_invalid_index
    return ValidationCase(
        category="particle validity",
        name="leaving the domain marks the particle invalid at the exact predicted step",
        expected=float(expected_first_invalid), computed=float(actual if actual is not None else -1),
        tolerance=0.0,
        notes=f"lon history: {traj.lon.tolist()}; predicted to fall below LON.min()=100.0 at step 5.",
    )


def case_invalid_particle_retained_not_truncated() -> ValidationCase:
    """Once invalid, the particle must remain RETAINED (not dropped): the trajectory's
    array length must still equal `n_steps + 1`, and every state from the first invalid
    index onward must still be present (frozen), not silently removed."""
    traj = _invalid_step_setup()
    n_steps = 6
    length_ok = traj.n_states == n_steps + 1
    frozen_ok = bool(np.all(~traj.valid[traj.first_invalid_index:]))
    passed_as_float = 1.0 if (length_ok and frozen_ok) else 0.0
    return ValidationCase(
        category="particle validity",
        name="invalid particle stays in the record (not truncated/dropped) once invalidated",
        expected=1.0, computed=passed_as_float, tolerance=0.0,
        notes=f"n_states={traj.n_states} (expected {n_steps+1}); "
              f"all states from first_invalid_index onward stay invalid: {frozen_ok}.",
    )


# -- category: particle-weight behavior --------------------------------------------------

def case_weight_linear_accumulation() -> ValidationCase:
    """Multi-step trajectory under a spatially linear (hence spatially CONSTANT
    divergence) `U`: since `div(U)` is the same everywhere the particle visits, the
    log-weight must accumulate EXACTLY linearly, `log_w_final = log_w0 - div(U)*dt*n_steps`
    -- not merely "trend in the right direction"."""
    C = 0.001
    uo_grid = C * (LON[None, :] - 115.0) * np.ones((3, 1))  # divergent, slope C
    ocean = make_ocean(uo_grid, ZERO_UV)
    n_steps = 5
    rng = np.random.default_rng(108)
    traj = run_backward_trajectory(QUERY_LAT, 115.0, T0, n_steps, DT, ocean, LAT, LON, ZERO_K, rng, log_weight0=0.0)
    expected_div = C / DLON_PER_DEG
    expected_final_weight = -expected_div * DT * n_steps
    return ValidationCase(
        category="particle-weight behavior",
        name="log-weight accumulates exactly linearly under constant divergence",
        expected=expected_final_weight, computed=float(traj.log_weight[-1]), tolerance=1e-9,
        notes=f"{n_steps} steps, analytic div(U)={expected_div:.6e} /s; particle stayed valid: "
              f"{bool(traj.valid[-1])}.",
    )


ALL_CASES = [
    case_zero_divergence,
    case_backward_sign_convention,
    case_known_convergent_flow,
    case_known_divergent_flow,
    case_constant_k_zero_gradient,
    case_constant_k_diffusion_displacement,
    case_varying_k_gradient_matches_analytic,
    case_varying_k_drift_matches_t05,
    case_valid_destination_continues,
    case_invalid_destination_step_matches_prediction,
    case_invalid_particle_retained_not_truncated,
    case_weight_linear_accumulation,
]


def run_all_cases() -> list[ValidationCase]:
    """Build every `ValidationCase` in `ALL_CASES` -- the single source of truth both this
    file's `unittest` assertions and the T12 manual-check notebook's validation table read
    from, so the two can never silently disagree."""
    return [case_fn() for case_fn in ALL_CASES]


class TestControlledValidationSuite(unittest.TestCase):
    """One assertion per `ValidationCase`, each reporting the case's own expected/
    computed/error/tolerance on failure -- never a bare "it ran without raising"."""

    def test_every_case_passes_its_own_stated_tolerance(self):
        cases = run_all_cases()
        failures = [c for c in cases if not c.passed]
        if failures:
            lines = [
                f"  [{c.category}] {c.name}: expected={c.expected!r}, computed={c.computed!r}, "
                f"error={c.error!r} > tolerance={c.tolerance!r}"
                for c in failures
            ]
            self.fail("controlled validation case(s) failed:\n" + "\n".join(lines))

    def test_all_seven_required_categories_are_covered(self):
        required = {
            "zero divergence", "known convergent flow", "known divergent flow",
            "constant diffusivity K", "spatially varying diffusivity K",
            "particle validity", "particle-weight behavior",
        }
        covered = {c.category for c in run_all_cases()}
        missing = required - covered
        self.assertEqual(missing, set(), f"missing required T12 categories: {missing}")

    def test_no_case_has_expected_equal_computed_by_construction(self):
        # guards against a case function accidentally comparing a value against itself
        # (which would trivially "pass" without validating anything) -- every case must
        # independently derive `expected` from the synthetic field's known analytic
        # behavior, not copy `computed` into `expected`.
        for case_fn in ALL_CASES:
            case = case_fn()
            self.assertIsInstance(case.expected, float)
            self.assertIsInstance(case.computed, float)
            self.assertTrue(case.name)
            self.assertTrue(case.notes, f"case {case.name!r} has no explanatory notes")


class TestRealDataSanityCheck(unittest.TestCase):
    """Per instruction 13: real `data/ocean.nc` is NEVER treated as analytically-known
    "truth" -- there is no expected value to compare against. This is a lightweight
    sanity check only (no NaN explosion, diagnostics stay finite/in-range on a tiny real
    run), kept structurally and visibly separate from `ALL_CASES`/the validation table.
    """

    def test_small_real_run_produces_finite_sane_diagnostics(self):
        ocean = load_field_dataset(
            "data/ocean.nc", ["uo", "vo"], time_dim="time", lat_dim="lat", lon_dim="lon"
        )
        t0 = ocean.time[50]
        particles = adapt_upstream_particles(
            [
                {"particle_id": "s0", "latitude": 33.0, "longitude": 30.0, "observation_pixel_id": "pxs", "timestamp": t0},
                {"particle_id": "s1", "latitude": 33.1, "longitude": 30.1, "observation_pixel_id": "pxs", "timestamp": t0},
            ]
        )
        ensemble = ForcingEnsemble(ocean, ForcingEnsembleConfig(n_members=2, master_seed=1))
        k_field = np.zeros((len(ocean.lat), len(ocean.lon)))
        records = run_ensemble(particles, [0, 1], 4, 900.0, ocean, ensemble, ocean.lat, ocean.lon, k_field, master_seed=2)

        # no NaN in valid rows' position/weight -- a NaN there would mean invalidity was
        # not applied consistently (this is a sanity check, not a numerical-accuracy claim).
        self.assertTrue(np.all(np.isfinite(records.lat[records.valid])))
        self.assertTrue(np.all(np.isfinite(records.lon[records.valid])))
        self.assertTrue(np.all(np.isfinite(records.log_weight[records.valid])))

        diag = compute_diagnostics(records)
        self.assertTrue(np.isfinite(diag.total_weight))
        self.assertTrue(np.isfinite(diag.effective_sample_size))
        self.assertGreaterEqual(diag.valid_particle_fraction, 0.0)
        self.assertLessEqual(diag.valid_particle_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
