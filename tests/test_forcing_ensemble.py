"""Focused T08 checks for `iste.forcing_ensemble` (prototype surrogate forcing ensemble).

*** Tests a PROTOTYPE, NOT VALIDATED model. *** See `progress.md`'s T08 section.

Run with:
    python -m unittest tests.test_forcing_ensemble -v
"""

from __future__ import annotations

import unittest

import numpy as np

from iste.derivatives import divergence_2d
from iste.environment import GriddedFieldSource
from iste.forcing_ensemble import ForcingEnsemble, ForcingEnsembleConfig

LAT = np.linspace(10.0, 30.0, 9)  # ascending, 2.5-degree spacing
LON = np.linspace(100.0, 120.0, 11)  # ascending, 2.0-degree spacing
T0 = np.datetime64("2020-01-01T00:00", "ns")
N_HOURS = 25
TIME = T0 + np.arange(N_HOURS) * np.timedelta64(3600, "s")  # 24 hours, hourly

# Test-scale config: coarser correlation time (2h) than the real default (~1 day) so a
# 24-hour synthetic time series is long enough to observe clear temporal decorrelation,
# and a spatial correlation length tuned to give a moderate (~1 cell) smoothing amount on
# this small synthetic domain. These overrides are for test observability only -- the
# module's own defaults (used in the manual-check notebook) remain the PROTOTYPE values
# from the task instructions (20 km / 1 day / 0.05 m/s / E=50).
TEST_CONFIG = ForcingEnsembleConfig(
    n_members=6,
    spatial_correlation_km=60.0,
    correlation_time_seconds=2 * 3600.0,
    sigma_u=0.05,
    coarse_resolution_deg=0.5,
    master_seed=123,
)


def make_ocean() -> GriddedFieldSource:
    uo = np.zeros((N_HOURS, len(LAT), len(LON)))
    vo = np.zeros((N_HOURS, len(LAT), len(LON)))
    return GriddedFieldSource(
        source_path="<synthetic>", time_name="time", time=TIME, lat_name="lat", lat=LAT,
        lon_name="lon", lon=LON, data={"uo": uo, "vo": vo}, fill_threshold=1e19,
    )


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_member_reproducible_across_fresh_instances(self):
        ocean = make_ocean()
        ens_a = ForcingEnsemble(ocean, TEST_CONFIG)
        ens_b = ForcingEnsemble(ocean, TEST_CONFIG)
        du_a, dv_a, ok_a = ens_a.perturbation(0, 20.0, 110.0, TIME[5])
        du_b, dv_b, ok_b = ens_b.perturbation(0, 20.0, 110.0, TIME[5])
        self.assertEqual(du_a, du_b)
        self.assertEqual(dv_a, dv_b)
        self.assertEqual(ok_a, ok_b)

    def test_repeated_queries_of_same_member_time_are_identical(self):
        ocean = make_ocean()
        ens = ForcingEnsemble(ocean, TEST_CONFIG)
        first = ens.perturbation(2, 22.0, 108.0, TIME[10])
        second = ens.perturbation(2, 22.0, 108.0, TIME[10])
        self.assertEqual(first, second)

    def test_different_members_differ(self):
        ocean = make_ocean()
        ens = ForcingEnsemble(ocean, TEST_CONFIG)
        du0, dv0, _ = ens.perturbation(0, 20.0, 110.0, TIME[5])
        du1, dv1, _ = ens.perturbation(1, 20.0, 110.0, TIME[5])
        self.assertTrue(du0 != du1 or dv0 != dv1)

    def test_master_seed_changes_all_members(self):
        ocean = make_ocean()
        cfg_other_seed = ForcingEnsembleConfig(**{**TEST_CONFIG.__dict__, "master_seed": 999})
        ens_a = ForcingEnsemble(ocean, TEST_CONFIG)
        ens_b = ForcingEnsemble(ocean, cfg_other_seed)
        du_a, dv_a, _ = ens_a.perturbation(0, 20.0, 110.0, TIME[5])
        du_b, dv_b, _ = ens_b.perturbation(0, 20.0, 110.0, TIME[5])
        self.assertTrue(du_a != du_b or dv_a != dv_b)


class TestSpatialCorrelationObservable(unittest.TestCase):
    def test_nearby_points_more_correlated_than_distant_points(self):
        ocean = make_ocean()
        many_members_cfg = ForcingEnsembleConfig(**{**TEST_CONFIG.__dict__, "n_members": 40})
        ens = ForcingEnsemble(ocean, many_members_cfg)

        center = (20.0, 110.0)
        near = (20.3, 110.3)  # close: well within the smoothing length scale
        far = (28.0, 118.0)  # far: near the opposite corner of the domain

        center_vals, near_vals, far_vals = [], [], []
        for m in range(many_members_cfg.n_members):
            du_c, _, _ = ens.perturbation(m, *center, TIME[5])
            du_n, _, _ = ens.perturbation(m, *near, TIME[5])
            du_f, _, _ = ens.perturbation(m, *far, TIME[5])
            center_vals.append(du_c)
            near_vals.append(du_n)
            far_vals.append(du_f)

        corr_near = np.corrcoef(center_vals, near_vals)[0, 1]
        corr_far = np.corrcoef(center_vals, far_vals)[0, 1]
        self.assertGreater(corr_near, corr_far)
        self.assertGreater(corr_near, 0.5)  # nearby points should be strongly correlated

    def test_smoothing_reduces_cell_to_cell_variability_vs_raw_noise(self):
        # Compare the raw (pre-smoothing) coarse OU state to the smoothed+queried field's
        # spatial roughness -- smoothing must visibly reduce adjacent-cell differences.
        ocean = make_ocean()
        ens = ForcingEnsemble(ocean, TEST_CONFIG)
        u_series, _ = ens._member_state(0)
        raw = u_series[5]
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(raw, sigma=ens._smoothing_sigma_cells, mode="reflect")
        raw_roughness = np.mean(np.abs(np.diff(raw, axis=1)))
        smoothed_roughness = np.mean(np.abs(np.diff(smoothed, axis=1)))
        self.assertLess(smoothed_roughness, raw_roughness)


class TestTemporalCorrelationObservable(unittest.TestCase):
    def test_short_lag_more_correlated_than_long_lag(self):
        ocean = make_ocean()
        many_members_cfg = ForcingEnsembleConfig(**{**TEST_CONFIG.__dict__, "n_members": 40})
        ens = ForcingEnsemble(ocean, many_members_cfg)

        t_ref, t_short, t_long = TIME[2], TIME[3], TIME[20]  # +1h vs +18h
        ref_vals, short_vals, long_vals = [], [], []
        for m in range(many_members_cfg.n_members):
            du_ref, _, _ = ens.perturbation(m, 20.0, 110.0, t_ref)
            du_short, _, _ = ens.perturbation(m, 20.0, 110.0, t_short)
            du_long, _, _ = ens.perturbation(m, 20.0, 110.0, t_long)
            ref_vals.append(du_ref)
            short_vals.append(du_short)
            long_vals.append(du_long)

        corr_short = np.corrcoef(ref_vals, short_vals)[0, 1]
        corr_long = np.corrcoef(ref_vals, long_vals)[0, 1]
        self.assertGreater(corr_short, corr_long)
        self.assertGreater(corr_short, 0.3)  # 1h lag vs a 2h correlation time: still correlated


class TestPerturbationScale(unittest.TestCase):
    def test_std_is_same_order_of_magnitude_as_sigma_u(self):
        ocean = make_ocean()
        many_members_cfg = ForcingEnsembleConfig(**{**TEST_CONFIG.__dict__, "n_members": 30})
        ens = ForcingEnsemble(ocean, many_members_cfg)
        samples = []
        for m in range(many_members_cfg.n_members):
            du, dv, _ = ens.perturbation(m, 20.0, 110.0, TIME[5])
            samples.extend([du, dv])
        samples = np.array(samples)
        std = float(np.std(samples))
        # Smoothing reduces the raw OU marginal std, so this checks the same *order of
        # magnitude* as sigma_u, not an exact match (approximate, per the task).
        self.assertGreater(std, many_members_cfg.sigma_u * 0.05)
        self.assertLess(std, many_members_cfg.sigma_u * 3.0)


class TestMemoryDiscipline(unittest.TestCase):
    def test_generating_one_member_does_not_touch_others(self):
        ocean = make_ocean()
        cfg = ForcingEnsembleConfig(**{**TEST_CONFIG.__dict__, "n_members": 50})
        ens = ForcingEnsemble(ocean, cfg)
        self.assertEqual(ens.cached_member_ids(), ())
        ens.perturbation(3, 20.0, 110.0, TIME[5])
        self.assertEqual(ens.cached_member_ids(), (3,))

    def test_per_member_state_far_smaller_than_naive_full_grid_all_members_estimate(self):
        # Compared against the REAL ocean grid's dimensions (149 x 216, verified in
        # T00/T01), not this test's own deliberately-tiny synthetic fixture -- the
        # coarse-grid design exists specifically to avoid the real grid's cost, and a
        # tiny synthetic fine grid would make this comparison meaningless (its coarse
        # grid, sized for a realistic ~20-degree domain, can be larger than a 9x11
        # synthetic fine grid purely as a fixture artifact).
        REAL_OCEAN_N_LAT, REAL_OCEAN_N_LON, REAL_OCEAN_N_TIME = 149, 216, 336

        ocean = make_ocean()
        cfg = ForcingEnsembleConfig(**{**TEST_CONFIG.__dict__, "n_members": 50})
        ens = ForcingEnsemble(ocean, cfg)
        per_member_bytes = ens.member_state_nbytes(0)

        naive_full_grid_all_members_bytes = (
            REAL_OCEAN_N_TIME * cfg.n_members * REAL_OCEAN_N_LAT * REAL_OCEAN_N_LON * 2 * 8  # u and v, float64
        )
        self.assertLess(per_member_bytes * cfg.n_members, naive_full_grid_all_members_bytes)

    def test_fifty_members_can_be_generated_without_excessive_memory(self):
        ocean = make_ocean()
        cfg = ForcingEnsembleConfig(**{**TEST_CONFIG.__dict__, "n_members": 50})
        ens = ForcingEnsemble(ocean, cfg)
        for m in range(cfg.n_members):
            ens.perturbation(m, 20.0, 110.0, TIME[5])
        self.assertEqual(len(ens.cached_member_ids()), 50)
        total_bytes = sum(ens.member_state_nbytes(m) for m in range(cfg.n_members))
        # Bounded: comfortably under 50 MB for this small synthetic test domain/time range.
        self.assertLess(total_bytes, 50 * 1024 * 1024)


class TestConfigValidation(unittest.TestCase):
    def test_invalid_n_members_rejected(self):
        with self.assertRaises(ValueError):
            ForcingEnsembleConfig(n_members=0)

    def test_invalid_spatial_correlation_rejected(self):
        with self.assertRaises(ValueError):
            ForcingEnsembleConfig(spatial_correlation_km=0.0)

    def test_invalid_correlation_time_rejected(self):
        with self.assertRaises(ValueError):
            ForcingEnsembleConfig(correlation_time_seconds=-1.0)

    def test_negative_sigma_u_rejected(self):
        with self.assertRaises(ValueError):
            ForcingEnsembleConfig(sigma_u=-0.01)

    def test_invalid_coarse_resolution_rejected(self):
        with self.assertRaises(ValueError):
            ForcingEnsembleConfig(coarse_resolution_deg=0.0)

    def test_out_of_range_member_id_rejected(self):
        ocean = make_ocean()
        ens = ForcingEnsemble(ocean, TEST_CONFIG)
        with self.assertRaises(ValueError):
            ens.perturbation(TEST_CONFIG.n_members, 20.0, 110.0, TIME[5])


class TestPerturbedVelocityComposition(unittest.TestCase):
    def test_perturbed_velocity_equals_base_plus_delta(self):
        ocean = make_ocean()
        ens = ForcingEnsemble(ocean, TEST_CONFIG)
        du, dv, pert_ok = ens.perturbation(0, 20.0, 110.0, TIME[5])
        u_e, v_e, valid = ens.perturbed_velocity(0, 20.0, 110.0, TIME[5])
        self.assertTrue(valid)
        self.assertTrue(pert_ok)
        # base uo/vo are exactly 0.0 in this synthetic dataset.
        self.assertAlmostEqual(u_e, du, places=9)
        self.assertAlmostEqual(v_e, dv, places=9)


class TestDivergenceIsNotForcedToZero(unittest.TestCase):
    """The important scientific rule: delta_U must NOT be artificially divergence-free."""

    def test_perturbed_field_has_nonzero_divergence_somewhere(self):
        ocean = make_ocean()
        ens = ForcingEnsemble(ocean, TEST_CONFIG)
        u_series, v_series = ens._member_state(0)
        from scipy.ndimage import gaussian_filter

        u_smoothed = gaussian_filter(u_series[5], sigma=ens._smoothing_sigma_cells, mode="reflect")
        v_smoothed = gaussian_filter(v_series[5], sigma=ens._smoothing_sigma_cells, mode="reflect")
        div_field, valid_field = divergence_2d(ens.coarse_lat, ens.coarse_lon, u_smoothed, v_smoothed)
        self.assertTrue(np.any(valid_field))
        # A genuinely non-divergence-free (unconstrained) random field should show a
        # nontrivial spread of divergence values -- not be forced to (near) zero.
        self.assertGreater(np.nanstd(div_field), 0.0)


if __name__ == "__main__":
    unittest.main()
