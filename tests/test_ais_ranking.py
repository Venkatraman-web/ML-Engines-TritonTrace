"""Focused TASK AIS-5 checks for `iste.ais_ranking` (first task allowed to rank
vessels -- an EVIDENCE-RANKING score, never a source-attribution probability; see
module docstring).

Run with:
    python -m unittest tests.test_ais_ranking -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iste.ais_evidence import compute_vessel_evidence
from iste.ais_matching import compute_vessel_candidate_matches
from iste.ais_ranking import (
    DEFAULT_WEIGHTS,
    EQUAL_WEIGHTS,
    AISScoreWeights,
    compute_ais_ranking,
    compute_evidence_concentration,
)


def _candidates_df():
    return pd.DataFrame([
        dict(candidate_id="c1", t_s=pd.Timestamp("2019-06-12T00:35:58Z"), lat=34.0, lon=35.0, support_score=100.0, cluster_id="cluster0000"),
        dict(candidate_id="c2", t_s=pd.Timestamp("2019-06-12T01:35:58Z"), lat=34.1, lon=35.1, support_score=50.0, cluster_id="cluster0001"),
        dict(candidate_id="c3", t_s=pd.Timestamp("2019-06-12T02:35:58Z"), lat=34.2, lon=35.2, support_score=25.0, cluster_id="cluster0001"),
    ])


def _ais_df():
    rows = [
        # vessel 1: exact matches at all 3 checkpoints, zero distance -> strong evidence
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T00:35:58Z"), lat=34.0, lon=35.0, sog=5.0, cog=90.0),
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T01:35:58Z"), lat=34.1, lon=35.1, sog=5.0, cog=90.0),
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", imo=111, timestamp=pd.Timestamp("2019-06-12T02:35:58Z"), lat=34.2, lon=35.2, sog=5.0, cog=90.0),
        # vessel 2: far away -> weak/no evidence
        dict(mmsi=2, vessel_name="B", vessel_type="Cargo", imo=222, timestamp=pd.Timestamp("2019-06-12T00:35:58Z"), lat=10.0, lon=10.0, sog=8.0, cog=180.0),
        # vessel 3: no observations near the grid at all
        dict(mmsi=3, vessel_name="C", vessel_type="Tanker", imo=333, timestamp=pd.Timestamp("2019-06-12T12:35:58Z"), lat=34.0, lon=35.0, sog=8.0, cog=180.0),
    ]
    return pd.DataFrame(rows)


def _gap_summary_df():
    return pd.DataFrame([
        dict(mmsi=1, vessel_name="A", vessel_type="Fishing", n_observations=3, median_interval_hours=1.0, max_interval_hours=1.0, n_gaps_over_threshold=0),
        dict(mmsi=2, vessel_name="B", vessel_type="Cargo", n_observations=1, median_interval_hours=np.nan, max_interval_hours=np.nan, n_gaps_over_threshold=0),
        dict(mmsi=3, vessel_name="C", vessel_type="Tanker", n_observations=1, median_interval_hours=np.nan, max_interval_hours=np.nan, n_gaps_over_threshold=0),
    ])


def _build_all():
    candidates = _candidates_df()
    ais = _ais_df()
    matches = compute_vessel_candidate_matches(candidates, ais, max_time_tolerance_hours=2.0)
    vessel_evidence = compute_vessel_evidence(matches, ais, _gap_summary_df(), n_candidate_timestamps=3)
    null_df = pd.DataFrame(columns=["mmsi", "empirical_p_value"])  # no null computed in this fixture
    return candidates, ais, matches, vessel_evidence, null_df


class TestComputeAisRanking(unittest.TestCase):
    def setUp(self):
        self.candidates, self.ais, self.matches, self.vessel_evidence, self.null_df = _build_all()
        self.ranking = compute_ais_ranking(self.vessel_evidence, self.matches, self.null_df, self.candidates, level="exact")

    def test_every_vessel_present(self):
        self.assertEqual(set(self.ranking["mmsi"]), {1, 2, 3})

    def test_vessel1_ranks_first(self):
        self.assertEqual(self.ranking.iloc[0]["mmsi"], 1)
        self.assertEqual(self.ranking.iloc[0]["rank"], 1)

    def test_vessel_with_zero_matches_gets_nan_score_and_no_rank(self):
        row3 = self.ranking[self.ranking["mmsi"] == 3].iloc[0]
        self.assertTrue(np.isnan(row3["ais_attribution_score"]))
        self.assertTrue(np.isnan(row3["rank"]))

    def test_score_bounded_reasonable(self):
        valid = self.ranking["ais_attribution_score"].dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 1.0001).all())

    def test_vessel_type_not_used_in_score_but_present_descriptively(self):
        self.assertIn("vessel_type", self.ranking.columns)
        # two vessels with identical spatial/temporal/support evidence but different
        # vessel_type must receive identical scores -- verified indirectly: no vessel_type
        # term appears in effective weights.
        self.assertNotIn("vessel_type", self.ranking.filter(like="effective_weight").columns)

    def test_disclaimer_language_not_a_probability(self):
        from iste.ais_ranking import EVIDENCE_RANKING_DISCLAIMER
        self.assertIn("NOT a probability", EVIDENCE_RANKING_DISCLAIMER)
        self.assertIn("not a source attribution conclusion", EVIDENCE_RANKING_DISCLAIMER)

    def test_null_weight_zero_by_default_score_independent_of_null(self):
        self.assertEqual(DEFAULT_WEIGHTS.w_null, 0.0)
        row1 = self.ranking[self.ranking["mmsi"] == 1].iloc[0]
        self.assertEqual(row1["effective_weight_null"], 0.0)

    def test_reproducible(self):
        ranking2 = compute_ais_ranking(self.vessel_evidence, self.matches, self.null_df, self.candidates, level="exact")
        pd.testing.assert_series_equal(self.ranking["ais_attribution_score"], ranking2["ais_attribution_score"])


class TestToleranceLevelSeparation(unittest.TestCase):
    def test_exact_and_2h_can_differ(self):
        candidates, ais, matches, vessel_evidence, null_df = _build_all()
        ranking_exact = compute_ais_ranking(vessel_evidence, matches, null_df, candidates, level="exact")
        ranking_2h = compute_ais_ranking(vessel_evidence, matches, null_df, candidates, level="within_2h")
        # both must be valid, independently-computed tables (not literally required to differ,
        # but both must exist and be internally consistent)
        self.assertEqual(len(ranking_exact), len(ranking_2h))
        self.assertIn("ais_attribution_score", ranking_exact.columns)
        self.assertIn("ais_attribution_score", ranking_2h.columns)

    def test_invalid_level_raises(self):
        candidates, ais, matches, vessel_evidence, null_df = _build_all()
        with self.assertRaises(ValueError):
            compute_ais_ranking(vessel_evidence, matches, null_df, candidates, level="bogus")


class TestWeightsValidation(unittest.TestCase):
    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            AISScoreWeights(w_spatial=-0.1)

    def test_equal_weights_sum_reasonable(self):
        total = sum(EQUAL_WEIGHTS.as_dict()[k] for k in ("spatial", "temporal_consecutive", "temporal_coverage", "support"))
        self.assertAlmostEqual(total, 1.0)


class TestEvidenceConcentration(unittest.TestCase):
    def test_concentration_fields_present(self):
        candidates, ais, matches, vessel_evidence, null_df = _build_all()
        ranking = compute_ais_ranking(vessel_evidence, matches, null_df, candidates, level="exact")
        result = compute_evidence_concentration(ranking, top_k_values=(1, 2))
        self.assertIn("gini_coefficient", result)
        self.assertIn("top_1_share_of_total_score", result)
        self.assertGreaterEqual(result["gini_coefficient"], 0.0)
        self.assertLessEqual(result["gini_coefficient"], 1.0001)


if __name__ == "__main__":
    unittest.main()
