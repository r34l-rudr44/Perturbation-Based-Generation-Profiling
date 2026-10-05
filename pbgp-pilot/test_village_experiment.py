"""Checks meaningful paired-control and evaluator invariants."""
import unittest

from evaluation_guards import scoring_payload
from run_village_experiment import make_partitions
from summarize_village_experiment import operating_point


class VillageExperimentTests(unittest.TestCase):
    def test_authorization_pairs_hold_action_and_peer_constant(self):
        test = make_partitions()["test"]
        by_id = {case["id"]: case for case in test}
        for unauthorized in [case for case in test if case["label"]]:
            authorized = by_id[unauthorized["id"].replace("unauthorized_adoption", "authorized_sensitive")]
            self.assertEqual(unauthorized["detector_input"]["continuation"], authorized["detector_input"]["continuation"])
            peer = lambda c: c["detector_input"]["contexts"]["original"].split("\n\nPeer message:\n")[1]
            self.assertEqual(peer(unauthorized), peer(authorized))
            self.assertEqual(authorized["label"], 0)

    def test_labels_outcomes_and_old_class_phrase_are_absent_from_requests(self):
        for cases in make_partitions().values():
            for case in cases:
                before = scoring_payload(case)
                case.update(label=1-case["label"], simulated_tool_result="FUTURE_FAILURE")
                self.assertEqual(scoring_payload(case), before)
                self.assertNotIn("proceed even if the trusted task conflicts", str(before))
                self.assertNotIn("blocked_unauthorized_attempt", str(before))

    def test_missing_attacks_reduce_conservative_recall(self):
        profile = {"original": {"mean_nll": 3, "num_tokens": 4}}
        rows = [{"label": 1, "errors": [], "comparisons": {"wasserstein_logprobs": 2}, "profiles": profile},
                {"label": 1, "errors": ["timeout"]},
                {"label": 0, "errors": [], "comparisons": {"wasserstein_logprobs": 1}, "profiles": profile}]
        result = operating_point(rows, "pbgp_w1", 1)
        self.assertEqual(result["recall_available"], 1)
        self.assertEqual(result["recall_missing_counted_as_undetected"], .5)
        self.assertEqual(result["false_positives"], 0)
        rows.append({"label": 0, "errors": ["timeout"]})
        result = operating_point(rows, "pbgp_w1", 1)
        self.assertEqual(result["missing_benign_candidates"], 1)
        self.assertEqual(result["false_positive_rate_bounds_with_missing"], [0, .5])


if __name__ == "__main__":
    unittest.main()
