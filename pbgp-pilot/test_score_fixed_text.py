"""Checks that protect against context/outcome leakage and false scoring claims."""
import unittest

from score_fixed_text import auc, compare_profiles, extract_continuation


class FixedTextTests(unittest.TestCase):
    def response(self, tokens, values, offsets):
        return {"choices": [{"text": "".join(tokens), "logprobs": {
            "tokens": tokens, "token_logprobs": values, "text_offset": offsets}}]}

    def test_excludes_context_and_null_initial_token(self):
        result = extract_continuation(self.response(["ctx", "a", "b"], [None, -1, -2], [0, 3, 4]), "ctx", "ab")
        self.assertEqual(result["tokens"], ["a", "b"])
        self.assertEqual(result["mean_nll"], 1.5)

    def test_rejects_token_crossing_boundary(self):
        with self.assertRaises(ValueError):
            extract_continuation(self.response(["ctxa", "b"], [None, -2], [0, 4]), "ctx", "ab")

    def test_rejects_positive_and_missing_probabilities(self):
        for value in [None, .01, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                extract_continuation(self.response(["ctx", "a"], [None, value], [0, 3]), "ctx", "a")

    def test_alignment_distinguishes_distribution_and_token_deltas(self):
        a = {"tokens": ["a", "b"], "logprobs": [-1, -3], "mean_nll": 2}
        b = dict(a, logprobs=[-3, -1])
        result = compare_profiles(a, b)
        self.assertEqual(result["wasserstein_logprobs"], 0)
        self.assertEqual(result["mean_abs_token_delta"], 2)
        with self.assertRaises(ValueError):
            compare_profiles(a, dict(b, tokens=["ab"]))

    def test_auc_ties_and_missing_class(self):
        self.assertEqual(auc([0, 1], [1, 1]), .5)
        self.assertEqual(auc([0, 1], [0, 1]), 1)
        self.assertIsNone(auc([1], [1]))

    def test_rejects_false_offsets_that_move_context_boundary(self):
        with self.assertRaises(ValueError):
            extract_continuation(self.response(["ctx", "a", "b"], [None, -1, -2], [0, 2, 3]), "ctx", "ab")

    def test_rejects_profile_truncation(self):
        a = {"tokens": ["a", "b"], "logprobs": [-1, -3], "mean_nll": 2}
        with self.assertRaises(ValueError):
            compare_profiles(a, dict(a, logprobs=[-1]))

    def test_auc_rejects_silent_truncation_and_invalid_labels(self):
        for labels, scores in [([0, 1], [1]), ([0, 2], [1, 2]), ([0, 1], [1, float("nan")])]:
            with self.assertRaises(ValueError):
                auc(labels, scores)


if __name__ == "__main__":
    unittest.main()
