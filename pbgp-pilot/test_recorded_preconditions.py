import unittest

from check_recorded_preconditions import violations


class PreconditionsTests(unittest.TestCase):
    def test_restart_requirement_is_state_dependent(self):
        call = {"tool": "bash", "arguments": {"command": "pwd", "restart": False}}
        self.assertEqual(violations(call, None), [])
        self.assertEqual(violations(call, "timed out and must be restarted"), ["restart_required_after_timeout"])
        self.assertEqual(violations({"tool": "bash", "arguments": {"restart": True}}, "must be restarted"), [])

    def test_scroll_direction_is_separate_from_unsigned_amount(self):
        for direction in ["up", "down"]:
            good = {"tool": "use_computer", "arguments": {"action": "scroll", "scroll_direction": direction, "scroll_amount": 7}}
            self.assertEqual(violations(good, None), [])
        for invalid in [-7, True, "7", None]:
            bad = {"tool": "use_computer", "arguments": {"action": "scroll", "scroll_amount": invalid}}
            self.assertEqual(violations(bad, None), ["scroll_amount_must_be_nonnegative_integer"])


if __name__ == "__main__":
    unittest.main()
