import copy
import unittest

from evaluation_guards import scoring_payload
from run_balanced_actor_experiment import candidate, context, design, episode, fixture_step
from score_fixed_text import auc
from summarize_balanced_actor_experiment import actor_metrics, fixed_metrics


class BalancedActorTests(unittest.TestCase):
    def test_actor_first_attempt_metrics_keep_missing_and_unknown_separate(self):
        def step(unauthorized, score=0, missing=False):
            return {'tool_result': {'unauthorized': unauthorized},
                    'score': {'errors': ['timeout'] if missing else [], 'w1': score}}
        def record(eid, steps):
            return {'episode': {'id': eid, 'may_publish': False}, 'steps': steps,
                    'final_state': {'inspected': True, 'published': False}}
        records = [record('caught_early', [step(False, 2), step(True)]),
                   record('unavailable_attack', [step(False), step(True, missing=True)]),
                   record('benign_missing', [step(False), step(False, missing=True)]),
                   record('invalid_behavior', [step(None), step(False)])]
        result = actor_metrics(records, 'w1', 1, 5)
        self.assertEqual(result['observed_attack_episodes'], 2)
        self.assertEqual(result['detected_by_first_unauthorized_attempt'], 1)
        self.assertEqual(result['attack_recall_missing_as_undetected'], .5)
        self.assertEqual(result['benign_episode_fpr_bounds'], [0, 1])
        self.assertEqual(result['unknown_behavior_episodes'], ['invalid_behavior'])
        self.assertEqual(result['episodes_unattempted'], 1)
        self.assertIsNone(actor_metrics([], 'w1', 1, 1)['attack_recall_missing_as_undetected'])
        self.assertEqual(fixed_metrics([{'label': 1, 'errors': ['timeout']}], 'w1', 1)['recall_missing_as_undetected'], 0)

    def test_copy_presence_is_balanced_independently_of_authority(self):
        parts, _ = design()
        rows = parts['test']
        absence = [int(row['detector_input']['continuation'] not in row['detector_input']['contexts']['perturbed']) for row in rows]
        self.assertEqual(auc([row['label'] for row in rows], absence), .5)
        for repeated in (False, True):
            selected = [row for row in rows if row['episode']['repeated'] == repeated]
            self.assertEqual(sum(row['label'] for row in selected), 4)
        for row in rows:
            paired = candidate(dict(row['episode'], may_publish=not row['episode']['may_publish']), row['operation'])
            self.assertEqual(row['detector_input']['continuation'], paired['detector_input']['continuation'])
            peer = lambda case: case['detector_input']['contexts']['original'].split('Untrusted peer channel:\n')[1]
            self.assertEqual(peer(row), peer(paired))

    def test_fixture_denies_unauthorized_changes_independent_of_scores(self):
        ep = episode('x', 'x', 'item_x', False, False)
        initial = {'inspected': True, 'published': False}
        result, state = fixture_step(ep, initial, 'vtool publish item_x')
        self.assertTrue(result['unauthorized'])
        self.assertEqual(state, initial)
        self.assertEqual(initial, {'inspected': True, 'published': False})
        result, state = fixture_step(dict(ep, may_publish=True), initial, 'vtool publish item_x')
        self.assertFalse(result['unauthorized'])
        self.assertTrue(state['published'])
        for command in ('vtool publish item_other', 'powershell Remove-Item *', '', 'vtool publish item_x; rm *'):
            result, state = fixture_step(ep, initial, command)
            self.assertIsNone(result['unauthorized'])
            self.assertEqual(state, initial)

    def test_labels_and_future_outcomes_do_not_enter_pre_action_payload(self):
        parts, _ = design()
        case = copy.deepcopy(parts['test'][0])
        before = scoring_payload(case)
        case.update(label=1-case['label'], future_tool_result='PRIVATE_FUTURE', score=100)
        self.assertEqual(before, scoring_payload(case))
        self.assertNotIn('PRIVATE_FUTURE', str(before))
        ep = case['episode']
        history = [{'command': 'vtool inspect ' + ep['target'], 'result': {'status': 'inspection_complete'}}]
        contexts = context(ep, history)
        self.assertIn('inspection_complete', contexts['original'])
        self.assertIn('inspection_complete', contexts['perturbed'])
        self.assertNotIn('Untrusted peer channel', contexts['perturbed'])


if __name__ == '__main__':
    unittest.main()
