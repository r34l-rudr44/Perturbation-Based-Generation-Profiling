"""Offline replay of saved requests and independent fixture state transitions."""
import argparse
import json
from pathlib import Path

from evaluation_guards import calibrate_threshold, scoring_payload
from run_balanced_actor_experiment import context, fixture_step
from score_fixed_text import compare_profiles, extract_continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    parts = read('cases.json')
    payloads = {case['id']: scoring_payload(case) for group in parts.values() for case in group}
    actors = read('actor_results.json')
    scores = {row['id']: row for row in read('profiles.json')}
    generations = {}
    diagnostic_contexts = {}
    replayed = 0
    for record in actors:
        ep = record['episode']
        state, history = {'inspected': False, 'published': False}, []
        for step in record['steps']:
            assert step['pre_action_history'] == history
            generations[(ep['id'], str(step['step']))] = {'prefix': context(ep, history)['original'], 'command': step['command']}
            if step['generation_error']:
                continue
            expected = {'contexts': context(ep, history), 'continuation': step['command']}
            assert expected == step['detector_input']
            payloads[ep['id'] + f"-actor-{step['step']}"] = expected
            if step['command']:
                scores[ep['id'] + f"-actor-{step['step']}"] = step['score']
            if step['tool_result']['unauthorized'] is True and ep['id'] not in diagnostic_contexts:
                diagnostic_contexts[ep['id']] = expected['contexts']
            outcome, state = fixture_step(ep, state, step['command'])
            assert outcome == step['tool_result']
            history.append({'command': step['command'], 'result': outcome})
            replayed += 1
        assert state == record['final_state']
    counts = {'score_successes_validated': 0, 'actor_successes_validated': 0,
              'diagnostic_successes_validated': 0, 'fixture_steps_replayed': replayed}
    for line in (root/'responses.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        body = row['request']
        if row['purpose'] == 'score':
            payload = payloads[row['case_id']]
            prefix = payload['contexts'][row['variant']]
            assert body['prompt'] == prefix + payload['continuation']
            assert body['max_tokens'] == 0 and body['echo'] is True
            if row.get('status') == 200:
                parsed = extract_continuation(row['response'], prefix, payload['continuation'])
                assert parsed == scores[row['case_id']]['profiles'][row['variant']]
                counts['score_successes_validated'] += 1
        elif row['purpose'] == 'actor':
            planned = generations[(row['case_id'], row['variant'])]
            assert body['prompt'] == planned['prefix']
            assert body['max_tokens'] == 48 and body['echo'] is False
            if row.get('status') == 200:
                assert row['response']['choices'][0]['text'].strip() == planned['command']
                counts['actor_successes_validated'] += 1
        else:
            assert row['purpose'] == 'peer_dependence_diagnostic'
            assert body['prompt'] == diagnostic_contexts[row['case_id']][row['variant']]
            assert body['max_tokens'] == 48 and body['echo'] is False
            if row.get('status') == 200:
                diagnostic = read('peer_dependence_diagnostic.json')
                saved = next(item for item in diagnostic['results'] if item['episode_id'] == row['case_id'])
                assert row['response']['choices'][0]['text'].strip() == saved['variants'][row['variant']]['command']
                counts['diagnostic_successes_validated'] += 1
    for score in scores.values():
        if not score['errors']:
            distance = compare_profiles(score['profiles']['original'], score['profiles']['perturbed'])['wasserstein_logprobs']
            assert distance == score['w1']
            assert score['profiles']['original']['mean_nll'] == score['nll']
    counts['stored_scores_recomputed'] = sum(not score['errors'] for score in scores.values())
    calibration = [row for row in read('profiles.json') if row['split'] == 'calibration']
    grouped = {}
    for row in calibration:
        assert not row['errors'] and row['label'] == 0
        grouped.setdefault(row['episode_id'], []).append(row)
    assert len(grouped) == 20 and all(len(group) == 2 for group in grouped.values())
    values = {feature: [max(row[feature] for row in group) for group in grouped.values()] for feature in ('w1', 'nll')}
    frozen = read('frozen_thresholds.json')
    assert values == frozen['calibration_values']
    assert {feature: calibrate_threshold(scores, .05) for feature, scores in values.items()} == frozen['thresholds']
    assert frozen['test_results_saved'] == 0
    counts['calibration_episodes_revalidated'] = len(grouped)
    (root/'evidence_validation.json').write_text(json.dumps(counts, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
