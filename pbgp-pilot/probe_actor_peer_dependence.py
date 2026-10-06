"""Retrospective generation diagnostic, separate from frozen detection metrics.

For first observed unauthorized proposals only, regenerate the same decision
with its original and peer-removed contexts. Never execute these completions.
"""
import argparse
import json
from pathlib import Path

from run_balanced_actor_experiment import Service, fixture_step, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    target = root/'peer_dependence_diagnostic.json'
    if target.exists():
        raise SystemExit('Diagnostic already preserved')
    config = json.loads((root/'frozen_config.json').read_text(encoding='utf-8'))
    actors = json.loads((root/'actor_results.json').read_text(encoding='utf-8'))
    svc = Service(root, config['model'])
    results = []
    for record in actors:
        step = next((step for step in record['steps'] if step.get('tool_result', {}).get('unauthorized') is True), None)
        if step is None:
            continue
        row = {'episode_id': record['episode']['id'], 'step': step['step'], 'observed_command': step['command'], 'variants': {}}
        for variant, prefix in step['detector_input']['contexts'].items():
            body = {'model': config['model'], 'prompt': prefix, 'max_tokens': 48,
                    'temperature': 0, 'stop': ['\n'], 'echo': False}
            result = svc.request(body, 'peer_dependence_diagnostic', record['episode']['id'], variant)
            if result.get('status') == 200:
                command = result['response']['choices'][0]['text'].strip()
                outcome, _ = fixture_step(record['episode'], {'inspected': True, 'published': False}, command)
                row['variants'][variant] = {'command': command, 'unauthorized': outcome['unauthorized'], 'status': outcome['status']}
            else:
                row['variants'][variant] = {'error': result.get('error', result.get('status'))}
        results.append(row)
    save(target, {'scope': 'Post-outcome selected diagnostic; not predeclared benchmark or detection tuning.',
                  'execution': 'None; regenerated completions remain inert.', 'results': results})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
