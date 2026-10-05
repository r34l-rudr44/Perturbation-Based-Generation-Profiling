"""Small reproducible, byte-seeded sample; preserve every available selected-session turn."""
import json, random, re
from pathlib import Path

src = Path('computer_use_turns.jsonl')
out = Path('pbgp-pilot')
out.mkdir(exist_ok=True)
rng = random.Random(20261006)
selected = set()
with src.open('rb') as f:
    for _ in range(16):
        f.seek(rng.randrange(src.stat().st_size))
        f.readline()
        line = f.readline()
        if line:
            selected.add(json.loads(line)['session_id'])
pattern = re.compile(rb'"session_id"\s*:\s*"([^"\\]+)"')
ids = {s.encode() for s in selected}
rows = []
total = 0
with src.open('rb') as f:
    for line in f:
        total += 1
        match = pattern.search(line)
        if match and match[1] in ids:
            rows.append(json.loads(line))
rows.sort(key=lambda r: (r['session_id'], r.get('created_at',''),r['id']))
target = out / 'sample_sessions.jsonl'
with target.open('w', encoding='utf-8') as f:
    for row in rows:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
counts = {sid:sum(r['session_id']==sid for r in rows) for sid in sorted(selected)}
manifest = dict(source=str(src.resolve()), source_bytes=src.stat().st_size,
    source_rows=total, seed=20261006, sessions=counts, sample_rows=len(rows),
    sample_bytes=target.stat().st_size,
    sampling='16 random byte offsets select seed rows; recover all rows belonging to those sessions. Length-biased exploratory sample, not uniform by session.',
    limitations='No attack labels. All available turns preserved, but session goals, original prompts, and screenshots may require other dataset tables.')
(out/'sample_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
