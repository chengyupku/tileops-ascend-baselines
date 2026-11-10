"""CPU-only lookup of an exported case; never imports or executes a baseline."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--baseline')
p.add_argument('--operator')
p.add_argument('--case')
p.add_argument('--category')
p.add_argument('--details', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parent
amnesty_path = root / 'provisionally_retained_legacy_success.csv'
amnesty = set()
if amnesty_path.exists():
    with amnesty_path.open(encoding='utf-8-sig', newline='') as f:
        amnesty = {(r['baseline'], r['operator'], r['case_id']) for r in csv.DictReader(f)}
with (root / 'ALL_CASES.csv').open(encoding='utf-8-sig', newline='') as f:
    matches = [r for r in csv.DictReader(f)
               if (not a.baseline or r['baseline'] == a.baseline)
               and (not a.operator or r['operator'] == a.operator)
               and (not a.case or r['case_id'] == a.case)
               and (not a.category or r['category'] == a.category)]
if a.details:
    lines = {int(r['snapshot_line']) for r in matches}
    for i, line in enumerate((root / 'EXACT_TARGET_SETS.snapshot.jsonl').open(), 1):
        if i in lines:
            record = json.loads(line)
            if tuple(record['key']) in amnesty:
                record['current_user_decision'] = 'temporarily_retained_completed; no further historical audit/revert/rerun; original provenance remains unconfirmed'
            print(json.dumps(record, ensure_ascii=False, indent=2))
else:
    for r in matches:
        if (r['baseline'], r['operator'], r['case_id']) in amnesty:
            r['current_user_decision'] = 'temporarily_retained_completed; no further historical audit/revert/rerun; original provenance remains unconfirmed'
        print(json.dumps(r, ensure_ascii=False))
print(f'{len(matches)} matched case(s)')
