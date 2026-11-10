"""Let the queued short decision experiment finish before the last large baseline."""
import json
import os
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
while True:
    try:
        state = json.loads((HERE / 'public-lut-v5-status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    if state.get('stage') in ('ready_for_review', 'candidate_rejected', 'stopped_for_review'):
        break
    time.sleep(5)
assert b'/baseline_batch.py' in Path('/proc/1368403/cmdline').read_bytes()
os.kill(1368403, signal.SIGCONT)
(HERE / 'catlass-resumed-after-lut.json').write_text(json.dumps(state, indent=2))
