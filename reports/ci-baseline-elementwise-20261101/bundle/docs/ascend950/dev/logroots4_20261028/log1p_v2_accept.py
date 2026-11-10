"""CPU coordinator for corrected Log1p timing, then resume existing sequence."""
from pathlib import Path
import subprocess
import signal
import os
import json
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(HERE.parents[3])
requests=[('final-Log1pFwdOp-v2','env.sh','pilot.py',[],'final'),('baseline-ops-math-Log1pFwdOp-retry','env-math.sh','baselines.py',['--library','ops-math'],'baseline')]
for label,env,script,extra,folder in requests:
    command=['python3',str(HERE/'remote.py'),'--label',label,'--env',env,'--timeout','600','--',str(REL/script),'--op','Log1pFwdOp',*extra,'--output',str(REL/folder/(label+'.jsonl'))]
    subprocess.run(command,check=True)
os.kill(1026045,signal.SIGCONT)
(HERE/'coordinator-hold.json').write_text(json.dumps({'pid':1026045,'state':'resumed after candidate validation and v2 final measurement'},indent=2))
