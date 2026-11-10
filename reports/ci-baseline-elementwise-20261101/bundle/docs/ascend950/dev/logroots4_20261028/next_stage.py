"""CPU queue: complete the critical opponent first, then compare configurations."""
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
subprocess.run(['python3',str(HERE/'baseline_batch.py'),'--phase','compile','--op','Log1pFwdOp'],check=True)
subprocess.run(['python3',str(HERE/'complete_search_batch.py')],check=True)
