"""Parent runs in a fresh interpreter after serial publication; CPU only."""
import runpy,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.argv=[str(Path(__file__).with_name('cpu_checks.py')),'--published']
runpy.run_path(sys.argv[0],run_name='__main__')
