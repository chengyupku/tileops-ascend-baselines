import sys,json,platform
from pathlib import Path
ROOT=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration')
sys.path.insert(0,str(ROOT))
import registration as M
pins=M.metadata()['PINS.json']
assert platform.machine()==pins['host']['machine'] and platform.node()==pins['host']['node']
M.require_guard();M.snapshot(runtime=True)
print('native9 frozen closure preflight passed',flush=True)
