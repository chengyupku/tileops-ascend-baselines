from pathlib import Path
import subprocess,tarfile
W=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/transformer_gdn'
excludes=['TMPDIR','PIP_CACHE_DIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','CCACHE_DIR','__pycache__']
cmd='tar '+ ' '.join('--exclude='+x for x in excludes)+' -czf - -C '+remote+' .'
p=subprocess.Popen([ssh,'openlibing-DevEnv-055409',cmd],stdout=subprocess.PIPE)
with tarfile.open(fileobj=p.stdout,mode='r|gz') as t:
 for m in t:
  target=(W/m.name).resolve()
  if not target.is_relative_to(W):raise RuntimeError(m.name)
  t.extract(m,W,filter='data')
assert p.wait()==0
print('Copied receipts, adapters and native logs; remote originals retained.')
