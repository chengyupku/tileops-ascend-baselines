"""Publish only this task's exact-case wrapper under the existing provider lock."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
W = Path(sys.argv[1])
N = W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
code = (P/'provider.py').read_text()
append = "\n# Single-agent server2: original NSA compressed attention, three exact cases.\nfrom .single_server2_nsa_compressed import wrap as _single_server2_nsa_wrap\nPROVIDERS['NSACompressedVarlenFwdOp'] = _single_server2_nsa_wrap(PROVIDERS['NSACompressedVarlenFwdOp'])\n"
ast.parse(code)
with open('/tmp/baseline700-provider-ops_transformer.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    before = (N/'__init__.py').read_text()
    if '_single_server2_nsa_wrap' in before:
        raise RuntimeError('Already published; do not append twice')
    ast.parse(before + append)
    (P/('provider_init_before_' + sys.argv[2] + '.py')).write_text(before)
    (N/'single_server2_nsa_compressed.py').write_text(code)
    temp = N/'__init__.single-server2-nsa.tmp'
    temp.write_text(before + append)
    temp.replace(N/'__init__.py')
receipt = {'host': sys.argv[2], 'provider': str(N/'single_server2_nsa_compressed.py'),
    'sha256': hashlib.sha256(code.encode()).hexdigest(),
    'provider_before_sha256': hashlib.sha256(before.encode()).hexdigest(),
    'provider_after_sha256': hashlib.sha256((before+append).encode()).hexdigest(),
    'prior_prefix_preserved': True, 'new_device_runs': 0,
    'lock': '/tmp/baseline700-provider-ops_transformer.lock'}
(P/('PUBLICATION_' + sys.argv[2] + '.json')).write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
