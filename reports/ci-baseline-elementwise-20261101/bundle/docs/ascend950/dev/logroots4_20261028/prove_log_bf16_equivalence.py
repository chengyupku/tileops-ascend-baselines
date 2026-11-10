"""Prove unchanged selected kernels before adopting the BF16 logarithm fix."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
import torch
import torch_npu
from tileops.kernels.ascend.elementwise import _logroots as old
from candidate import _logroots_v10 as new

torch.npu.set_device(0)
old_hash = hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()
new_hash = hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest()
assert old_hash == '995d5405dce1be237694944bcdff0988e922a86e86b0a309fe81585bda8b8b55'
records = []
out = HERE / 'search/source-equivalence-logbf16-v10'
out.mkdir(exist_ok=False)
for row in json.loads((HERE / 'proposal-v5-configs.json').read_text()):
    op, case = row['op'], row['case_id']
    size, dtype = case.split('-')
    if op == 'LogFwdOp' and dtype == 'bfloat16':
        continue
    kind = {'LogFwdOp': 'log', 'Log1pFwdOp': 'log1p', 'SqrtFwdOp': 'sqrt', 'RsqrtFwdOp': 'rsqrt'}[op]
    n = int(size.removesuffix('M')) * 1024 * 1024
    config = row['config']
    args = (n, dtype, kind, config['tile'], config['blocks'], config['stages'])
    before = old.build(*args).get_kernel_source()
    after = new.build(*args).get_kernel_source()
    assert before == after, (op, case, 'generated code changed')
    name = op + '-' + case + '.c'
    (out / name).write_text(after)
    records.append(dict(op=op, case_id=case, config=config, exact_match=True,
                        file=name, sha256=hashlib.sha256(after.encode()).hexdigest()))
    print('EXACT_C', op, case, flush=True)
assert len(records) == 18
assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() == old_hash
assert hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest() == new_hash
proof = dict(old_helper_sha256=old_hash, new_helper_sha256=new_hash,
             guard_label='prove-log-bf16-v10-equivalence', comparisons=records,
             scope='configuration evidence only; final public measurements require current source hashes')
(HERE / 'search/source-equivalence-logbf16-v10.json').write_text(json.dumps(proof, indent=2))
torch.npu.synchronize()
