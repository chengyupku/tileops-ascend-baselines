from pathlib import Path
import numpy as np,json
P=Path(__file__).resolve().parent
# Pure CPU layout proof only. Device acceptance is exclusively the full checker receipt.
rng=np.random.default_rng(17)
y=rng.integers(0,256,size=(4096,512),dtype=np.uint8)
shifts=np.arange(8,dtype=np.uint8)
original=((y.reshape(-1,1)>>shifts)&1).reshape(4096,4096)
packed=np.broadcast_to(y.reshape(-1,1),(2097152,8)).copy(order='C')
expanded=np.broadcast_to(shifts.reshape(1,8),(2097152,8)).copy(order='C')
actual=((packed>>expanded)&1).reshape(4096,4096)
assert np.array_equal(original,actual)
assert not np.shares_memory(y,packed)
x=json.loads((P/'CPU_PROOF.json').read_text());x.update(original_full_shape_cpu_layout_equal=True,original_packed_shape=[4096,512],original_result_shape=[4096,4096],full_layout_elements=16777216,real_device_pending=False,device_evidence='threehost_isnan_unpack1/runs/tilelang-ascend_IsnanFwdOp_16M-float32_expanded/checker.json')
(P/'CPU_PROOF.json').write_text(json.dumps(x,indent=2));print('Full original row layout matches for all 16M unpacked CPU elements')
