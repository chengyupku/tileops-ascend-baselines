import pathlib,json,shutil,difflib,hashlib
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-20261026/nn_norm';P=D/'packet'; old=W/'.cache/all-pairs/full_rollout_20261024/local24/group1/resume25_common_repair/continuation0320/tasks/current-original-nn-norm-launch-repair20/packet'
shutil.copytree(old,P,dirs_exist_ok=True)
s=json.loads((D/'SCOPE.json').read_text());(P/'EXACT_KEYS.json').write_text(json.dumps([x['key'] for x in s]))
v={x['operator']:str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn') for x in s};v['RMSNormFwdOp']=json.loads((old/'VENDORS.json').read_text())['RMSNormFwdOp'];(P/'VENDORS.json').write_text(json.dumps(v,indent=2))
r=(old/'run_case.py').read_text();start=r.index("for name in ['PACKET_PINS.json'");end=r.index("W=Path",start);r=r[:start]+r[end:];r=r.replace('assert work!=P and P not in work.parents','assert work.parent.parent==P.parent');(P/'run_case.py').write_text(r)
norm=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/normalization.py';n=norm.read_text();n=n.replace('out=torch.empty_like(x);mean=torch.empty', 'original_dtype=x.dtype; x=x.float();weight=weight.float();bias=bias.float()\n            out=torch.empty_like(x);mean=torch.empty',1).replace('            return out\n        if name==\'BatchNormFwdOp\'', '            return out.to(original_dtype)\n        if name==\'BatchNormFwdOp\'',1)
(P/'frozen/normalization.py').write_text(n)
(D/'adapter.patch').write_text(''.join(difflib.unified_diff(norm.read_text().splitlines(True),n.splitlines(True),fromfile='a/benchmarks/ascend_all_pairs/ops_nn/normalization.py',tofile='b/benchmarks/ascend_all_pairs/ops_nn/normalization.py')))
(P/'launch.py').unlink()
(D/'PRIVATE_PINS.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*.py')},indent=2))
