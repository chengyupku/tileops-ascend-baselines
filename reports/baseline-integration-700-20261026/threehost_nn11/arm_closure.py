from pathlib import Path
import json,hashlib,shlex
P=Path(__file__).resolve().parent;B=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build')
link=shlex.split((B/'build/CMakeFiles/cann-ops-nn-allpairs_linux-aarch64_ascendc_cust_op_proto.dir/link.txt').read_text());deps=set();objects={}
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for a in link:
 if a.endswith('.o'):
  f=B/'build'/a;objects[str(f)]=h(f)
  d=f.with_suffix('.o.d')
  if d.exists():
   for word in shlex.split(d.read_text().replace('\\\n',' ')):
    f=Path(word)
    if f.is_absolute() and f.is_file():deps.add(f)
checks={str(p):h(p) for p in deps};original={str(p.relative_to(B/'source')):h(p) for p in deps if p.is_relative_to(B/'source')}
restore='matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp';original[restore]=h(P/'sources/ascend_baselines/ops-nn'/restore)
(P/'ARM_LINK_CLOSURE.json').write_text(json.dumps({'dependency_sha256':checks,'source_sha256_after_original_restore':original,'historical_object_sha256':objects,'restored_object_sha256':h(P/'original_infershape.o'),'restored_library_sha256':h(P/'libcust_opsproto_rt2.0.so')},indent=2));print(len(deps),'dependencies',len(original),'source inputs')
