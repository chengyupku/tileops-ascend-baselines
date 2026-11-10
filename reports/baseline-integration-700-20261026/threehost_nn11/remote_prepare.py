from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
P=Path(__file__).resolve().parent
B=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins=json.loads((P/'ORIGINAL_MATMUL_PINS.json').read_text());checked={k:{'expected':v,'actual':h(B/'source'/k) if (B/'source'/k).is_file() else None} for k,v in pins.items()}
assert all(v['actual']==v['expected'] for v in checked.values()),[k for k,v in checked.items() if v['actual']!=v['expected']]
link=B/'build/CMakeFiles/cann-ops-nn-allpairs_linux-aarch64_ascendc_cust_op_proto.dir/link.txt'
commands=json.loads((B/'build/compile_commands.json').read_text())
(P/'ARM_SOURCE_AUDIT.json').write_text(json.dumps({'source_files':checked,'proto_link_command':link.read_text(),'compile_commands':commands,'generated_proto':(B/'build/tbe/graph/ops_proto_nn.cpp').read_text(),'generated_exc_proto':(B/'build/autogen/exc/mat_mul_v3_proto.cpp').read_text(),'note':'Original ARM build provenance checked; not local known patched x86 install3. No binary copied across architecture.'},indent=2))
for name in ['NATIVE_READY.json','PACKAGED_SOURCE_COMPARISON.json','ORIGINAL_JIT_REUSE.json','SYMBOL_CHECK.json','ELF_DEPENDENCIES.json','ARTIFACT_SHA256.json']:
 shutil.copy2(B/name,P/('UPSTREAM_'+name))
shutil.copytree(B/'install3/vendors/allpairs_nn',P/'vendor/allpairs_nn',symlinks=True)
so=list((P/'vendor').rglob('*.so'))
elf={str(f):subprocess.check_output(['file',str(f)],text=True) for f in so}
assert all('ARM aarch64' in s or 'symbolic link' in s for s in elf.values()),elf
(P/'ARM_DEPLOYMENT.json').write_text(json.dumps({'elf':elf,'files':{str(f):h(f) for f in (P/'vendor').rglob('*') if f.is_file()},'at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('original source checked',len(checked),'ARM libraries',len(so))
