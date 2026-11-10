from pathlib import Path
import json,hashlib,shutil,subprocess,shlex
P=Path(__file__).resolve().parent
B=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins=json.loads((P/'ORIGINAL_MATMUL_PINS.json').read_text());checks={k:{'expected':v,'actual':h(B/'source'/k) if (B/'source'/k).is_file() else None} for k,v in pins.items()};bad=[k for k,v in checks.items() if v['actual']!=v['expected']]
assert bad==['matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp'],bad
changed=bad[0];original=P/'sources/ascend_baselines/ops-nn'/changed;assert h(original)==pins[changed]
(P/'ARM_SOURCE_BEFORE.json').write_text(json.dumps(checks,indent=2))
(P/'INFER_RESTORE.diff').write_text(subprocess.run(['diff','-u',str(B/'source'/changed),str(original)],capture_output=True,text=True).stdout)
cmds=json.loads((B/'build/compile_commands.json').read_text());selected=[x for x in cmds if x['file']==str(B/'source'/changed)]
assert len(selected)==1,selected
args=shlex.split(selected[0]['command']);oi=args.index('-o')+1;oldobj=args[oi];obj=P/'original_infershape.o';args[oi]=str(obj);args=[str(original) if a==str(B/'source'/changed) else a for a in args]
# Preserve source-relative includes while restoring original source text.
args.extend(['-I',str((B/'source'/changed).parent)])
linkpath=B/'build/CMakeFiles/cann-ops-nn-allpairs_linux-aarch64_ascendc_cust_op_proto.dir/link.txt';link=shlex.split(linkpath.read_text());matches=[a for a in link if a.endswith('.o') and (B/'build'/a).resolve()==(Path(selected[0]['directory'])/oldobj).resolve()]; assert len(matches)==1,matches; oldobj=matches[0]
link=[str(obj) if a==oldobj else a for a in link];link[link.index('-o')+1]=str(P/'libcust_opsproto_rt2.0.so')
(P/'ARM_BUILD_COMMANDS.json').write_text(json.dumps({'compile':args,'link':link,'directory':selected[0]['directory']},indent=2))
with (P/'arm_build.log').open('w') as f:
 subprocess.run(args,cwd=selected[0]['directory'],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(link,cwd=B/'build',stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copytree(B/'install3/vendors/allpairs_nn',P/'vendor/allpairs_nn',symlinks=True)
lib=P/'vendor/allpairs_nn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so';shutil.copy2(P/'libcust_opsproto_rt2.0.so',lib)
for n in ['NATIVE_READY.json','PACKAGED_SOURCE_COMPARISON.json','ORIGINAL_JIT_REUSE.json','ELF_DEPENDENCIES.json']:shutil.copy2(B/n,P/('UPSTREAM_'+n))
files={str(f):h(f) for f in (P/'vendor').rglob('*') if f.is_file()}
(P/'ARM_DEPLOYMENT.json').write_text(json.dumps({'status':'original_source_restored_private_arm_proto','source_checks':checks,'restored_original':{str(original):h(original)},'compile_commands':str(P/'ARM_BUILD_COMMANDS.json'),'files':files,'elf':subprocess.check_output(['readelf','-h',str(lib)],text=True),'old_proto_sha256':h(B/'install3/vendors/allpairs_nn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so'),'new_proto_sha256':h(lib)},indent=2))
print('Restored original ARM proto',h(lib),flush=True)
