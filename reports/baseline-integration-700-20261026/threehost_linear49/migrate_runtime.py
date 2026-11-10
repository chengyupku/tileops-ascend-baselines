from pathlib import Path
import subprocess,json,time
P=Path(__file__).resolve().parent;r=json.loads((P/'ORIGINAL92_RECIPE.json').read_text());ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';D='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_linear49'
roots=[r['vendor'],r['ld_library_path_order'][1],r['required_environment']['ASCEND_OPP_PATH']]
# CPU-only original AArch64 package copy. No x86 libraries or fabricated metadata.
t=time.time()
cmd='tar -chf - '+ ' '.join('--transform='+__import__('shlex').quote('s,^'+x.lstrip('/')+','+n+',')+' '+__import__('shlex').quote(x) for x,n in zip(roots,['vendor','opapi-lib','opp']))
# Independent transforms would also match renamed prefixes only once; these source roots are disjoint.
with (P/'runtime92_aarch64.tar.gz').open('wb') as f:
 a=subprocess.Popen([ssh,'openlibing-DevEnv-055409',cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 b=subprocess.Popen(['gzip','-1'],stdin=a.stdout,stdout=f);a.stdout.close();err=a.stderr.read();br=b.wait();ar=a.wait()
(P/'migration_archive.log').write_bytes(err);assert ar==br==0,(ar,br)
with (P/'runtime92_aarch64.tar.gz').open('rb') as f:
 subprocess.run([ssh,'openlibing-DevEnv-661495',f'mkdir -p {D}/runtime92 && tar -xzf - -C {D}/runtime92'],stdin=f,check=True)
r['vendor']=D+'/runtime92/vendor';r['opapi']=r['vendor']+'/op_api/lib/libcust_opapi.so';r['required_environment']['ASCEND_OPP_PATH']=D+'/runtime92/opp';r['ld_library_path_order'][1]=D+'/runtime92/opapi-lib'
subprocess.run([ssh,'openlibing-DevEnv-661495',f'cat > {D}/ORIGINAL92_RECIPE.json'],input=json.dumps(r,indent=2).encode(),check=True)
subprocess.run([ssh,'openlibing-DevEnv-661495',f'cat > {D}/remote_run.py'],input=(P/'remote_run.py').read_bytes(),check=True)
(P/'MIGRATION_RECEIPT.json').write_text(json.dumps({'seconds':time.time()-t,'archive_bytes':(P/'runtime92_aarch64.tar.gz').stat().st_size,'from':'server2','to':'server1','same_arch':'aarch64','original_roots':roots,'private_destination':D+'/runtime92','kernel_changes':False},indent=2));print('migration complete',flush=True)
