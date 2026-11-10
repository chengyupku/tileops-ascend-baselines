import pathlib,json,subprocess,hashlib
roots=['/usr/local/Ascend/cann-9.1.0','/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2']
for root in roots:
 p=pathlib.Path(root);print('ROOT',root,'RESOLVED',p.resolve())
 for sub in ['asc','aarch64-linux/asc','tools/bisheng_compiler/bin/bisheng','compiler/ccec_compiler/bin/bisheng','aarch64-linux/bin/bisheng','version.cfg','version.info','aarch64-linux/asc/version.info']:
  f=p/sub;print('PATH',str(f),'EXISTS',f.exists(),'RESOLVED',str(f.resolve()))
  if f.is_file() and 'version' in sub: print(f.read_text(errors='replace')[:2500])
 print(subprocess.run(['find',root,'-maxdepth','6','-name','bisheng'],capture_output=True,text=True).stdout)
 for asc in [p/'asc',p/'aarch64-linux/asc']:
  if asc.exists():
   x=subprocess.run(['grep','-R','-n','-E','asc_pack_to_high|asc_pack_to_low|asc_storealign_pack_quarter|asc_load_l2_cache_mode',str(asc/'include')],capture_output=True,text=True)
   print('SYMBOLS',str(asc),x.returncode,x.stdout[:18000],x.stderr[:1000])
 for f in p.glob('**/bin/bisheng'):
  print('BISHENG',str(f),str(f.resolve()),hashlib.sha256(f.read_bytes()).hexdigest())
  print(subprocess.run([str(f),'--version'],capture_output=True,text=True).stdout)
