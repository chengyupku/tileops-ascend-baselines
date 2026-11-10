import pathlib,urllib.request,urllib.parse,json,hashlib,zipfile,time,os
files=[{'url': 'https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl', 'sha256': '6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604'}, {'url': 'https://files.pythonhosted.org/packages/cf/74/cfc6bc7d3b251b8e0942ad27e966ac8131e286da73c987a0ff63db451bbe/torch_npu-2.13.0rc1-cp312-cp312-manylinux_2_28_aarch64.whl', 'sha256': '6f7da48ba3db796b06ded6087e1a9a1827af4cada0fb298be76451e5947f10a8'}]
r=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-retest32-20261030/server2/torch213_env');r.mkdir(exist_ok=True);(r/'downloads').mkdir(exist_ok=True);overlay=r/'site-packages';overlay.mkdir(exist_ok=True);rows=[]
for row in files:
 name=urllib.parse.unquote(row['url'].rsplit('/',1)[1]);p=r/'downloads'/name
 if not p.exists():
  t=p.with_suffix('.partial')
  with urllib.request.urlopen(row['url'],timeout=60) as src,open(t,'wb') as dst:
   while True:
    b=src.read(4*1024*1024)
    if not b:break
    dst.write(b)
   dst.flush();os.fsync(dst.fileno())
  assert hashlib.sha256(t.read_bytes()).hexdigest()==row['sha256'];t.rename(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 with zipfile.ZipFile(p) as z:z.extractall(overlay)
 rows.append({**row,'file':str(p),'size':p.stat().st_size});print(name,'verified and extracted',flush=True)
(r/'VERIFIED.json').write_text(json.dumps({'at':time.time(),'files':rows,'status':'CPU_import_pending','original_environment_unchanged':True},indent=2))
