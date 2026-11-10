import pathlib,subprocess,json,os,sys
P=pathlib.Path(__file__).resolve().parent;cid=sys.argv[1]; key=['ops-transformer','GQADenseFwdOp',cid]
assert key in [r['key'] for r in json.loads((P/'SCOPE.json').read_text())]
subprocess.run(['python3',str(P/'inspect_remote.py')],check=True,stdout=subprocess.DEVNULL)
for row in json.loads((P/'EXACT_HISTORY.json').read_text()):
 if row.get('key')==key:assert '/amnesty_keys/' not in row['location']
for receipt in [*P.glob('runs/*/output/checker.json'),*P.glob('batches/*/*/checker.json')]:
 d=json.loads(receipt.read_text());g=receipt.parents[1]/'GUARD_RESULT.json'
 if [d.get('library'),d.get('op'),d.get('case_id')]==key and d.get('status')=='verified' and g.exists() and json.loads(g.read_text()).get('returncode')==0:
  raise SystemExit('Exact key already verified; preserve existing receipt: '+str(receipt))
case=P/'runs'/(cid+('--'+sys.argv[2]) if len(sys.argv)>2 else cid);case.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
for var in ['TMPDIR','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
 dest=case/var;dest.mkdir();env[var]=str(dest)
env['PYTHONDONTWRITEBYTECODE']='1'
cmd=['npu-guard','run','--owner','selected-integration-20261026/transformer_attention','--wait','3600','--timeout','300','--','/opt/tileops-all-pairs/bin/allpairs-python',str(P/'run_case.py'),cid,str(case/'output')]
(case/'COMMAND.json').write_text(json.dumps(cmd,indent=2))
with (case/'guard.log').open('x') as f:rc=subprocess.run(cmd,env=env,cwd=case,stdout=f,stderr=subprocess.STDOUT).returncode
(case/'GUARD_RESULT.json').write_text(json.dumps({'returncode':rc}));print(json.dumps({'case_id':cid,'guard_exit_code':rc}));sys.exit(rc)
