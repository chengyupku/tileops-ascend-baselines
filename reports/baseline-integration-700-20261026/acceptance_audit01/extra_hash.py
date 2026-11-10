import pathlib,json,hashlib,subprocess,shlex
O=pathlib.Path(__file__).parent;E=json.load(open(O/'EVIDENCE_INDEX.json'));S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';W='/opt/tileops-all-pairs/workspace';req={'server1':{},'server2':{}}
for x in E:
 if x['host']=='local':continue
 for p,h in x['oracle'].items():req[x['host']][p if p.startswith('/') else W+'/'+p]=h
 for p,h in x['source'].items():
  if x['worker']=='tla_simple_tilelang' or p.endswith('liboriginal_bias_gate_gelu.so') or p.endswith('BIAS_GATE_GELU.cpp') or p.endswith('flash_attention_score_apt.py') or p.endswith('moe_gating_top_k_apt.py') or p.endswith('rotary_position_embedding_apt.py'):
   req[x['host']][p if p.startswith('/') else W+'/'+p]=h
old=json.load(open(O.parent/'PARENT_GDN_ROUTE_CHECK_1343.json'))
req['server2'].update(old['hashes'])
for host,paths in req.items():
 code='import json,pathlib,hashlib\nx=json.loads('+repr(json.dumps(paths))+')\nout={}\nfor s,h in x.items():\n p=pathlib.Path(s)\n try:out[s]={"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"expected":h}\n except Exception as e:out[s]={"error":str(e),"expected":h}\nprint(json.dumps(out))'
 r=subprocess.run([S,{'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host],'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=60);(O/(host+'_extra_hashes.json')).write_text(r.stdout);print(host,r.returncode)
