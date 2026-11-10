import pathlib,json,subprocess,shlex
O=pathlib.Path(__file__).parent;S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code='''import pathlib,json,hashlib
p=pathlib.Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json');j=json.loads(p.read_text());cfg=j['groups']['attention'];selected={}
for s,h in cfg['pins'].items():
 if any(s.endswith(x) for x in ['libcust_opapi.so','libnnopbase.so','ascend_graph_call.py','ascend_transformer_graph.py','ascend_transform_baselines.py','flash_attention_score_apt.py','flash_attention_score.cpp','flash_attention_score.h']):
  q=pathlib.Path(s);selected[s]={'expected':h,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
print(json.dumps({'manifest_path':str(p),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pin_count':len(cfg['pins']),'selected_pins':selected,'sources':j['sources']}))
'''
r=subprocess.run([S,'openlibing-DevEnv-055409','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=60);(O/'ATTENTION_MANIFEST_AUDIT.json').write_text(r.stdout);print(r.returncode,r.stderr)
