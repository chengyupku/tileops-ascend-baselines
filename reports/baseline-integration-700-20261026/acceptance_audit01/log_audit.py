import pathlib,json,re,hashlib
O=pathlib.Path(__file__).parent;B=O.parent;E=json.load(open(O/'EVIDENCE_INDEX.json'));out={}
# Whole explicitly named shared-process directories are retained, including later cases' launches.
groups={
 'attention_gqa128':('transformer_attention_misc/runs/qwen3-30b-128k-float16','flashattentionscore'),
 'attention_gqa64':('transformer_attention_misc/runs/qwen3-30b-64k-float16','flashattentionscore'),
 'mha_b2':('transformer_attention_misc/runs/serving-b2-float16','flashattentionscore'),
 'mha_batch':('transformer_attention_misc/runs/serving-kv1k-float16','flashattentionscore'),
 'mla_b1':('transformer_attention_misc/runs/mla-tp-b1-float16','flashattentionscore'),
 'mla_batch':('transformer_attention_misc/runs/mla-b128-4096-bfloat16','flashattentionscore'),
 'topk':('transformer_dense/topk_check_graph','moegatingtopk'),
 'longrope':('transformer_dense/rope_check_graph_optional','rotarypositionembedding'),
 'yarn':('transformer_dense/rope_check_graph_cpu_state','rotarypositionembedding'),
 'gelu':('transformer_dense','bias_gate_gelu')}
for name,(d,k) in groups.items():
 root=B/d; matches=[]
 logs=(root/'ASCEND_PROCESS_LOG_PATH').rglob('*.log')
 for p in logs:
  hits=[]
  with p.open(errors='replace') as f:
   for n,line in enumerate(f,1):
    low=line.lower()
    if k in low and ('StreamLaunchKernel' in line or ('LogTaskDetails' in line and 'File[' in line)):
     hits.append({'line':n,'text':line.strip()})
  if hits:matches.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'hits':hits})
 out[name]=matches
 print(name,sum(len(x['hits']) for x in matches))
(O/'NATIVE_LOG_AUDIT.json').write_text(json.dumps(out,indent=2))
