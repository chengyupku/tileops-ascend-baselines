import json,pathlib,hashlib
W=pathlib.Path('/root/dyq/workspace');P=W/'reports/baseline-integration-700-20261026';O=P/'acceptance_audit02'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
comp=json.loads((P/'nn_recovery01/CURRENT_SOURCE_COMPARE.json').read_text());bad=[]
for r in comp:
 for k in ['source','build']:
  if sha(r[k])!=r['sha256']:bad.append(r[k])
rows=[];seen=set()
for r in json.loads((P/'nn_registration01/REGISTRATION.json').read_text())['rows']:
 op=r['key'][1]
 if op in seen:continue
 seen.add(op);c=json.loads(pathlib.Path(r['checker']['path']).read_text());target={'FusedAddRMSNormFwdOp':'norm/inplace_add_rms_norm','FusedAddLayerNormFwdOp':'norm/add_layer_norm','GemmW4A16FwdOp':'matmul/weight_quant_batch_matmul_v2','GemmFP8FwdOp':'matmul/quant_batch_matmul_v3'}[op];name=target.split('/')[-1];checked=0;issues=[];primary={}
 for p,h in c['source_sha256'].items():
  if '/ascendc/'+name+'/' not in p and not p.endswith('.so'):continue
  fp=W/p
  if not fp.exists():issues.append('missing '+p);continue
  if sha(fp)!=h:issues.append('recorded hash mismatch '+p)
  if '/ascendc/'+name+'/' in p:
   orig=W/'ascend_baselines/ops-nn'/target/p.split('/ascendc/'+name+'/')[1]
   if not orig.exists():orig=W/'ascend_baselines/ops-nn'/target/'op_kernel'/p.split('/ascendc/'+name+'/')[1]
   if not orig.exists() or sha(orig)!=h:issues.append('original mismatch '+str(orig))
   checked+=1
  else:primary[p]=h
 rows.append({'op':op,'original_kernel_source_files_checked':checked,'runtime_primary_hashes':primary,'issues':issues})
out={'build_original_files_checked':len(comp),'build_original_mismatches':bad,'families':rows};(O/'NN_SOURCE_CHECK.json').write_text(json.dumps(out,indent=2));print([(r['op'],r['original_kernel_source_files_checked'],r['issues']) for r in rows])
