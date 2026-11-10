import json,hashlib,pathlib,re
O=pathlib.Path(__file__).resolve().parent
N=O.parent/'threehost_tilelang_bmm_native3'
keys=['mha-decode-pv-bfloat16','mha-decode-pv-float16','moe-prefill-bfloat16']
rows=json.loads((N/'RESULT.json').read_text()); assert len(rows)==3
checks=[]
for row in rows:
 assert row['status']=='verified' and row['inputs_preserved'] is True and row['dispatch_count'] in (64,128)
 assert row['baseline']=='tilelang' and row['operator']=='BmmFwdOp'
 seal=json.loads((N/row['native_seal']).read_text())
 # seal must retain pointer/bytes/maps evidence and completion linkage
 blob=json.dumps(seal)
 assert '__tvm_ffi_main' in blob and 'maps' in blob.lower()
 assert row['native_artifacts'][0]['binary_bytes']==21096
 assert row['native_artifacts'][0]['sha256']
 checks.append({'key':row['case_id'],'seed':row['input_seed'],'dispatch_count':row['dispatch_count'],'pointer':row['native_artifacts'][0]['pointer'],'binary_sha256':row['native_artifacts'][0]['sha256'],'binary_bytes':row['native_artifacts'][0]['binary_bytes'],'map':row['native_artifacts'][0]['map'],'stat_device':row['native_artifacts'][0]['stat_device'],'maps_device':row['native_artifacts'][0]['maps_device'],'inputs_preserved':row['inputs_preserved']})
# targeted checker linkage and hostguard receipts
for k in keys:
 c=next(r for r in rows if r['case_id']==k)
 assert (N/c['checker_receipt']).is_file()
 g=N/c['guard_receipt']; assert g.is_file()
assert json.loads((N/'DELIVERY.json').read_text())['dependency_files']==1857
(O/'PARENT_READY_PROPOSAL.json').write_text(json.dumps({'status':'ready_for_parent_review','baseline':'tilelang','operator':'BmmFwdOp','keys':keys,'accepted_added':0,'native_receipts_verified':True,'checks':checks,'historical_boundary':'native3 closure has 1857 dependency files; this package hashes only its frozen source/PINS and receipt metadata; no whole-system rescan','minimal_gap':'parent must review deployment and formal shared registry publication'},indent=2)+'\n')
(O/'RECEIPT_AUDIT.json').write_text(json.dumps({'passed':True,'hostguard':'all three guard receipts present; original DELIVERY says hostguard0/uncontaminated','rows':checks,'accepted_added':0},indent=2)+'\n')
print(json.dumps({'passed':True,'rows':len(checks)}))
