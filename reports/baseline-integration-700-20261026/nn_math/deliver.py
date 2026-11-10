import datetime,json,pathlib,hashlib,collections
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];r=json.loads((P/'RESULT.json').read_text());scope=json.loads((P/'SCOPE.json').read_text());start=json.loads((P/'STARTED.json').read_text())['utc'];now=datetime.datetime.now(datetime.timezone.utc)
regs=[]
for d in P.iterdir():
 if d.is_dir() and (d/'REGISTRATION.json').exists():regs.append(json.loads((d/'REGISTRATION.json').read_text()))
keys={(x['baseline'],x['operator'],x['case_id']) for x in r['rows']};pending=[x for x in scope if (x['baseline'],x['operator'],x['case_id']) not in keys]
receipts=[]
for d in P.iterdir():
 if d.is_dir() and (d/'GUARD_RESULT.json').exists():
  g=json.loads((d/'GUARD_RESULT.json').read_text())
  if 'batch_guard' not in g:receipts.append(dict(path=str(d/'GUARD_RESULT.json'),**g))
out=dict(host='local',workspace=str(W),scope_count=len(scope),updated_at=now.isoformat(),elapsed_wall_s=(now-datetime.datetime.fromisoformat(start)).total_seconds(),summary=dict(collections.Counter(x['status'] for x in r['rows'])),in_progress=r.get('in_progress',True),unprocessed_keys=[[x['baseline'],x['operator'],x['case_id']] for x in pending],registered_mappings=regs,guards=receipts,changed_files=sorted(set([x['module'] for x in regs]+[str(P.relative_to(W))])),retired_unregistered_modules=['integration700_nn_math_ada01.py','integration700_nn_math_w401.py'],reproduction={'guard_prefix':"npu-guard run --owner 'baseline700-20261026/nn_math' --wait 3600 --timeout 300 --",'validation':'Each attempt COMMAND.json or parent batch COMMAND.json; run_case.py private route; do not replay completed cases.','formal_provider':'benchmarks.ascend_all_pairs.ops_nn.PROVIDERS[operator](case)','environment':{'OPS_NN_VENDOR':str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'),'ASCEND_CUSTOM_OPP_PATH':str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'),'PYTHONDONTWRITEBYTECODE':'1','FLA_NPU_DISABLE_PTH':'1'},'checker':str(W/'TileOPs/scripts/ascend_pair_check.py'),'note':'Formal provider uses source-identical relocated factory; registration verified device-free without duplicate NPU replay.'},unresolved=[dict(key=[x['baseline'],x['operator'],x['case_id']],status=x['status'],reason=x['reason'],evidence=x.get('evidence')) for x in r['rows'] if x['status']!='completed'],source_comparison=str(P/'SOURCE_COMPARE.json'),memory_modified=False,vendor_modified=False)
t=P/'DELIVERY.tmp';t.write_text(json.dumps(out,indent=2));t.replace(P/'DELIVERY.json')
print(out['summary'])
