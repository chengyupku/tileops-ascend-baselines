from pathlib import Path
import json,datetime
p=Path(__file__).parent;r=json.loads((p/'RESULT.json').read_text());lookup={(x['baseline'],x['operator'],x['case_id']):x for x in r['rows']}
for f in [*p.glob('REGISTRATION_SIBLINGS*.json'),*p.glob('REGISTRATION_FINAL*.json')]:
 reg=json.loads(f.read_text())
 for entry in reg['rows']:
  key=tuple(entry['key']);row=lookup[key];receipt=entry['checker_receipt'];c=json.loads((p/receipt).read_text());assert c['status']=='verified';folder=Path(receipt).parts[0];guard={'all_siblings':'all_siblings_guard.log','quant_prefill_qkv':'final_build_guard.log','quant_prefill_down':'last_checks_guard.log','fp8_bmm_half_check':'last_checks_guard.log'}.get(folder,folder+'_guard.log');snap=p/folder/'guard_snapshot.json';q=None
  if snap.exists():
   d=json.loads(snap.read_text());q=(datetime.datetime.fromisoformat(d['acquired_at'])-datetime.datetime.fromisoformat(d['requested_at'])).total_seconds()
  tag=('final'+f.stem.removeprefix('REGISTRATION_FINAL')) if f.stem.startswith('REGISTRATION_FINAL') else ('siblings'+f.stem.removeprefix('REGISTRATION_SIBLINGS'));mod=entry['factory'].split('.')[-2]+'.py'
  row.update(status='completed',registered=True,reason='Original complete quantized GroupedMatmul API passed canonical full checker; exact-key sibling registered under provider lock.',factory=entry['factory'],factory_sha256=entry['module_sha256'],guard_receipt=guard,checker_receipt=receipt,checked_outputs=c['verification']['checked_outputs'],total_outputs=c['verification']['total_outputs'],changed_files=[mod,f'provider_init.{tag}.patch'],timings={'queue_s':q,'device_s':None,'compile_s':None,'checker_wall_s':c['wall_seconds']})
  row['evidence']=list(dict.fromkeys(row['evidence']+[f.name,receipt]))
r.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=sum(x['status']=='completed' for x in r['rows']));(p/'RESULT.tmp').write_text(json.dumps(r,indent=2));(p/'RESULT.tmp').replace(p/'RESULT.json');print('completed',r['completed'])
