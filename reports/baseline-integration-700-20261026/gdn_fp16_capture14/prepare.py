from pathlib import Path
import shutil,json,datetime,ast
W=Path(__file__).resolve().parent; prior=W.parent/'gdn_fp16_recovery14'
shutil.copytree(prior/'package',W/'package')
shutil.copy2(prior/'ALL_CONTRACTS.json',W/'ALL_CONTRACTS.json')
shutil.copy2(prior/'metadata_optional_cpu.patch',W/'metadata_optional_cpu.patch')
r=json.loads((W/'package/RUNTIME_RECIPE.json').read_text());owner='baseline700-20261026/gdn_fp16_capture14'
r['required_environment']['GDN_CASE_OWNER']=owner;r['required_environment']['ASCEND_OPP_PATH']=r['opp'];r['guard']['owner']=owner
(W/'package/RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2))
p=W/'package/binding.py';s=p.read_text().replace('import json,hashlib','import json,hashlib,os')
s=s.replace('  assert a_log is None and dt_bias is None','  capture=Path(os.environ["CAPTURE_DIR"])\n  torch.save(tuple(x.detach().cpu().clone() if isinstance(x,torch.Tensor) else x for x in (q,k,v,g,beta,initial,offsets,offsets_cpu,a_log,dt_bias)),capture/"inputs.pt")\n  assert a_log is None and dt_bias is None')
s=s.replace('  return out,final','  torch.npu.synchronize()\n  torch.save((out.cpu(),final.cpu()),capture/"outputs.pt")\n  torch.save((aqk.cpu(),akk.cpu()),capture/"optional_aqk_akk.pt")\n  return out,final');p.write_text(s)
p=W/'package/entry.py';s=p.read_text();s=s.replace('from binding import build','from binding import build\nimport capture_hooks\ncapture_hooks.install()');p.write_text(s)
(W/'STARTED.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker':'d70b81f07c8942e1','owner':owner,'scope':14,'authorized_probes':['decode-b1-float16','varlen-ragged-grouped-float16'],'api_queue_seconds':None},indent=2))
for p in (W/'package').glob('*.py'):ast.parse(p.read_text())
