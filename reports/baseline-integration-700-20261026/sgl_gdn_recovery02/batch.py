import sys,runpy,json,datetime,time
from pathlib import Path
R=Path(__file__).resolve().parent
batch=Path(sys.argv[1]); rows=json.loads(batch.read_text()); start=time.monotonic()
for row in rows:
    if (R/'FEEDBACK.txt').exists(): print('FEEDBACK', (R/'FEEDBACK.txt').read_text(),flush=True)
    t=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (R/'HEARTBEAT.json').write_text(json.dumps(dict(utc=t,phase='device_batch',current_keys=[row],last_progress_at=t,blocked_reason=None)))
    op,cid=row; work=R/'runs'/f'{op}--{cid}'; work.mkdir(exist_ok=False)
    sys.argv=[str(R/'run_case.py'),'sgl-kernel-npu',op,cid,str(work/'output')]
    try: runpy.run_path(str(R/'run_case.py'),run_name='__main__')
    except SystemExit: pass
    receipt=json.loads((work/'output/checker.json').read_text())
    expected='sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_'
    assert receipt.get('api','').startswith(expected),receipt
    assert any(k.endswith('integration700_sgl_gdn_recovery02.py') for k in receipt['source_sha256']),receipt
    print('BATCH_CASE',op,cid,receipt['status'],flush=True)
    if receipt['status']!='verified': break
    if time.monotonic()-start>240: break
