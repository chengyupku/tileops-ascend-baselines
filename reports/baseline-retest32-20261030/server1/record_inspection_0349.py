from pathlib import Path
import subprocess,json,time,datetime,fcntl
h=Path(__file__).resolve().parent;x=json.loads((h/'INSPECTION.latest.json').read_text());assert x['process']['pid']==2847187 and x['process']['matches'] and not x['pending_review'];assert json.loads((h/'MLA_FP16_AUDIT_0349.json').read_text())['all_eight_accepted']
code="""from pathlib import Path
import json,shutil,time
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
print(json.dumps({'at':time.time(),'free_bytes':shutil.disk_usage(h).free,'status':json.loads((h/'DISK_STATUS.json').read_text()),'policy':json.loads((h/'DISK_POLICY.json').read_text())}))
"""
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True);d=json.loads(r.stdout);timers=subprocess.check_output(['systemctl','is-active','baseline-retest32-server1-archive.timer','baseline-retest32-server1-halfhour-20261030.timer'],text=True).splitlines()
report={'recorded_at':time.time(),'alarm_due':'2026-11-01T03:49:06.992535+00:00','remote_checked_at':x['checked_at'],'remote_checked_utc':datetime.datetime.fromtimestamp(x['checked_at'],datetime.timezone.utc).isoformat(),'cached':x['cached'],'remaining':x['remaining'],'pending':x['pending_review'],'identity':x['process'],'guard':x['guard'],'phases':x['phases'],'disk':d,'free_GiB':d['free_bytes']/2**30,'timers':timers,'migration':json.loads((h/'MIGRATION_STATUS.json').read_text()),'actions':['Actual inspection PID2847187/start248630394,hostguard/device,disk and latest journal/checker verified','New FP16 MLA f9d73684ed9ede64777a fullchecker/profile/valid32core cache audited MLA_FP16_AUDIT_0349.json;1186cached,1remaining,pending0','Last mixedBF16 d36b75a34eb33230af41 child1737845 sampled nonblocking;MLA_LOCALS_0349.txt originalCPUref_program finalsequence15872:32256,h0=40,r0=11648,offsets0/512/1536/3584/7680/15872/32256;continue full check,not restart','Mandatory complete receipts incrementally collected;all environments deployed;no cached reruns;continuousmigration and halfhour timersactive;50GiBfloor/50.5admission unchanged']}
name='DIAGNOSIS_20261101_0349.json';(h/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');p=h/'TAKEOVER.json';t=json.loads(p.read_text());t.update(updated_at=time.time(),last_inspection=str(h/name),identity=x['process'],session=97191,status='v73_running_final_environment_mla_retry',known_held=f"{x['cached']}cached,{x['remaining']}remaining,pending0;FP16 accepted;last mixedMLA running;{report['free_GiB']:.2f}GiBfree",repair_audit=str(h/'MLA_FP16_AUDIT_0349.json'),next_deployment_review=str(h/'QUANT_GEMM_ROUTE_REVIEW.json'));p.write_text(json.dumps(t,ensure_ascii=False,indent=2)+'\n')





with Path('.agent_memory/THREADS/baseline-inspector-20261023/MEMORY.md').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 f.write(f'\n\n## 2026-11-01 一号03:49巡检：仅剩混长MLA\n\n实际检查DIAGNOSIS_20261101_0349.json；{x["cached"]}/1187、{x["remaining"]}未完成、pending0。FP16 MLA f9d73684ed9ede64777a完整checker/profile/有效guard32核cache已审计MLA_FP16_AUDIT_0349.json。v73 PID2847187/start248630394/session97191真实匹配；最后混长BF16 d36b75a34eb33230af41正在完整检查，子PID1737845，MLA_LOCALS_0349.txt非阻塞采样原CPU ref_program最后序列begin15872/end32256、h0=40/r0=11648，offsets=[0,512,1536,3584,7680,15872,32256]，仍推进。保持3600秒及原完整checker，不重启。环境已齐无pending；磁盘{report["free_GiB"]:.2f}GiB，持续迁移和半小时timer active。完整回执增量收回；原内核/compiler/schedule/checker不改，cached不重跑；全部1187完整验收才关timer。\n')
 f.flush();fcntl.flock(f,fcntl.LOCK_UN)
print({'cached':x['cached'],'remaining':x['remaining'],'free_GiB':report['free_GiB']})
