from pathlib import Path
import subprocess,json,time,datetime,fcntl
h=Path(__file__).resolve().parent;x=json.loads((h/'INSPECTION.latest.json').read_text());assert x['process']['pid']==2847187 and x['process']['matches'] and not x['pending_review'];assert json.loads((h/'MLA_KIMI_AUDIT_0249.json').read_text())['all_eight_accepted']
code="""from pathlib import Path
import json,shutil,time
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
print(json.dumps({'at':time.time(),'free_bytes':shutil.disk_usage(h).free,'status':json.loads((h/'DISK_STATUS.json').read_text()),'policy':json.loads((h/'DISK_POLICY.json').read_text())}))
"""
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True);d=json.loads(r.stdout);timers=subprocess.check_output(['systemctl','is-active','baseline-retest32-server1-archive.timer','baseline-retest32-server1-halfhour-20261030.timer'],text=True).splitlines()
report={'recorded_at':time.time(),'alarm_due':'2026-11-01T03:19:06.992535+00:00','remote_checked_at':x['checked_at'],'remote_checked_utc':datetime.datetime.fromtimestamp(x['checked_at'],datetime.timezone.utc).isoformat(),'cached':x['cached'],'remaining':x['remaining'],'pending':x['pending_review'],'identity':x['process'],'guard':x['guard'],'phases':x['phases'],'disk':d,'free_GiB':d['free_bytes']/2**30,'timers':timers,'migration':json.loads((h/'MIGRATION_STATUS.json').read_text()),'actions':['Actual inspection PID2847187/start248630394,hostguard/device,disk and latest journal/checker verified','Nonblocking MLA_LOCALS_0319.txt confirms active FP16 f9d73684ed9ede64777a originalCPUref_program at final8/8sequence begin28672,end32768,h0=48,r0=1664;previoussample was2/8,progress established;no failed pending','Retain existing3600second fullcheck and original source/checker;remaining mixedBF16 enabled;no cached rerun','Mandatory incremental fullreceipt collection;continuousmigration and halfhour timersactive;disk50GiBfloor/50.5admission unchanged']}
name='DIAGNOSIS_20261101_0319.json';(h/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');p=h/'TAKEOVER.json';t=json.loads(p.read_text());t.update(updated_at=time.time(),last_inspection=str(h/name),identity=x['process'],session=97191,status='v73_running_final_environment_mla_retry',known_held=f"{x['cached']}cached,{x['remaining']}remaining,pending0;FP16 finalsequence;remaining MLA running;{report['free_GiB']:.2f}GiBfree",repair_audit=str(h/'MLA_KIMI_AUDIT_0249.json'),next_deployment_review=str(h/'QUANT_GEMM_ROUTE_REVIEW.json'));p.write_text(json.dumps(t,ensure_ascii=False,indent=2)+'\n')





with Path('.agent_memory/THREADS/baseline-inspector-20261023/MEMORY.md').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 f.write(f'\n\n## 2026-11-01 一号03:19巡检：MLA FP16最后序列推进\n\n实际检查DIAGNOSIS_20261101_0319.json；{x["cached"]}/1187、{x["remaining"]}未完成、pending0。v73 PID2847187/start248630394/session97191真实匹配；FP16 f9d73684ed9ede64777a子PID3861628，非阻塞采样MLA_LOCALS_0319.txt确认原CPU ref_program第8/8序列begin28672/end32768、h0=48/r0=1664，相比上轮2/8继续推进，未停止或重跑。最后混长BF16 d36b75a34eb33230af41已启用。维持3600秒及原完整checker，环境已齐，无失败待诊断。磁盘{report["free_GiB"]:.2f}GiB，持续迁移和半小时timer active；完整回执增量收回。原内核/compiler/schedule/checker不改，cached不重跑；全部1187完整验收才关timer。\n')
 f.flush();fcntl.flock(f,fcntl.LOCK_UN)
print({'cached':x['cached'],'remaining':x['remaining'],'free_GiB':report['free_GiB']})
