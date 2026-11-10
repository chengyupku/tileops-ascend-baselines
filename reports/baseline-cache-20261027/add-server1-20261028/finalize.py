from pathlib import Path
import json, time, datetime, fcntl, os
R=Path("reports/baseline-cache-20261027"); D=R/"add-server1-20261028"
now=time.time(); utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat()
snap={h:json.load(open(D/(h+"-final.json"))) for h in ("local","server1","server2")}
audit=json.load(open(D/"server1-AUDIT_pilots.json"))
report={"checked_at":now,"checked_at_utc":utc,"kind":"three_host_deployment_and_live_inspection","assignment_revision":6,"scope_count":6419,"changes":json.load(open(D/"ASSIGNMENT_CHANGES.json")),"hosts":{h:s["summary"] for h,s in snap.items()},"server1_pilot_audit":audit,"server1_launch":json.load(open(D/"server1-unlimited-launch.json")),"source_boundaries":"Only environment/deployment and CPU supervisor host selector changed. Original provider, kernel, compiler, schedule, checker and timing core preserved.","known_holds":["server1 DeepGEMM47: original deep_gemm package import requires tilelang.ascend namespace absent in installed remote runtime; no source substitution, failures preserved"],"next_inspection_utc":"2026-10-28T16:15:44.214085+00:00","inspection_interval_seconds":14400,"alarm_mode":"idle-only one-shot; original anchor retained","audit_scope":"Fresh hashes for server1 pilot cache prefix; local/server2 existing sources unchanged, native process and actual production checked this time"}
(D/"RESULT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
with (R/"INSPECTIONS.jsonl").open("a") as f:f.write(json.dumps(report,ensure_ascii=False)+"\n")
(R/"LAST_INSPECTION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
m=Path(".agent_memory/THREADS/baseline-inspector-20261023/STATE.json")
with (m.parent/".state.lock").open("a") as f:
 fcntl.flock(f,fcntl.LOCK_EX);d=json.load(open(m));t=d["baseline_cache_20261027"]
 t.update(updated_at=utc,status="three_host_cache_sweep_running",current_cached={h:s["summary"]["cached"] for h,s in snap.items()},current_ready={h:s["summary"]["ready"] for h,s in snap.items()},threehost_report=str(D/"RESULT.json"),active_followup="Three CPU supervisors run serial guarded pairs independently. Preserve source bindings and all holds; no valid-cache reruns. Every4h idle-only inspection on all3 hosts; next16:15UTC. Server1 private cache workspace differs from prior operator workspace; do not use old tree for campaign. DeepGEMM47 dependency hold remains, no unchanged retry.",last_threehost_inspection={"at":utc,"report":str(D/"RESULT.json"),"server1_audited":audit.get("summary",{}),"host_native_checks":{h:s["summary"]["identities"] for h,s in snap.items()}},next_actions=["Check all3 current PLAN-filtered ready/cache counts,1h/4h production, native PID/start/boot and disk","Keep server1 DeepGEMM47 dependency hold unless actual original runtime deployment changes; preserve exact local and server2 prior holds","Fresh cache audit each4h; do not rerun already valid caches; retry only changed conditions"])
 tmp=m.with_suffix(".tmp");tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");os.replace(tmp,m)
with (R/"RECURRING_ALARMS.lock").open("a") as f:
 fcntl.flock(f,fcntl.LOCK_EX);a=json.load(open(R/"RECURRING_ALARMS.json"))
 note="\n三卡启动已验证，详见add-server1-20261028/RESULT.json。一号47项DeepGEMM因原包必需tilelang.ascend缺失而精确环境暂挂；不能替换实现或无变化重试。SGL/vLLM试跑完整验证与fresh-hash缓存审计已通过。"
 if note not in a["message"]:a["message"]+=note
 a["equal_pending_at_publish_revision"]=5
 tmp=R/"RECURRING_ALARMS.tmp";tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2)+"\n");os.replace(tmp,R/"RECURRING_ALARMS.json")
print(json.dumps({"report":str(D/"RESULT.json"),"cached":{h:s["summary"]["cached"] for h,s in snap.items()},"ready":{h:s["summary"]["ready"] for h,s in snap.items()}},ensure_ascii=False))
