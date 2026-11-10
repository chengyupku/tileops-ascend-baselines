import json,time,sys,collections
from pathlib import Path
sys.path.insert(0,"tools")
from run_baseline_cache import assigned_history,completed_ids,quarantine_key
h=sys.argv[1];R=Path("reports/baseline-cache-20261027");H=R/h; D=R/"inspection-20261029T001544"
r=lambda p:json.loads(p.read_text())
tasks=[t for t in r(R/"PLAN.json")["pairs"] if t["host"]==h];own={t["id"] for t in tasks}
rows=[json.loads(s) for s in (H/"journal.jsonl").read_text().splitlines() if s.strip()]; caches=[json.loads(s) for s in (H/"CACHE.jsonl").read_text().splitlines() if s.strip()]
latest,cached=assigned_history(tasks,rows,caches);q=r(H/"QUARANTINE.json");done=completed_ids(latest,r(H/"RETRY.json"),q,cached)
ready=[t for t in tasks if t["id"] not in done and not quarantine_key(t,q)]; groups={}
for x in rows:
 if x["id"] not in own or x.get("at",0)<time.time()-14400 or not x.get("error"):continue
 e=x["error"]; e=e.split("\n")[0] if not ("tl.cumsum" in e or "ConvertLinalgRToBinary" in e) else ("triton_cumsum_compile" if "tl.cumsum" in e else "triton_memory_plan_compile")
 if "Tensor-likes are not close" in x["error"]: e="numerical_mismatch"
 g=groups.setdefault(e,{"count":0,"wall_seconds":0,"ids":[],"operators":collections.Counter(),"examples":[]})
 g["count"]+=1;g["wall_seconds"]+=x.get("wall_seconds",0) or 0;g["ids"].append(x["id"]);g["operators"]["::".join(x["key"][:2])]+=1
 if len(g["examples"])<2:g["examples"].append(x)
ordered=sorted(groups.items(),key=lambda z:z[1]["wall_seconds"],reverse=True)
result={"at":time.time(),"host":h,"groups":dict(ordered),"ready":ready}
(D/(h+"-analysis.json")).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"host":h,"groups":[{"error":k,"count":g["count"],"wall_seconds":round(g["wall_seconds"]),"operators":g["operators"]} for k,g in ordered],"ready":len(ready)},ensure_ascii=False))
