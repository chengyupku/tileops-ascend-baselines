import pathlib, json, subprocess, platform, sys, datetime, shutil
out={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"arch":platform.machine(),"python":sys.version,"executable":sys.executable,"paths":{}}
for name in ["/etc/hccl_rootinfo.json","/usr/local/Ascend/driver/topo/950","/opt/tileops-all-pairs/bin/allpairs-python","/opt/tileops-ci/runtime"]:
 p=pathlib.Path(name);out["paths"][name]={"exists":p.exists(),"is_dir":p.is_dir(),"is_file":p.is_file()}
 if p.is_dir() and name.endswith(".json"):out["paths"][name]["entries"]=[x.name for x in p.iterdir()]
out["mounts"]=[x for x in pathlib.Path("/proc/mounts").read_text().splitlines() if "hccl_rootinfo" in x]
for cmd in [["npu-guard","status","--json"]]:
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=25);out["guard"]={"command":cmd,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
 except Exception as e:out["guard"]={"error":str(e)}
print(json.dumps(out,indent=2))
