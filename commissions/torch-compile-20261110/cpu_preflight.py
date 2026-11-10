"""CPU-only inventory: never imports torch/torch_npu/acl or probes the device.
Usage: python cpu_preflight.py --output ENVIRONMENT.cpu.json
Run under the intended venv; review/redact paths before public submission.
"""
import argparse, hashlib, importlib.metadata as metadata, json, os, pathlib, platform, shutil, subprocess

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20), b''):h.update(chunk)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args()
    out=pathlib.Path(args.output)
    if out.exists():raise SystemExit('Refusing to overwrite existing environment receipt')
    d={'device_imports':False,'hardware_verified':False,'python':platform.python_version(),'machine':platform.machine(),'packages':{},'files':{},'resource_limits':{},'missing':[]}
    for name in ('torch','torch-npu','tilelang','numpy','torchair'):
        try:
            dist=metadata.distribution(name);d['packages'][name]={'version':dist.version,'root':str(dist.locate_file(''))}
            if name=='torch-npu':
                for relative in dist.files or ():
                    s=str(relative)
                    if s.endswith(('libtorch_npu.so','libnpu_profiler.so')):
                        p=pathlib.Path(dist.locate_file(relative));d['files'][str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}
        except metadata.PackageNotFoundError:d['missing'].append(name)
    cann=os.environ.get('ASCEND_HOME_PATH')
    candidates=[pathlib.Path(cann)/'version.cfg'] if cann else []
    compiler=shutil.which('bisheng')
    if compiler:
        p=pathlib.Path(compiler).resolve();candidates.append(p)
        r=subprocess.run([str(p),'--version'],capture_output=True,text=True,timeout=20)
        d['compiler_version']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    else:d['missing'].append('bisheng')
    for p in candidates:
        if p.is_file():d['files'][str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}
        else:d['missing'].append(str(p))
    for name in ('cpu.max','cpuset.cpus.effective','memory.max','memory.current','cpu/cpu.cfs_quota_us','cpu/cpu.cfs_period_us','memory/memory.limit_in_bytes','memory/memory.usage_in_bytes'):
        p=pathlib.Path('/sys/fs/cgroup')/name
        if p.is_file():d['resource_limits'][name]=p.read_text().strip()
    d['cpu_affinity_count']=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    d['disk_free_bytes']=shutil.disk_usage(out.parent).free
    d['environment']={k:os.environ.get(k) for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','ASCEND_OPP_PATH','TORCHINDUCTOR_NPU_BACKEND','TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_WORKER_START')}
    with out.open('x') as f:json.dump(d,f,indent=2)
    print('CPU receipt written; device guard/probe, wheel provenance and actual loaded DSO evidence still required.')
if __name__=='__main__':main()
