import pathlib,subprocess,os,json,time,hashlib
P=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_sdk_recovery36')
sdk=pathlib.Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2')
src=pathlib.Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages/tilelang/src')
# Compile unchanged original header as translation unit. No generated kernel survives previous temporary directory.
header=src/'tl_templates/ascend/debug.h'
cmd=[str(sdk/'tools/bisheng_compiler/bin/bisheng'),'-O2','-fPIC','-std=c++20','-mllvm','-cce-aicore-dcpreload-args=false','--npu-arch=dav-3510','-I'+str(src),'-I'+str(sdk/'asc/include'),'-I'+str(sdk/'asc'),'-I'+str(sdk/'include'),'--cce-aicore-only','-xcce',str(header),'-o',str(P/'header_probe_includes.o')]
env=os.environ.copy();env.update(ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk))
t=time.time();r=subprocess.run(cmd,env=env,cwd=P,capture_output=True,text=True,timeout=120)
(P/'header_probe_includes.log').write_text(r.stdout+r.stderr)
receipt=dict(command=cmd,source_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),returncode=r.returncode,seconds=time.time()-t,scope='unchanged original debug.h translation unit; not canonical kernel compilation')
(P/'header_probe_includes.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt));print(r.stdout+r.stderr)
