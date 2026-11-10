"""Private original ops-nn Foreach package; calculation source remains unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
BASE=ROOT/'.cache/logroots4_20261028/ops-nn';SOURCE=BASE/'source'
if __name__ == "__main__":
    tool=SOURCE/'third_party/lib_cache/protoc/protoc'
    reference=ROOT/'.cache/arithmetic9_20261027/ops-math-add/source/third_party/lib_cache/protoc/protoc'
    if tool.exists() and not (BASE/'protoc-original-x86').exists():shutil.copy2(tool,BASE/'protoc-original-x86')
    shutil.copy2(reference,tool)
    start=time.monotonic()
    env=ROOT/'.cache/all-pairs/remote_ops_math_continuation/build_env_pinned.sh'
    command=['bash','-c','source "$1"\nexport PYTHONPATH="/opt/tileops-all-pairs/workspace/.cache/arithmetic9_20261027/python-deps:$PYTHONPATH"\ncd "$2"\nexport TMPDIR="$3/tmp" XDG_CACHE_HOME="$3/xdg" ASCEND_CACHE_PATH="$3/runtime-cache" ASCEND_PROCESS_LOG_PATH="$3/logs" ASCEND_WORK_PATH="$3"\nmkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"\nexec bash ./build.sh --pkg --jit --soc=ascend950 --ops=foreach_log,foreach_log1p,foreach_sqrt,foreach_pow_scalar --vendor_name=logroots4 -j8 --build-type=Release --ccache=off','arithmetic9-build',str(env),str(SOURCE),str(BASE)]
    result=subprocess.run(command);record={'argv':command,'build_rc':result.returncode,'elapsed_seconds':time.monotonic()-start}
    if result.returncode==0:
        installed=subprocess.run(['cmake','--install',str(SOURCE/'build'),'--prefix',str(BASE/'install')]);record['install_rc']=installed.returncode
        if installed.returncode==0:
            vendor=next((BASE/'install/packages/vendors').iterdir());target=BASE/'graph-runtime/vendors'/vendor.name
            shutil.copytree(vendor,target,ignore=shutil.ignore_patterns('libcust_opapi.so'),dirs_exist_ok=True)
            record['vendor']=str(target)
    (BASE/'build_receipt_v3.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    raise SystemExit(record.get('install_rc',result.returncode))
