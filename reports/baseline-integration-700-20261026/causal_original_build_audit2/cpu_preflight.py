#!/usr/bin/env python3
# Read-only CPU preflight: stdlib only; never import torch, ACL, or vendor Python.
import pathlib,sys,json,hashlib
src=pathlib.Path(sys.argv[1]); sdk=pathlib.Path(sys.argv[2])
required_source=['build.sh','cmake/func.cmake','cmake/scripts/util/ascendc_ops_config.py','cmake/scripts/util/ascendc_bin_param_build.py','mamba/causal_conv1d/op_api/aclnn_causal_conv1d_update.cpp','mamba/causal_conv1d/op_host/causal_conv1d_def.cpp','mamba/causal_conv1d/op_kernel/causal_conv1d_apt.cpp','third_party/lib_cache/protoc/protoc','third_party/opbase','third_party/ops-tensor','third_party/catlass']
required_sdk=['bin/asc_opc','tools/opbuild/op_build','tools/bisheng_compiler/bin/bisheng','python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py','aarch64-linux/asc/include/kernel_operator.h','share/info/asc-devkit/version.info','lib64/libnnopbase.so']
checks=[{'path':str(root/rel),'exists':(root/rel).exists()} for root,rels in [(src,required_source),(sdk,required_sdk)] for rel in rels]
cache=[src/'third_party/cann-cmake/function/prepare.cmake',src/'build/_deps/cann-cmake-src/function/prepare.cmake']
checks.append({'path':'local cann-cmake source (one of '+', '.join(map(str,cache))+')','exists':any(f.is_file() for f in cache)})
api=src/required_source[4]
print(json.dumps({'checks':checks,'api_sha256':hashlib.sha256(api.read_bytes()).hexdigest() if api.is_file() else None,'scope':'presence/hash only, not compiler/runtime validation'},indent=2))
sys.exit(0 if all(c['exists'] for c in checks) else 1)
