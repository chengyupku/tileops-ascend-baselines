from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent
s=(O.parent/'threehost_native9_registration/registration_parent.py').read_text()
s=s.replace('threehost_native9_registration',O.name).replace('NATIVE9_','BMM_VARIANT_').replace('_native9_','_bmm_variant_').replace('native9','Bmm backend variant').replace('Nine reviewed exact contracts','One reviewed exact Bmm contract')
a=s.index("FP8 = REPORT");b=s.index('\n\ndef checked',a)
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_backend_full1_r2'
s=s[:a]+f"ORIGINAL = {R!r}\nADAPTER = ORIGINAL + '/adapter.py'\nBINDING = ORIGINAL + '/binding.py'\nBASE = ORIGINAL + '/TileOPs/benchmarks/ascend_baselines.py'\nMATH = ORIGINAL + '/TileOPs/benchmarks/ascend_math_more.py'\nAPI = ORIGINAL + '/TileOPs/benchmarks/api.py'\nDATA_HASHES = {{}}  # sealed after contract generation\n"+s[b:]
s=s.replace('def admit(runtime=True):','def admit(data, runtime=True):').replace('    data=snapshot(runtime=runtime)','    if runtime:\n        for path,digest in metadata()[\'PINS.json\'][\'runtime\'].items(): checked(path,digest)')
s=s.replace("            data,guard,env=admit(runtime=state['inner'] is None)","            admitted,guard,env=admit(data, runtime=state['inner'] is None)\n            require_backend()")
s=s.replace("                mod=load(FP8 if key(frozen)[0]=='BmmFP8FwdOp' else BF16)\n                state['inner']=(mod.fp8_bmm if key(frozen)[0]=='BmmFP8FwdOp' else mod.make_candidate)(frozen)","                mod=load(ADAPTER)\n                state['inner']=mod.candidate(frozen,native_factory=load(BINDING).dense_fp32)")
s=s.replace('        return result\n','        return result\n')
s=s.replace("'Exact Bmm backend variant contract; lazy original complete ARM9.1 factory; private pinned closure.'","'Exact Bmm BF16 contract; requires explicitly approved mkldnn.enabled=False evaluation context; original V5 closure.'")
pos=s.index('\ndef make_candidate')
s=s[:pos]+'''\ndef require_backend():
    # Validation only: candidate never changes global backend or oracle.
    import torch
    if torch.backends.mkldnn.enabled is not False or torch.get_num_threads()!=4:
        raise RuntimeError('Approved evaluation requires public mkldnn.enabled=False and 4 CPU threads')
    if str(torch.__version__)!='2.10.0+cpu':
        raise RuntimeError('Tested CPU reference runtime required')

'''+s[pos:]
(O/'registration.py').write_text(s)
# Build real manifest case, preserving full existing Case serialization.
b=(O.parent/'threehost_native9_registration/build_contracts.py').read_text()
b=b.replace("O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration')","O=Path(__file__).resolve().parent")
b=b.replace("wanted={(r['op'],r['case_id']) for r in json.loads((O.parent/'threehost_native_completion9/CONTRACTS.json').read_text())}","wanted={('BmmFwdOp','square-4k-bfloat16')}")
b=b.replace('assert len(out)==9','assert len(out)==1');b=b[:b.index("if __name__=='__main__':")]
(O/'build_contracts.py').write_text(b)
