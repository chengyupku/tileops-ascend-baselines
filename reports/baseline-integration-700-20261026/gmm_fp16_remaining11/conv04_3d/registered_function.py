
def verified_conv04_3d(case):
    assert case.id == 'unet-aspp-float16'
    path=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11/conv04_3d/remaining_binding.py')
    spec=importlib.util.spec_from_file_location('_gmm_fp16_remaining11_conv04_3d',path)
    m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(m)
    finally:sys.path.remove(str(path.parent))
    return m.remaining(case)
