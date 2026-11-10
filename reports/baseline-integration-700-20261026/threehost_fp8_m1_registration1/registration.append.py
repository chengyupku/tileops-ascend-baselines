# Publication fragment for the parent; NOT applied by this worker.
# Deploy the complete directory (module + frozen/ + pin/contract JSON files) first.
import importlib.util as _fp8_util
_fp8_path = '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_fp8_m1_registration1/integration700_threehost_fp8_m1_registration1.py'
_fp8_spec = _fp8_util.spec_from_file_location('_integration700_fp8_m1', _fp8_path)
_fp8_module = _fp8_util.module_from_spec(_fp8_spec)
_fp8_spec.loader.exec_module(_fp8_module)
PROVIDERS['GemmFP8FwdOp'] = _fp8_module.wrap(PROVIDERS.get('GemmFP8FwdOp'))
