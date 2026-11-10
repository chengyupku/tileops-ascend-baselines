"""Original public fixtures/checker with explicit CPU reference, due measured NPU oracle drift."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import sinusoidal
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   (selected if 'sinusoidal' in item.name else removed).append(item)
  items[:]=selected;config.hook.pytest_deselected(items=removed)
  from workloads.elementwise import sinusoidal_reference
  from workloads.device import run_device
  def cpu_reference(seq_len,d_model,dtype,device=None):
   return sinusoidal_reference(seq_len,d_model,dtype,device='cpu').to(run_device())
  for item in selected:item.module.sinusoidal_reference=cpu_reference
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_special_elementwise.py'),'-q','--junitxml='+str(Path(__file__).with_name('sinusoidal-public-cpu-v2.xml'))],plugins=[Select()]))
