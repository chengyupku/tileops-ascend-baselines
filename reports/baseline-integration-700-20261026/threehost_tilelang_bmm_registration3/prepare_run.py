"""Create a fresh private CPU test run; never starts a guard or imports device libraries."""
import pathlib,tempfile,os,json
O=pathlib.Path(__file__).resolve().parent
run=pathlib.Path(tempfile.mkdtemp(prefix='run-',dir=O/'runs'))
run.joinpath('RUN_EXCLUSIVE').write_text('fresh private run; CPU lifecycle tests only\n')
run.joinpath('cleanup.marker').write_text('cleanup-on-exit required\n')
env={'TILELANG_BMM_RUN_ROOT':str(run),'TILELANG_BMM_GUARD_OWNER':'baseline700-20261026/threehost_tilelang_bmm_registration3'}
(run/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
print(run)
