"""CPU-only serial payload preparation; preserve original seed/oracle/checker."""
import sys,os,json,pathlib,runpy,gc,time
P=pathlib.Path(__file__).resolve().parent
rows=json.loads((P/'TASKS.json').read_text())
order=[0,5,12,20,26,33,40,48,56,64]+[i for i in range(len(rows)) if i not in [0,5,12,20,26,33,40,48,56,64]]
for i in order:
 if (P/'runs'/str(i)/'prepare.json').exists():continue
 sys.argv=[str(P/'runner.py'),'prepare',str(i)]
 try:runpy.run_path(str(P/'runner.py'),run_name='__main__')
 except SystemExit as e:
  if e.code:raise
 gc.collect()
print('ALL_CPU_PREPARED',flush=True)
