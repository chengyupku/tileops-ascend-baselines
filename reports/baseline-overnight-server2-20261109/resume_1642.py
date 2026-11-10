import pathlib,subprocess
p=pathlib.Path(__file__).resolve().parent
subprocess.run(['python3',str(p/'run_one_queue.py'),'130-sampling-v3.json','sampling-search-probe-r2','probe'])
subprocess.run(['python3',str(p/'batch_general_queue.py')],check=True)
