import pathlib,json,subprocess,base64
O=pathlib.Path(__file__).resolve().parent;ns={};exec((O/'integration700_gdn_recurrent_registration1.py').read_text(),ns)
files={n:base64.b64encode((O/n).read_bytes()).decode() for n in ['integration700_gdn_recurrent_registration1.py','cpu_proof.py']}
script='import pathlib,json,hashlib,base64,subprocess\nO=pathlib.Path('+repr(str(O))+');O.mkdir(exist_ok=True)\nfiles='+repr(files)+'\nfor n,b in files.items():(O/n).write_bytes(base64.b64decode(b))\npins='+repr(dict(ns['RUNTIME_PINS']))+'\nrows=[]\nfor s,h in pins.items():\n p=pathlib.Path(s);actual=hashlib.sha256(p.read_bytes()).hexdigest();rows.append({"path":s,"expected":h,"actual":actual,"match":actual==h})\nr=subprocess.run(["python3","-B",str(O/"cpu_proof.py")],capture_output=True,text=True)\nprint(json.dumps({"rows":rows,"cpu_returncode":r.returncode,"cpu_stdout":r.stdout,"cpu_stderr":r.stderr}))\n'
r=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','python3 -'],input=script,text=True,capture_output=True)
assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(O/'REMOTE_PIN_CHECK.json').write_text(json.dumps(x,indent=2)+'\n'); print('checked',len(x['rows']),'bad',sum(not v['match'] for v in x['rows']),'remote CPU',x['cpu_returncode'],x['cpu_stdout'],x['cpu_stderr'][:300])
