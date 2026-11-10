import json, pathlib, sys, time
h=sys.argv[1];R=pathlib.Path('reports/baseline-cache-20261027');H=R/h;D=R/'inspection-20261029T001544'
token='inspection-20261029T001544/rebalance-pending-only\n'
p=H/'STOP'
if p.exists():
    assert p.read_text()==token, 'Existing control STOP belongs to someone else'
else:
    with p.open('x') as f:f.write(token)
l=json.loads((H/'LIVE.json').read_text())
(D/(h+'-BOUNDARY_REQUEST.json')).write_text(json.dumps({'at':time.time(),'token':token,'live':{k:l.get(k) for k in ['at','phase','identity','key','out']}},indent=2)+'\n')
print(json.dumps({'host':h,'phase':l['phase'],'key':l.get('key'),'stop_at_normal_boundary':True}))
