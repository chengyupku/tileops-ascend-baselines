import json,pathlib,math,collections
P=pathlib.Path(__file__).resolve().parent
rows=json.load(open(P/'QUEUE.json'))['tasks'];done={json.load(open(P/'representatives'/f'{i}.json'))['key'][1] for i in range(6)}|{'RoPEFwdOp'}
groups=collections.defaultdict(list)
for r in rows:
 if r['task']['key'][1] not in done:groups[r['task']['key'][1]].append(r)
def size(r):
 return sum(math.prod(t[0])*({'float64':8,'int64':8,'float32':4,'int32':4}.get(t[1],2)) for t in r['task']['case']['tensors'].values())
loads=[0,0];assign=[[],[]];report=[]
for i,(op,rs) in enumerate(sorted(groups.items()),6):
 r=min(rs,key=size);t=dict(r['task']);t.update({k:r[k] for k in ['reasons','old_keys','source_contract']});(P/'representatives'/f'{i}.json').write_text(json.dumps(t,indent=2))
 host=min(range(2),key=lambda h:loads[h]);loads[host]+=size(r);assign[host].append(i);report.append({'index':i,'host':f'server{host+1}','key':t['key'],'tensor_bytes':size(r)})
(P/'EXPANDED_REPRESENTATIVES.json').write_text(json.dumps({'rows':report,'groups':assign,'estimated_tensor_bytes':loads},indent=2));print(assign)
