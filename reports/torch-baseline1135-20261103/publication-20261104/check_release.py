import json,pathlib,urllib.request,subprocess
P=pathlib.Path(__file__).parent
headers={'Authorization':'Bearer '+pathlib.Path('/tmp/tileops-remote-ci-auth').read_text().strip(),'User-Agent':'baseline-publication'}
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=25) as r:return json.load(r)
reports=[]
for repo,local in [('TileOPs.github.io','/tmp/torch1135-site'),('TileOPs','/tmp/torch1135-ci')]:
 sha=subprocess.check_output(['git','-C',local,'rev-parse','HEAD'],text=True).strip()
 data=get('https://api.github.com/repos/yyttt6/'+repo+'/actions/runs?head_sha='+sha+'&per_page=15')
 for r in data['workflow_runs']:
  row={k:r[k] for k in ['id','name','status','conclusion','html_url','head_sha']};row['repo']=repo
  jobs=get('https://api.github.com/repos/yyttt6/'+repo+'/actions/runs/'+str(r['id'])+'/jobs')
  row['jobs']=[{k:j[k] for k in ['name','status','conclusion']} for j in jobs['jobs']];reports.append(row)
  print(repo,row['id'],row['name'],row['status'],row['conclusion'],row['jobs'],flush=True)
(P/'ACTIONS.json').write_text(json.dumps(reports,indent=2))
