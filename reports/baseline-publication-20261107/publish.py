"""Publish a reviewed local commit using Git objects and a non-force ref update."""
import sys,json,subprocess,urllib.request
from pathlib import Path
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent.parent/'.tmp/tileops-remote-ci'));from github_api import api
kind=sys.argv[1];S=P/kind;repo='repos/yyttt6/'+('TileOPs.github.io' if kind=='site' else 'TileOPs')+'/';branch='main' if kind=='site' else 'ascend950'
def git(*args):return subprocess.check_output(['git','-C',str(S),*args],text=True).strip()
assert not git('status','--porcelain'), 'Uncommitted files'
parent=git('rev-parse','HEAD^');head=git('rev-parse','HEAD');remote=api(repo+'git/ref/heads/'+branch)['object']['sha']
assert remote==parent,(remote,parent)
entries=[]
for f in git('diff','--name-only','HEAD^','HEAD').splitlines():
 blob=api(repo+'git/blobs',method='POST',data={'content':(S/f).read_text(),'encoding':'utf-8'})
 assert blob['sha']==git('rev-parse','HEAD:'+f)
 entries.append({'path':f,'mode':'100644','type':'blob','sha':blob['sha']});print('blob',f,flush=True)
t=api(repo+'git/trees',method='POST',data={'base_tree':api(repo+'git/commits/'+parent)['tree']['sha'],'tree':entries});assert t['sha']==git('rev-parse','HEAD^{tree}')
a=git('show','-s','--format=%an%n%ae%n%aI','HEAD').splitlines();c=git('show','-s','--format=%cn%n%ce%n%cI','HEAD').splitlines()
commit=api(repo+'git/commits',method='POST',data={'message':git('show','-s','--format=%B','HEAD'),'tree':t['sha'],'parents':[parent],'author':dict(zip(['name','email','date'],a)),'committer':dict(zip(['name','email','date'],c))})
assert api(repo+'git/ref/heads/'+branch)['object']['sha']==parent
api(repo+'git/refs/heads/'+branch,method='PATCH',data={'sha':commit['sha'],'force':False})
assert api(repo+'git/ref/heads/'+branch)['object']['sha']==commit['sha']
(P/(kind+'-published.json')).write_text(json.dumps({'commit':commit['sha'],'local_commit':head,'tree':t['sha'],'parent':parent},indent=2));print('PUBLISHED',commit['sha'],flush=True)
