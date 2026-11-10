import sys,json,subprocess,urllib.request
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
from pathlib import Path
sys.path.insert(0,'/root/dyq/workspace/.tmp/tileops-remote-ci');from github_api import api
S=Path('/tmp/tileops-ci-baseline2208-20261106');P=Path('/root/dyq/workspace/reports/baseline-publication-20261106');repo='repos/yyttt6/TileOPs/'
def git(*args):return subprocess.check_output(['git','-C',str(S),*args],text=True).strip()
parent=git('rev-parse','HEAD^');head=git('rev-parse','HEAD');remote=api(repo+'git/ref/heads/ascend950')['object']['sha']
if remote==head:print('Already published',head);sys.exit()
assert remote==parent,remote
files=git('diff','--name-only','HEAD^','HEAD').splitlines();tree=[]
for f in files:
 blob=api(repo+'git/blobs',method='POST',data={'content':(S/f).read_text(),'encoding':'utf-8'});assert blob['sha']==git('rev-parse','HEAD:'+f)
 tree.append({'path':f,'mode':'100644','type':'blob','sha':blob['sha']});print('blob',f,flush=True)
t=api(repo+'git/trees',method='POST',data={'base_tree':api(repo+'git/commits/'+parent)['tree']['sha'],'tree':tree});assert t['sha']==git('rev-parse','HEAD^{tree}')
a=git('show','-s','--format=%an%n%ae%n%aI','HEAD').splitlines();c=git('show','-s','--format=%cn%n%ce%n%cI','HEAD').splitlines()
commit=api(repo+'git/commits',method='POST',data={'message':git('show','-s','--format=%B','HEAD'),'tree':t['sha'],'parents':[parent],'author':dict(zip(['name','email','date'],a)),'committer':dict(zip(['name','email','date'],c))})
assert api(repo+'git/ref/heads/ascend950')['object']['sha']==parent
api(repo+'git/refs/heads/ascend950',method='PATCH',data={'sha':commit['sha'],'force':False})
assert api(repo+'git/ref/heads/ascend950')['object']['sha']==commit['sha']
P.joinpath('CI_API_PUBLISH.json').write_text(json.dumps({'commit':commit['sha'],'local_commit':head,'same_tree':t['sha'],'parent':parent},indent=2));print('PUBLISHED',commit['sha'],flush=True)
