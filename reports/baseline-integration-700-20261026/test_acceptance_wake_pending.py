"""CPU-only fixture for durable acceptance wake/disposition semantics."""
import json, pathlib, importlib.util, sqlite3, tempfile

def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x))
def main():
    spec=importlib.util.spec_from_file_location('refill_fixture', pathlib.Path(__file__).with_name('refill_fleet.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    root=pathlib.Path(tempfile.mkdtemp(prefix='acceptance-wake-'))
    mod.D=root; mod.W=root.parent
    (root/'acceptance_root').mkdir(); (root/'group').mkdir()
    (root/'STATE.json').write_text(json.dumps({'enabled':True,'phase':'running','deadline_utc':'2099-01-01T00:00:00+00:00'})); put(root/'ACCEPTED.json',{'rows':[]})
    checker=root/'group'/'receipt'; checker.write_text('v1')
    put(root/'group'/'PUBLISH_QUEUE.json',{'entries':[{'key':['b','o','c'],'ready_to_publish':True,'checker':str(checker)}]})
    queue={'tasks':[{'name':'acceptance_root','enabled':True,'phase':'running','ready':False,'job_id':'j1','publish_wake_delivered_signatures':[]}]}
    put(root/'ROOT_CONTROL_QUEUE.json',queue)
    db=sqlite3.connect(':memory:'); db.execute('create table jobs(id text,status text,data text)'); db.execute('insert into jobs values(?,?,?)',('j1','completed','{}')); db.commit()
    state={'coordinator_groups':['group']}; rec={'rejected':[]}
    mod.wake_acceptance_for_new_publish(state,db,rec)
    e=json.loads((root/'ROOT_CONTROL_QUEUE.json').read_text())['tasks'][0]; assert e['ready'] and e['publish_wake_pending']
    pending=e['publish_wake_pending'][0]; sig=pending['signature']; digest=pending['candidate_digests'][0]
    # Receiver dies before reading: pending stays and is retried with backoff.
    e['ready']=False; e['job_id']='j2'; put(root/'ROOT_CONTROL_QUEUE.json',{'tasks':[e]}); db.execute('insert into jobs values(?,?,?)',('j2','completed','{}')); db.commit(); rec={'rejected':[]}
    mod.wake_acceptance_for_new_publish(state,db,rec); e=json.loads((root/'ROOT_CONTROL_QUEUE.json').read_text())['tasks'][0]; assert e['ready'] and e['publish_wake_pending'] and rec.get('wake_retries')
    # Actual disposition handles candidate even when rejected; no duplicate wake.
    e['ready']=False; e['job_id']='j3'; put(root/'ROOT_CONTROL_QUEUE.json',{'tasks':[e]}); db.execute('insert into jobs values(?,?,?)',('j3','completed','{}')); db.commit()
    put(root/'acceptance_root'/'WAKE_DISPOSITION.json',{'processed_signature':sig,'processed_candidate_digests':[digest]})
    mod.wake_acceptance_for_new_publish(state,db,{'rejected':[]}); e=json.loads((root/'ROOT_CONTROL_QUEUE.json').read_text())['tasks'][0]; assert not e['publish_wake_pending'] and digest in e['publish_wake_handled_digests'] and not e['ready']
    # Same evidence remains quiet; changed receipt creates a new wake.
    mod.wake_acceptance_for_new_publish(state,db,{'rejected':[]}); e=json.loads((root/'ROOT_CONTROL_QUEUE.json').read_text())['tasks'][0]; assert not e['ready']
    checker.write_text('v2'); mod.wake_acceptance_for_new_publish(state,db,{'rejected':[]}); e=json.loads((root/'ROOT_CONTROL_QUEUE.json').read_text())['tasks'][0]; assert e['ready'] and e['publish_wake_pending']
    e['phase']='paused'; put(root/'ROOT_CONTROL_QUEUE.json',{'tasks':[e]}); mod.wake_acceptance_for_new_publish(state,db,{'rejected':[]}); assert json.loads((root/'ROOT_CONTROL_QUEUE.json').read_text())['tasks'][0]['ready']
    print('PASS acceptance pending/dead-receiver retry/handled-digest/new-evidence/paused fixture')
if __name__=='__main__': main()
