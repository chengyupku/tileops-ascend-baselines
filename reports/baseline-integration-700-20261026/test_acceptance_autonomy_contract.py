"""Synthetic scheduling fixtures only: no real pair evidence, workers or NPU access."""
import importlib.util,json,pathlib,sqlite3,tempfile,time,unittest,datetime
FILE=pathlib.Path(__file__).with_name('refill_fleet.py')
class AcceptanceRecovery(unittest.TestCase):
 def setUp(self):
  spec=importlib.util.spec_from_file_location('fixture_fleet',FILE);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=pathlib.Path(self.tmp.name);self.m.D=self.d
  for n in ['g','acceptance_root']:(self.d/n).mkdir()
  self.state={'enabled':True,'phase':'running','coordinator_groups':['g'],'deadline_utc':datetime.datetime.fromtimestamp(time.time()+3600,datetime.timezone.utc).isoformat()}
  self.m.put(self.d/'STATE.json',self.state);self.m.put(self.d/'ACCEPTED.json',{'rows':[]})
  self.m.put(self.d/'ROOT_CONTROL_QUEUE.json',{'tasks':[{'name':'acceptance_root','job_id':'old','ready':False}]})
  self.item={'key':['synthetic','operator','case'],'status':'verified_ready_for_parent_serial_publication','ready_to_publish':True,'evidence_version':1}
  self.m.put(self.d/'g/PUBLISH_QUEUE.json',{'entries':[self.item]})
  self.db=sqlite3.connect(':memory:');self.addCleanup(self.db.close);self.db.execute('create table jobs(id text,status text,data text)')
  self.db.execute('insert into jobs values(?,?,?)',('old','completed',json.dumps({'finished_at':time.time()-600})))
  self.db.execute('insert into jobs values(?,?,?)',('new','failed',json.dumps({'finished_at':time.time()-600})))
  self.m.live=lambda *a:False
 def call(self):
  self.m.wake_acceptance_for_new_publish(self.state,self.db,{'rejected':[]})
 def entry(self):return self.m.read(self.d/'ROOT_CONTROL_QUEUE.json')['tasks'][0]
 def putentry(self,e):self.m.put(self.d/'ROOT_CONTROL_QUEUE.json',{'tasks':[e]})
 def prepare_pending(self):
  self.call();e=self.entry();self.assertTrue(e['ready']);e['ready']=False;e['job_id']='new'
  # Model a receiver that exited long enough ago to pass a bounded retry backoff.
  for k in list(e):
   if k.endswith('_at') and isinstance(e[k],(float,int)):e[k]=time.time()-600
  for pending in e.get('publish_wake_pending',[]):
   pending['created_at']=datetime.datetime.fromtimestamp(time.time()-600,datetime.timezone.utc).isoformat()
  self.putentry(e);return e
 def test_failed_receiver_requeues_pending(self):
  self.prepare_pending();self.call();self.assertTrue(self.entry().get('ready'),'pending dead receiver must be retried')
 def test_live_receiver_not_duplicated(self):
  self.prepare_pending();self.db.execute("update jobs set status='running' where id='new'");self.call();self.assertFalse(self.entry().get('ready'))
 def test_rejected_disposition_not_replayed_and_new_evidence_wakes(self):
  e=self.prepare_pending();sig=e['publish_wake_pending'][0]['signature']
  self.m.put(self.d/'acceptance_root/WAKE_DISPOSITION.json',{'processed_signatures':[sig]})
  self.call();self.assertFalse(self.entry().get('ready'),'handled unchanged rejected item must not wake again')
  self.assertFalse(self.entry().get('publish_wake_pending'))
  self.item['evidence_version']=2;self.m.put(self.d/'g/PUBLISH_QUEUE.json',{'entries':[self.item]})
  self.call();self.assertTrue(self.entry().get('ready'),'genuinely new evidence should wake')
 def test_paused_no_wake(self):
  self.state['enabled']=False;self.m.put(self.d/'STATE.json',self.state);self.call();self.assertFalse(self.entry().get('ready'))
 def test_explicit_stop_no_wake(self):
  self.state['dispatch_stop_at']=datetime.datetime.fromtimestamp(time.time()-1,datetime.timezone.utc).isoformat();self.m.put(self.d/'STATE.json',self.state);self.call();self.assertFalse(self.entry().get('ready'))
 def test_missed_delivery_sla_does_not_abandon_work(self):
  self.state['deadline_utc']=datetime.datetime.fromtimestamp(time.time()-1,datetime.timezone.utc).isoformat();self.m.put(self.d/'STATE.json',self.state);self.call();self.assertTrue(self.entry().get('ready'))
if __name__=='__main__':unittest.main()
