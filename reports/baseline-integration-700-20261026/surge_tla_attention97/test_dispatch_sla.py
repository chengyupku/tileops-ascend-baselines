import ast,datetime,time,unittest,pathlib
# Execute only pure guard definitions, never the production spawn body.
source=pathlib.Path(__file__).with_name('replace_worker.py').read_text()
tree=ast.parse(source)
ns={'datetime':datetime,'time':time}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('dispatch_budget','ensure_released')],type_ignores=[]),'<guards>','exec'),ns)
class Contract(unittest.TestCase):
 def setUp(self):
  self.state={'enabled':True,'deadline_utc':'2026-10-27T05:31:00Z'}
  self.group={'deadline_utc':'2026-10-27T05:31:00Z'}
  self.now=datetime.datetime.fromisoformat('2026-10-27T06:31:00+00:00').timestamp()
 def test_missed_sla_keeps_full_budget(self):
  self.assertEqual(ns['dispatch_budget'](self.state,self.group,self.now),3600)
 def test_explicit_stop(self):
  for container in (self.state,self.group):
   container['dispatch_stop_at']='2026-10-27T05:31:00Z'
   with self.assertRaises(SystemExit): ns['dispatch_budget'](self.state,self.group,self.now)
   del container['dispatch_stop_at']
 def test_paused_exhausted_hold_unsupported(self):
  for phase in ('paused','stopped','completed','dispatch_exhausted','HOLD','api_unsupported'):
   for container in (self.state,self.group):
    container['phase']=phase
    with self.assertRaises(SystemExit): ns['dispatch_budget'](self.state,self.group,self.now)
    del container['phase']
 def test_disabled(self):
  self.group['enabled']=False
  with self.assertRaises(SystemExit):ns['dispatch_budget'](self.state,self.group,self.now)
 def test_independent_budget(self):
  self.group['worker_timeout_seconds']=1800
  self.assertEqual(ns['dispatch_budget'](self.state,self.group,self.now),1800)
 def test_future_explicit_stop_caps_budget(self):
  self.state['dispatch_stop_at']='2026-10-27T06:33:00Z'
  self.assertEqual(ns['dispatch_budget'](self.state,self.group,self.now),120)
 def test_clearance_tied_to_previous_job(self):
  c=dict(previous_job_id='old',legal_unfinished=True,ownership_released=True,device_drained=True)
  ns['ensure_released'](c,'old')
  with self.assertRaises(SystemExit): ns['ensure_released'](c,'new')
  for k in ('legal_unfinished','ownership_released','device_drained'):
   bad=dict(c);bad[k]=False
   with self.assertRaises(SystemExit):ns['ensure_released'](bad,'old')
if __name__=='__main__': unittest.main()
