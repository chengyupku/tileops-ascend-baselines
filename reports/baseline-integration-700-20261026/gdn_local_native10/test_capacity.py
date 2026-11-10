import unittest,tempfile,json,sys
from pathlib import Path
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'chunk_package'))
from resource_gate import capacity,require_capacity
class CapacityTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(dir=W/'tmp');self.base=Path(self.t.name);self.proc=self.base/'proc';(self.proc/'self').mkdir(parents=True);self.root=self.base/'cg';self.leaf=self.root/'parent/leaf';self.leaf.mkdir(parents=True)
  (self.proc/'meminfo').write_text('MemAvailable: 1000000 kB\n');(self.proc/'self/cgroup').write_text('0::/parent/leaf\n');(self.proc/'self/mountinfo').write_text(f'1 0 0:1 / {self.root} rw - cgroup2 cgroup rw\n')
  for p in (self.leaf,self.leaf.parent):
   for n,v in {'memory.max':'max','memory.high':'max','memory.current':'100','memory.stat':'inactive_file 40\nactive_file 80\nshmem 10\nfile_dirty 2\nfile_writeback 3\nunevictable 5\n'}.items():(p/n).write_text(v)
 def tearDown(self):self.t.cleanup()
 def test_finite_ancestor(self):
  (self.leaf.parent/'memory.max').write_text('200');self.assertEqual(capacity(self.proc)['effective_available_host_bytes'],165)
 def test_high(self):
  (self.leaf.parent/'memory.high').write_text('120');self.assertEqual(capacity(self.proc)['effective_available_host_bytes'],85)
 def test_missing_field_unlimited(self):
  (self.leaf/'memory.stat').write_text('inactive_file 4\n')
  with self.assertRaises(KeyError):capacity(self.proc)
 def test_invisible(self):
  (self.proc/'self/cgroup').write_text('0::/other/leaf\n')
  with self.assertRaises(RuntimeError):capacity(self.proc)
 def test_traversal(self):
  (self.proc/'self/cgroup').write_text('0::/parent/../leaf\n')
  with self.assertRaises(RuntimeError):capacity(self.proc)
 def test_unknown_mountroot(self):
  (self.proc/'self/mountinfo').write_text(f'1 0 0:1 /hidden {self.root} rw - cgroup2 cgroup rw\n')
  with self.assertRaises(RuntimeError):capacity(self.proc)
 def test_root(self):self.assertEqual(capacity(self.proc)['effective_available_host_bytes'],1024000000)
 def test_insufficient(self):
  from unittest.mock import patch
  here=self.base/'package';here.mkdir();run=self.base/'case';run.mkdir();(here/'RESOURCE_ESTIMATES.json').write_text('{}');(self.base/'CASE_RESOURCE_ESTIMATES.json').write_text(json.dumps({'case':{'minimum_effective_available_host_bytes':1000,'minimum_free_disk_bytes':0}}))
  (self.leaf.parent/'memory.max').write_text('200')
  with patch('resource_gate.capacity',return_value=capacity(self.proc)):
   with self.assertRaises(RuntimeError):require_capacity(here,run)
if __name__=='__main__':unittest.main(verbosity=2)
