import importlib.util,json,pathlib,struct,tempfile,unittest
P=pathlib.Path('/root/dyq/workspace/reports/baseline-v3-full-20261102/server1/v3/kernel_task_profile.py')
s=importlib.util.spec_from_file_location('p',P);p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
class ParserTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.r=pathlib.Path(self.tmp.name);self.d=self.r/'device_0/data';self.d.mkdir(parents=True)
  (self.d.parent/'info.json.0').write_text(json.dumps({'platform_version':'15','DeviceInfo':[{'hwts_frequency':1000}]}))
  self.track([(10,1,1),(11,1,2)])
  (self.d/'hash_dic.slice_0').write_text('1:Add\n2:Sub\n')
  self.ffts();self.star()
 def tearDown(self):self.tmp.cleanup()
 def track(self,rows):
  (self.d/'task_track.slice_0').write_bytes(b''.join(struct.pack('<8q',0,0,0,(seq<<32)|(stream<<16),0,h,0,0) for seq,stream,h in rows))
 def ffts(self):
  rows=[]
  for role in [0,1<<32]:
   v=[0]*16;v[0]=(10<<32)|42;v[1]=10;v[2]=role;v[14]=1000;v[15]=3000;rows.append(struct.pack('<16q',*v))
  (self.d/'ffts_profile.data').write_bytes(b''.join(rows))
 def star(self,kind=5):
  (self.d/'stars_soc.data.0.slice_0').write_bytes(b''.join(struct.pack('<4HQHH3L',(kind<<10)|fn,0,11,0,t,0,0,0,0,0) for fn,t in [(0,4000),(1,9000)]))
 def test_mix_and_aicpu(self):
  x=p.parse_profile(self.r,1);self.assertAlmostEqual(x['latency_ms'],.007);self.assertEqual(x['profile_kernel_count'],2)
 def test_aicpu_only(self):
  (self.d/'ffts_profile.data').unlink();self.track([(11,1,2)]);self.assertAlmostEqual(p.parse_profile(self.r,1)['latency_ms'],.005)
 def test_missing(self):
  (self.d/'stars_soc.data.0.slice_0').unlink()
  with self.assertRaisesRegex(ValueError,'Uncovered'):p.parse_profile(self.r,1)
 def test_truncated(self):
  (self.d/'stars_soc.data.0.slice_0').write_bytes(b'x')
  with self.assertRaisesRegex(ValueError,'Truncated'):p.parse_profile(self.r,1)
 def test_unclosed(self):
  f=self.d/'stars_soc.data.0.slice_0';f.write_bytes(f.read_bytes()[:32])
  with self.assertRaisesRegex(ValueError,'Unclosed'):p.parse_profile(self.r,1)
 def test_unknown_task(self):
  self.star(7)
  with self.assertRaisesRegex(ValueError,'Unsupported'):p.parse_profile(self.r,1)
 def test_large_star_clock(self):
  base=1670885013873567
  f=self.d/'stars_soc.data.0.slice_0'
  f.write_bytes(b''.join(struct.pack('<4HQHH3L',(5<<10)|fn,0,11,0,base+t,0,0,0,0,0) for fn,t in [(0,0),(1,5001)]))
  self.assertAlmostEqual(p.parse_profile(self.r,1)['latency_ms'],.007001,places=12)
 def test_no_task_track(self):
  (self.d/'task_track.slice_0').unlink()
  with self.assertRaisesRegex(ValueError,'Missing named'):p.parse_profile(self.r,1)
 def test_multistream(self):
  self.track([(10,1,1),(11,2,2)])
  with self.assertRaisesRegex(ValueError,'Multi-stream'):p.parse_profile(self.r,1)
 def test_exclusion(self):
  self.assertAlmostEqual(p.parse_profile(self.r,1,exclude_names={'Sub'})['latency_ms'],.002)
 def test_digest_covers_star_and_info(self):
  before=p.trace_digest(self.r);self.star(4);self.assertNotEqual(before,p.trace_digest(self.r));before=p.trace_digest(self.r)
  f=self.d.parent/'info.json.0';f.write_text(f.read_text()+' ');self.assertNotEqual(before,p.trace_digest(self.r))
 def test_split_star(self):
  f=self.d/'stars_soc.data.0.slice_0';b=f.read_bytes();f.write_bytes(b[:32]);(self.d/'stars_soc.data.0.slice_1').write_bytes(b[32:]);self.assertAlmostEqual(p.parse_profile(self.r,1)['latency_ms'],.007)
if __name__=='__main__':unittest.main()
