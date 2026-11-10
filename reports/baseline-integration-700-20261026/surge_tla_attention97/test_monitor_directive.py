import pathlib,tempfile,json,hashlib,io,contextlib,unittest
block=pathlib.Path(__file__).with_name('monitor.py').read_text().split('directive=read(P/"DISPATCH_DIRECTIVE.json",{})',1)[1]
block='directive=read(P/"DISPATCH_DIRECTIVE.json",{})'+block
class Contract(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=pathlib.Path(self.tmp.name)
  (self.p/'DISPATCH_DIRECTIVE.json').write_text('{"priority_0538":"pending"}')
 def read(self,p,default=None):
  return json.loads(p.read_text()) if p.exists() else default
 def output(self):
  out=io.StringIO()
  with contextlib.redirect_stdout(out):exec(block,dict(P=self.p,read=self.read,hashlib=hashlib,json=json))
  return out.getvalue()
 def ack(self):
  (self.p/'DISPATCH_DIRECTIVE_ACK.json').write_text(json.dumps({'directive_sha256':hashlib.sha256((self.p/'DISPATCH_DIRECTIVE.json').read_bytes()).hexdigest()}))
 def test_newer_handoff_is_not_ack(self):
  (self.p/'DISPATCH_HANDOFF.json').write_text('{}')
  self.assertIn('unconsumed_dispatch_directive',self.output())
 def test_explicit_ack_suppresses(self):
  self.ack();self.assertEqual('',self.output())
 def test_changed_directive_reappears(self):
  self.ack();(self.p/'DISPATCH_DIRECTIVE.json').write_text('{"new":"directive"}')
  self.assertIn('unconsumed_dispatch_directive',self.output())
if __name__=='__main__':unittest.main()
