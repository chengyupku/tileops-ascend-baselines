import pathlib,json,datetime,ast,tempfile,types
P=pathlib.Path(__file__).resolve().parent;D=P.parent;passed=[]
with tempfile.TemporaryDirectory(dir=P,prefix='group_gate_cpu_') as tmp:
 d=pathlib.Path(tmp);q=d/'group';q.mkdir();fixed=datetime.datetime.fromisoformat('2026-10-27T05:31:00+00:00').timestamp()
 for name in ['replace_worker.py','spawn_children.py']:
  tree=ast.parse((D/'surge_tla_attention97'/name).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ensure_dispatch_enabled');module=ast.Module(body=[fn],type_ignores=[]);clock=types.SimpleNamespace(time=lambda:fixed-120);env={'P':q,'json':json,'datetime':datetime,'time':clock};exec(compile(ast.fix_missing_locations(module),name,'exec'),env)
  def setup(enabled=True,group_enabled=True,deadline='2100-01-01T00:00:00+00:00'):
   (d/'STATE.json').write_text(json.dumps({'enabled':enabled,'phase':'running','deadline_utc':deadline}));(q/'GROUP.json').write_text(json.dumps({'enabled':group_enabled,'deadline_utc':deadline}))
  def blocked(label):
   try:env['ensure_dispatch_enabled']()
   except SystemExit:passed.append(name+': '+label)
   else:raise AssertionError(label)
  setup();env['ensure_dispatch_enabled']();passed.append(name+': valid before cutoff')
  setup(enabled=False);blocked('disabled task')
  setup(group_enabled=False);blocked('disabled group')
  setup(deadline='1970-01-01T00:00:00+00:00');blocked('shortened deadline')
  setup();clock.time=lambda:fixed+1;blocked('extended config cannot reset original cutoff')
(P/'GROUP_DISPATCH_GATE_CHECKS.json').write_text(json.dumps({'passed':len(passed),'checks':passed,'native_launches':0,'npu_access':False},indent=2)+'\n');print('Existing launcher dispatch gate CPU checks passed:',len(passed))
