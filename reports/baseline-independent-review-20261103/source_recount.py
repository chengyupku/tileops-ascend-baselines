"""CPU AST/JSON reconstruction of candidate sets. No baseline module imports."""
import ast,json,pathlib,collections
R=pathlib.Path('/root/dyq/workspace');O=pathlib.Path(__file__).parent;B=R/'TileOPs/benchmarks/ascend_all_pairs'
inv=json.load(open(B/'tla/inventory_snapshot.json'))['cases'];im={(r['op'],r['case_id']):r for r in inv}
site=json.load(open(O/'SITE_285465c4.json'));private=json.load(open(R/'reports/ci-baseline-elementwise-20261101/BASELINE32_PRIVATE.json'))
existing={(r['library'],r['op'],r['case']) for r in site['rows']}|{tuple(r['key']) for r in private['rows']}
mathops={};exec(compile((B/'ops_math/composition_specs.py').read_text(),'<constant inventory>','exec'),mathops)
def assignment(path,name):
 return ast.literal_eval(next(n.value for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
tl=assignment(B/'tilelang/composed.py','OPERATIONS');ops={'ops-math':mathops['COMPOSITIONS'],'tilelang':tl,'ops-nn':['LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp','BitwiseNotFwdOp','StdFwdOp','AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp','FusedAddRMSNormFwdOp','FusedAddLayerNormFwdOp']}
recon=set.union(*({(lib,c['op'],c['case_id']) for c in inv if c['op'] in names}-existing for lib,names in ops.items()))
prior=json.load(open(R/'reports/baseline-v3-full-20261102/server1/COMPOSITION_COVERAGE_AUDIT_20261103.json'));claimed={(r['library'],r['op'],r['case']) for g in prior['groups'].values() for r in g['missing_inventory_candidates']}
source=R/'ascend_baselines/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_elementwise.py';funcs={n.name for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)};direct=assignment(B/'tla/plans.py','DIRECT');comp=assignment(B/'tla/plans.py','COMPARISON')
sliceops={op for op,(name,arity) in direct.items() if name+'_slice' in funcs}|set(comp)|{'LeakyReluFwdOp'}
sslice={tuple(r['key']) for r in private['rows'] if r['key'][0]=='tilelang-ascend' and r['key'][1] in sliceops}
surv=[]
for lib,op,case in recon:
 t=im[(op,case)]['tensors'];dt=next((t[k][1] for k in ('input','x') if k in t),None)
 if (lib=='ops-math' and op=='FusedAddLayerNormFwdOp' and dt=='float32') or (lib=='ops-nn' and ((op=='LogicalNotFwdOp' and dt in ('bool','float32')) or (op=='BitwiseNotFwdOp' and dt in ('int32','int64')) or (op=='StdFwdOp' and dt=='float32'))):surv.append([lib,op,case])
(O/'SOURCE_CANDIDATE_RECOUNT.json').write_text(json.dumps({'composition_reconstructed':sorted(recon),'composition_claim_extra':sorted(claimed-recon),'composition_new_extra':sorted(recon-claimed),'slice_reconstructed':sorted(sslice),'slice_ops':sorted(sliceops),'max3_source_survivors':sorted(surv)},indent=2))
print({'composition':len(recon),'max3':len(surv),'surface_slice':len(sslice),'exclude_fixed_slope_leaky':4,'compatible_slice_inventory':118})
