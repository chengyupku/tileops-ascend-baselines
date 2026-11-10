from pathlib import Path
import json,hashlib,difflib,shutil,ast
W=Path('/opt/tileops-all-pairs/workspace');R=W/'reports/baseline-integration-20261026';D=R/'register_server2/repair01';P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):(D/name).write_text(json.dumps(obj,indent=2)+'\n')
m=json.loads((P/'validated_manifest.json').read_text());backup=json.loads((D/'BACKUP_HASHES.json').read_text());changes=[];diff=[]
for f in sorted(P.iterdir()):
 if not f.is_file():continue
 before=D/'backup/provider'/f.name;old=before.read_text() if before.exists() else '';new=f.read_text()
 if old!=new:
  changes.append({'remote_path':str(f),'before_sha256':sha(before) if before.exists() else None,'after_sha256':sha(f),'backup':str(before) if before.exists() else None})
  diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/'+f.name,tofile='after/'+f.name))
 if f.suffix=='.py':ast.parse(new)
 dest=D/'provider'/f.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(f,dest)
(D/'complete.diff').write_text(''.join(diff));dump('CHANGED_FILES.json',changes)
# Verify original reports have not changed after backup.
for path,h in backup.items():
 if '/ops_transformer/' not in path:assert sha(Path(path))==h,path
for name in ['__init__.py','adapters.py','build.py','check.py','run.sh','attention_graph_call.py','verified_keys.json']:
 assert sha(P/name)==sha(D/'backup/provider'/name),name
mapping=json.loads((D/'KEY_FACTORY_RECEIPTS.json').read_text());assert len(mapping)==39
checks=json.loads((D/'STATIC_CHECKS.json').read_text());assert checks['status']=='passed'
dump('DELIVERY.json',{
 'host':'server2','worker':'42d492144d234fa5','status':'registration_repaired_cpu_validated_no_device_replay',
 'numerical_evidence':'39 original successful checker/guard receipts preserved; no new numerical acceptance',
 'scoped_previously_verified_key_count':39,'formal_registration_validation':'CPU/static repair complete; parent review pending; no new device integration receipt','attention_keys':13,'gdn_keys':26,'excluded_keys':10,
 'resolved':[
 'Attention vendor read from original VENDOR.installed_vendor; installed closure and all original receipt source hashes verified.',
 'Full original attention functions privately loaded with import relocation only; AST identity proven. Startup env checked before factory runtime use.',
 'GDN CPU launcher establishes original child environment/LD ordering/python paths/cwd before exec; resource gate before exec and at factory; unchanged checker or ordinary caller without registry monkeypatch.',
 'GDN original build and two outputs preserved; exact source-pinned private binding/direct_acl/resource_gate; guard/pin/environment/contract checks enforced.',
 'Historical candidate_factories fallback preserved for unknown and all excluded keys; BMM unchanged.'],
 'artifacts':{'deployment':'DEPLOYMENT.md','changes':'CHANGED_FILES.json','diff':'complete.diff','mapping':'KEY_FACTORY_RECEIPTS.json','static_checks':'STATIC_CHECKS.json','environment_checks':'ENVIRONMENT_CHECKS.json','receipt_source_identity':'RECEIPT_SOURCE_IDENTITY.json','backup_hashes':'BACKUP_HASHES.json'},
 'checks':{'cpu_assertions':checks['count'],'source_AST':'all functions/classes identical; explicit import relocation only','NPU_initializations':0,'numeric_replays':0,'kernel_compiles':0,'original_reports_unchanged':True},
 'remaining':[
 'Device end-to-end launch deliberately not executed under CPU-only assignment; static/mock validation is not a new device execution receipt.',
 'Original 10 excluded keys retain prior oracle/resource blockers; none accepted by new branches.',
 'One original GDN Python search entry does not exist; retained as inert optional search path, all actual dependency pins match. See PATH_PIN_AUDIT.json.',
 'Fresh execution must pass foreground guard and current resource capacity, using provider launch_verified.py before runtime imports.'],
 'modified_scope':'Only authorized ops_transformer package and register_server2/repair01 report directory; no shared registry/runtime/vendor/original receipt/memory edits.'})
print('DELIVERY',len(changes),'changed provider files;',checks['count'],'checks')
