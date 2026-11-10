import pathlib,json,shutil
P=pathlib.Path(__file__).resolve().parent; B=P.parent;old=B/'threehost_tla_other54_r2'
diff=json.loads((B/'threehost_tilelang_registration3_audit/EVIDENCE_DIFF.json').read_text())
(P/'RUNTIME_PATHS.json').write_text(json.dumps(list(diff['runtime_library_records']),indent=2))
s=(old/'run_case.py').read_text().replace('baseline700-20261026/threehost_tla_other54_r2','baseline700-20261026/threehost_tilelang3_complete_capture')
s=s.replace("d=P/'cache'/n","d=out/'cache'/n")
s=s.replace('import torch,tilelang',"import observer\nobserver.install(P,out)\nimport torch,tilelang")
s=s.replace("(out/'completion.json').write_text", "observer.finish()\n (out/'completion.json').write_text")
(P/'run_case.py').write_text(s)
# Keep original observer/native loader, source math adapter, checker and closure intact.
s=(P/'registration.py').read_text().replace("threehost_tilelang_registration3_audit')","threehost_tilelang3_complete_capture')")
(P/'registration.py').write_text(s)
shutil.copy2(old/'server1_evidence/tilelang_adapter.py',P/'tilelang_adapter.py')
# scope uses original runner's schema, limited to three contracts
rows=json.loads((P/'CONTRACTS.json').read_text());(P/'SCOPE.json').write_text(json.dumps([dict(baseline='tilelang',operator=r['op'],case_id=r['case_id']) for r in rows],indent=2))
