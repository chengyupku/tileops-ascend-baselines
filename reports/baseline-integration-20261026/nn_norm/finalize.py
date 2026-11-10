import pathlib,json,datetime
D=pathlib.Path(__file__).resolve().parent;rows=[];s=json.loads((D/'SCOPE.json').read_text());apis={'RMSNormFwdOp':'RmsNorm','GroupNormFwdOp':'GroupNormV2','InstanceNormFwdOp':'GroupNormV2','LayerNormFwdOp':'LayerNormV4'}
for x in s:
 op,cid=x['operator'],x['case_id'];p=D/(op+'--'+cid);c=p/'output/checker.json';g=p/'GUARD_RESULT.json';cd=json.loads(c.read_text()) if c.exists() else {};gd=json.loads(g.read_text()) if g.exists() else {};v=cd.get('verification',{})
 row={'key':x['key'],'api':apis[op],'changed_files':['packet/run_case.py','packet/frozen/normalization.py','launch.py'],'guard_path':str(g) if g.exists() else None,'guard_log':str(p/'guard.log') if p.exists() else None,'checker_path':str(c) if c.exists() else None,'guard_exit_code':gd.get('returncode'),'checked_outputs':v.get('checked_outputs',0),'total_outputs':v.get('total_outputs'),'status':'blocked','reason':cd.get('error','No device result')}
 if cd.get('status')=='verified' and gd.get('returncode')==0 and v.get('checked_outputs') and v.get('checked_outputs')==v.get('total_outputs'):row.update(status='verified',reason=None)
 if cid=='image-eval-float16':row.update(api='BatchNormV3',total_outputs=3)
 if cid=='wider-channel-affine-float16':
  attempt=D/(op+'--'+cid+'--batchnorm');a=json.loads((attempt/'output/checker.json').read_text());row['additional_attempt']={'api':'BatchNormV3','checker_path':str(attempt/'output/checker.json'),'guard_path':str(attempt/'GUARD_RESULT.json'),'error':a.get('error')};row['total_outputs']=1
 if row['total_outputs'] is None:row['total_outputs']=1
 rows.append(row)
(D/'RESULT.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_count':len(s),'verified':sum(r['status']=='verified' for r in rows),'rows':rows},indent=2))
