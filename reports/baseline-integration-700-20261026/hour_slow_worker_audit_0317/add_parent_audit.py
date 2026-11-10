import pathlib,json,sqlite3,datetime,hashlib
P=pathlib.Path(__file__).parent;B=P.parent;a=json.loads((P/'AUDIT.json').read_text());c=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
def iso(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
rows=[]
for name in ['threehost_tilelang_registration3','threehost_tilelang_registration3_audit','threehost_tilelang_bmm_registration3','threehost_tilelang3_complete_capture']:
 s=json.loads((B/name/'SPAWN.json').read_text());s=json.loads(s['stdout']) if 'stdout'in s else s;j=json.loads(c.execute('select data from jobs where id=?',(s['id'],)).fetchone()[0]);d=json.loads((B/name/'DELIVERY.json').read_text());rows.append({'task':name,'job':j['id'],'start':iso(j['created_at']),'end':iso(j['finished_at']),'seconds':j['finished_at']-j['created_at'],'window_overlap_seconds':j['finished_at']-j['created_at'],'status':d.get('status'),'accepted_added':d.get('accepted_added'),'source':name+'/DELIVERY.json','timing_source':'/root/dyq/codex-workers/workers.sqlite:jobs.data','remaining_issues':d.get('remaining_issues'),'cpu_only':name!='threehost_tilelang3_complete_capture'})
iv=sorted((datetime.datetime.fromisoformat(r['start']).timestamp(),datetime.datetime.fromisoformat(r['end']).timestamp()) for r in rows if r['cpu_only']);merged=[]
for s,e in iv:
 if not merged or s>merged[-1][1]:merged.append([s,e])
 else:merged[-1][1]=max(e,merged[-1][1])
p=B/'threehost_tilelang3_complete_capture';f=p/'server1_evidence/runs/tilelang_GQADenseFwdOp_llama-8b-4k-bfloat16/compiler_subprocess.jsonl';samples=[]
for ln,l in enumerate(f.open(),1):
 x=json.loads(l)
 if x.get('executable')=='/usr/bin/g++':samples.append({'path':str(f.relative_to(B)),'line':ln,**{k:v for k,v in x.items() if k!='environment'}})
post=[x for x in json.loads((p/'server1_evidence/TOOL_POSTFREEZE.json').read_text()) if x.get('path')=='/usr/bin/g++']
a['parent_pipeline_addendum']={'requested_by':'hour_slow_worker_audit_0317/FEEDBACK.txt 2026-10-27T03:21:08.922443Z','workers':rows,'cpu_worker_seconds':sum(r['seconds'] for r in rows if r['cpu_only']),'cpu_active_union_wall_seconds':sum(e-s for s,e in merged),'timing_limit':'均在固定窗口内；CPU工作与执行任务重叠，不能加到四worker墙钟损失；登记CPU全时长是流水线投入，不代表全部可避免。','conclusions':[{'finding':'原fullchecker通过后的3个CPU登记/闭包worker消耗约1060 worker秒，union约822墙钟秒且未登记；流水线责任包括父任务分阶段审查。','evidence':[r['source'] for r in rows[:3]],'limit':'历史20库内容缺失有真实证据必要性；不能一概把历史闭包审查称浪费。'},{'finding':'g++执行关联身份已有：observer在Popen之前hash，调用前SHA与事后同inode/hash一致。仅因没有预约前冻结而继续阻止发布，是准备程序合规偏差，不能等同原实现身份未知；无证据表明该唯一偏差需要再跑完整checker。','evidence':{'compiler_pre_exec':samples,'observer_source':'threehost_tilelang3_complete_capture/observer.py:34-46','postfreeze_source':'threehost_tilelang3_complete_capture/server1_evidence/TOOL_POSTFREEZE.json','postfreeze_gpp':post,'delivery_time':'2026-10-27T03:13:48.167059+00:00','delivery':'threehost_tilelang3_complete_capture/DELIVERY.json'},'limit':'预约前冻结要求本来已在TASK中，并非该worker事后凭空新增；把这一过程要求继续作为数值/原实现验收硬阻点会造成事实上的验收范围加严。本审计不批准注册、不证明整个编译链完整；常规执行前hash仍存在hash到exec间竞态，但并无变化证据。'},{'finding':'03:11:55父中途将scope3扩为6，与初始TASK只3上限冲突；03:13:48交付仍3且恢复本地scope3。应计入控制面交接/丢scope问题，不能按6已交付或只归执行worker慢。','evidence':[{'path':'threehost_tilelang3_complete_capture/FEEDBACK.txt','time':'2026-10-27T03:11:55.527703+00:00'},{'path':'threehost_tilelang3_complete_capture/TASK.txt','field':'只3项不可扩大'},{'path':'threehost_tilelang3_complete_capture/DELIVERY.json','time':'2026-10-27T03:13:48.167059+00:00','fields':['scope=3','scope_note=observed6 restored3']}],'limit':'新增Bmm3确切额外耗时未知；不要把scope恢复当成完成新增3项。'}]}
a['prompt_corrections'][4]='父任务派发前固定验收证据清单，区分首次使用前身份与预约前准备程序；已有pre-exec SHA+post同inode/hash时只审明确缺口，不自动要求重跑。单key通过立即交登记；scope变更需新版本+接收确认，交付逐key对账，避免3→6仍交3。'
a['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(P/'AUDIT.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
s='''最大共性：部署/观测预检不足、过宽读取及父任务多轮登记闭包，使数值通过未及时转为accepted。
窗口2026-10-27 02:17:42–03:17:42Z。依linear49/attention53/nn11/pto23：总时长3432/3076/3524/2355秒，窗口重叠2430/2074/2521/2355秒。生命周期命令229/131/150/113，输出518556/959110/1374400/1731124字符。前三者窗口命令量未知，PTO全在窗口。并行时间不能相加。
窗口数值通过8/2/0/23唯一key；PTO26通过收据含3次补证重复，并非accepted重跑。未核账本，accepted_added=0。
证据见AUDIT.json：NN于02:17缺einops，02:29/36/49三次MatMulV3失败；PTO02:51–54四次torch.npu缺失。相同错误不证明环境完全不变。attention attempt05于02:16:12–02:28:20取消，窗口639秒，events行231删除stage_profile后成功；非纯profiler耗时。linear FULL_OPP_TRANSFER_STOPPED.json于02:23:10停止整套传输，纯浪费时长未知。PTO events行59单条SDK收据1048606字符。
父流水线：3个CPU登记worker约1060工作秒、并集822墙钟秒，不能另加为损失。complete_capture的compiler_subprocess.jsonl行4在03:08:13记录g++调用前SHA；03:09:59同inode/hash复核。03:13:48 DELIVERY仅因未预约前冻结阻止发布：程序偏差不等同身份缺失，未见再跑必要。父03:11:55扩scope3→6，交付仍3，属于交接问题。
五条修正：①精确CPU预检后最快单key闭环；②同错重试必须落盘可验证变化；③只读exactkey字段、每输出≤4000字符；④禁止额外全局profiling/调线程/整套runtime复制；⑤父固定证据清单，区分预约前与首次使用前身份，scope变更需接收确认，通过立即登记。
核查4份events、83份guard及父补证；未用NPU、未重跑、未改registry/原实现/记忆。'''
assert len(s)<=1000,len(s);(P/'SUMMARY.txt').write_text(s+'\n')
for n in ['RESULT.json','DELIVERY.json']:
 d=json.loads((P/n).read_text());d['finished_at']=a['finished_at'];d['parent_pipeline_addendum']='AUDIT.json#/parent_pipeline_addendum';d['checks']['parent_pipeline_workers']=4;(P/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
(P/'HEARTBEAT.json').write_text(json.dumps({'phase':'completed','at':a['finished_at'],'accepted_added':0,'npu_used':False},indent=2)+'\n')
files=[q for q in P.iterdir() if q.is_file() and q.name not in ['FILEHASHES.json','SPAWN.json','SCOPE.json','TASK.txt','FEEDBACK.txt']];(P/'FILEHASHES.json').write_text(json.dumps({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in files},indent=2)+'\n')
print('summary_chars',len(s),'cpu_worker_s',a['parent_pipeline_addendum']['cpu_worker_seconds'],'cpu_union_s',a['parent_pipeline_addendum']['cpu_active_union_wall_seconds'])
