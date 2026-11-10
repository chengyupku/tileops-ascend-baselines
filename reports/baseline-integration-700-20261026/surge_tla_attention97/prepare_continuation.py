import pathlib,json,datetime,shutil,sys
P=pathlib.Path(__file__).resolve().parent;w=P/sys.argv[1];tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S');a=w/('checkpoint_'+tag);a.mkdir()
for n in ['RESULT.json','DELIVERY.json','HEARTBEAT.json','FILEHASHES.json']:
 if (w/n).exists():shutil.copy2(w/n,a/n)
scope=json.loads((w/'SCOPE.json').read_text());x=json.loads((w/'RESULT.json').read_text()) if (w/'RESULT.json').exists() else [];rr=x if isinstance(x,list) else x.get('exactkeys',x.get('rows',x.get('results',[])));todo=[]
for row in scope:
 matches=[r for r in rr if r.get('key')==row['key']];r=matches[-1] if matches else {};st=r.get('status','unknown')
 if any(t in st for t in ['pending','unverified','not_run','not_started','guard_not_started','guard_requests','guard_visibility','guard_refused']) or not matches:todo.append({'key':row['key'],'previous_status':st,'previous_evidence':r.get('evidence',r.get('guard'))})
(w/'CONTINUATION_SCOPE.json').write_text(json.dumps(todo,ensure_ascii=False,indent=2))
extra=''
if w.name=='worker06':
 extra='你优先负责../DEPLOYMENT_TYPEERROR.json中llama-8b-p256-float16-float16的Path(None)薄部署/取证问题。这是新的明确假设，不是数值失败重放。先CPU定位外层jit_evidence Path(adapter.libpath)与private capture，原guard中只验证一个case的新假设；原native身份不能跳过，不改共享frozen checker。其他未跑key随后。'
text=(w/'TASK.txt').read_text()+f'''\n预算续接任务（不扩scope）：旧native已经退出，旧快照在{a.name}。你仍只拥有本SCOPE原key，CONTINUATION_SCOPE.json列出候选未启动/未验证集合；核实旧证据，已有完整数学失败、原编译/API限制和未变条件的实际device timeout不重跑。继承所有已有recipe/适配器，禁止从头大历史扫描。\n原native退出不证明远端guard已退出！开始先用原guard status及已有命令/收据确认旧owner前台请求是否仍运行。旧key还在执行时只做CPU准备或其他独立未跑key；不得与旧进程写同一run目录。保留全部旧收据，所有新尝试用continuation时间戳路径。不得kill外部进程或reset。\n此次native授权guard --wait600（由旧120调整以减少FIFO队列过期）；只改自己私有launcher等待预算，设备timeout按case原预算且<=600，NPU run全程前台。每次提交前核三机能力。截止05:31；任何新提交必须能在剩余时间容纳等待+设备timeout+收集，不盲目撑满600。05:20之后不得启动不能在截止前收口的长请求。\n不要用代表case编译失败伪装未跑exact key失败，但可明确共同原API/编译依赖阻点而停止同条件重复诊断。允许修原依赖部署和薄顶层接口，禁止kernel/compiler/pass/checker/reference/seed/tolerance修改。若无合法可执行工作，明确逐key阻点并完成交付，不填充重复任务。\n{extra}\n完成时汇总原整份SCOPE，写RESULT/DELIVERY/FILEHASHES/HEARTBEAT，accepted_added=0，成功则立即逐key发布准备。父唯一登记。\n'''
(w/'TASK_CONTINUE.txt').write_text(text);print(json.dumps({'worker':w.name,'checkpoint':a.name,'candidate_remaining':len(todo),'special_typeerror':w.name=='worker06'}))
