import pathlib,struct,json
O=pathlib.Path(__file__).parent;source=pathlib.Path('/root/dyq/workspace/reports/baseline-v3-full-20261102/server1/v3/kernel_task_profile.py');ns={};exec(compile(source.read_text(),str(source),'exec'),ns)
def record(t,start,end):
 v=[0]*16;v[0]=(t<<32)|42;v[1]=1;v[2]=1<<31;v[14]=start;v[15]=end;return struct.pack('<16q',*v)
results={}
p=O/'synthetic/cross_task_overlap';p.mkdir(parents=True,exist_ok=True);(p/'ffts_profile.bin').write_bytes(record(1,1000,2000)+record(2,1500,2500));r=ns['parse_profile'](p,1);results['cross_task_overlap']={'returned_ms':r['latency_ms'],'union_ms':0.0015,'accepted_without_task_track':True}
p=O/'synthetic/multi_device';
for dev in range(2):
 d=p/f'device_{dev}';d.mkdir(parents=True,exist_ok=True);(d/'ffts_profile.bin').write_bytes(record(1,1000,2000))
r=ns['parse_profile'](p,1);results['multi_device_id_collision']={'input_devices':2,'output_tasks':r['profile_kernel_count'],'accepted':True}
(O/'PARSER_CONTROLS.json').write_text(json.dumps(results,indent=2));print(results)
