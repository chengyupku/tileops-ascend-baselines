import json
from pathlib import Path
from datetime import datetime, timezone
p=Path('/root/dyq/workspace/reports/baseline-recheck-608-20261026/audit_attention_final')
root='/root/dyq/workspace/'
b=root+'ascend_baselines/tilelang-ascend/'
spec=root+'TileOPs/src/tileops/manifest/spec/attention.yaml'
sink=b+'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py'
fa=b+'examples/flash_attention/flash_attn_bhsd.py'
var=b+'examples/gqa_fwd_varlen/gqa_fwd_varlen.py'
now=lambda:datetime.now(timezone.utc).isoformat()
rows=[]
contracts={
'gemma4-26b-a4b-ragged-float16':(134,512,16,2,[1000,2048,3000],[1000,2048,3000]),
'gemma4-31b-q-lt-kv-float16':(133,512,32,4,[512,512],[1024,2048]),
'gemma4-31b-uniform-float16':(132,512,32,4,[2048,2048],[2048,2048]),
'gpt-oss-q-lt-kv-float16':(127,64,64,8,[512,512],[1024,2048]),
'gpt-oss-uniform-float16':(126,64,64,8,[512]*4,[512]*4),
'llama-8b-uniform-float16':(123,128,32,8,[512]*4,[512]*4),
}
for old in json.loads((p/'SCOPE.json').read_text())['rows']:
    key=old['key']; name=key[2]
    if key[1]!='GQAVarlenFwdOp': continue
    line,d,h,hkv,ql,kl=contracts[name]
    contract={'dtype':'float16','D':d,'H':h,'H_kv':hkv,'q_lens':ql,'kv_lens':kl,'is_causal':True,'window_size_left':-1,'window_size_right':-1,'sm_scale':f'1/sqrt({d})','softcap':None,'sinks':False,'pos_encoding_mode':'none','outputs':['o']}
    evidence=[f'{spec}:{line}',f'{spec}:85',root+'TileOPs/src/tileops/ops/attention/gqa/varlen.py:79']
    if d==512:
        reason=(f'要接：D=512,H={h}/Hkv={hkv},q_lens={ql},kv_lens={kl}，无window/sinks/RoPE。拒绝旧varlen共享L0路径，但同baseline原完整flash_attention_fwd原生D512、固定64x64且QK/PV分别分配L0C。每序列、query、head独立调用：选择对应KV head的完整bottom-right可见prefix L=kv_len-q_len+i+1；将整个prefix的每个K/V token等次数复制r=64/gcd(L,64)，得到S=rL，Q复制至S行，调用batch=1,heads=1,dim=512,seq_len=S并取一行输出。所有L均正；每个原key具有相同复制次数，完整FA的softmax归一化保留原输出；输入只做选择/复制/布局，原scale=1/sqrt(512)，无外加数学、无K分段归约合并、无schedule改动。不能把普通零padding当作此适配。')
        evidence += [fa+':9',fa+':15',fa+':20',fa+':41',fa+':47',fa+':89',fa+':93',fa+':107',fa+':139',fa+':159',fa+':162',fa+':194',fa+':200',b+'examples/flash_attention/test_flash_attn_bhsd.py:46',b+'tilelang/language/ascend.py:418',b+'tilelang/language/ascend.py:496',var+':239']
        route='Original examples/flash_attention/flash_attn_bhsd.py::flash_attention_fwd; per-query full visible prefix, uniform input replication for fixed tile alignment, select output row. Original varlen and sink D512 routes rejected; no kernel changes.'
    else:
        padding='Q/K/V的D维64→128补零，输出取前64维，显式sm_scale=1/sqrt(64)' if d==64 else '保留D=128，sm_scale=1/sqrt(128)'
        reason=(f'要接：q_lens={ql},kv_lens={kl},H={h}/Hkv={hkv}。改用同baseline完整gqa_sink_fwd，而非声称D64可原样复用旧varlen L0。{padding}；按原长度传q_seqlens/kv_seqlens与已有bottom-right causal tiled mask，Sinks填FP16 -inf关闭额外分母质量，window_size=None。仅4D布局/补零/长度mask元数据/输出裁剪，保留入口固定BM64/BN128/stages14。原case无sinks/window/RoPE/softcap，仅输出o，合同相符。')
        evidence += [sink+':34',sink+':50',sink+':58',sink+':60',sink+':66',sink+':132',sink+':283',sink+':465',sink+':481',sink+':565',sink+':580',sink+':592',sink+':708',sink+':730',var+':239',b+'tilelang/language/customize.py:186',b+'src/op/ascend.cc:521']
        route='Original gqa_sink_fwd with D padding if needed, explicit original scale, -inf sinks, unchanged default schedule; existing make_attention_mask.'
    rows.append(dict(key=key,decision='要接',reason=reason,evidence=evidence,contract=contract,api_or_rejected_routes=route,decided_at=now()))
(p/'decisions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
(p/'HEARTBEAT.json').write_text(json.dumps({'at':now(),'family':'GQAVarlenFwdOp','completed':len(rows),'blockers':[]},ensure_ascii=False,indent=2))
paged={
'opt-1.3b-b1-float16':(362,1,1,32,2048,128,[2048]),
'opt-1.3b-b16-float16':(363,16,1,32,32768,128,[2048]*16),
'serving-b2-float16':(358,2,1,8,1024,256,[1024]*2),
'serving-kv1k-float16':(359,1,1,8,1024,256,[1024]),
'serving-kv512-float16':(360,1,1,8,512,256,[512]),
'speculative-decode-float16':(361,2,4,8,1024,256,[1000]*2),
}
for old in json.loads((p/'SCOPE.json').read_text())['rows']:
    key=old['key'];name=key[2]
    if key[1]!='MHADecodePagedWithKVCacheFwdOp':continue
    line,batch,sq,h,pool,page,kl=paged[name];causal=sq==4
    reason=(f'要接：B={batch},Sq={sq},H={h},D64,共享池Nkv={pool},每请求有效长度={kl},page={page},causal={causal}。按block_table顺序gather各请求实际KV序列，转4D并将Q/K/V的D维64→128补零；调用原完整gqa_sink_fwd(groups=1,dim=128,sm_scale=1/sqrt(64),window_size=None)，Sinks=-inf；Q序列补至64行，KV补至128倍数，传原q/kv有效长度及原版tiled mask，输出裁回Sq×64。保留BM64/BN128/stages14。')
    if causal:reason+=' speculative四行分别可见997/998/999/1000个keys，已有bottom-right mask与长度截断精确表达，不把池中1024行全当有效。'
    if batch==16:reason+=' 32768是共享池容量而非每请求长度；16次独立请求均只有2048个有效keys。'
    reason+=' 此API完整执行QK/softmax/PV；页表gather、补零与mask元数据足够，不接旧varlen的D64共享L0错误路径，且原合同不要求写回KV状态。'
    evidence=[f'{spec}:{line}',f'{spec}:338',f'{spec}:348',root+'TileOPs/src/tileops/ops/attention/mha.py:20',sink+':34',sink+':58',sink+':60',sink+':66',sink+':132',sink+':283',sink+':465',sink+':481',sink+':565',sink+':580',sink+':592',sink+':730',var+':239',b+'tilelang/language/customize.py:186',b+'src/op/ascend.cc:521']
    row=dict(key=key,decision='要接',reason=reason,evidence=evidence,contract={'dtype':'float16','B':batch,'S_q':sq,'H':h,'D':64,'N_kv_pool':pool,'page_size':page,'kv_lens':kl,'is_causal':causal,'sm_scale':'1/sqrt(64)','sinks':False,'window':False,'outputs':['o'],'cache_mutation':False},api_or_rejected_routes='Original gqa_sink_fwd: page gather, dimension/sequence padding, explicit original scale, -inf sinks, existing length-aware causal or noncausal tiled mask; no schedule changes.',decided_at=now())
    rows.append(row)
    with (p/'decisions.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
audit={'rows':rows,'unresolved':[],'scope':'CPU static compatibility classification only; no runtime numerical acceptance','corrections':{'old_varlen_D64_unchanged_buffers':'Rejected: T.mma derives N=128,K=D from declared reused L0 buffers; copy lowering does not reshape them to PV dimensions.','D512_blanket_rejection':'Rejected: original complete flash_attn_bhsd has separate QK/PV L0C and explicit D512 test source.','opt_b16_pool_vs_length':'N_kv=32768 is shared pool; per-request kv_len=2048.'},'adaptation_boundary':'D512 route uses uniform replication of the ENTIRE visible K/V prefix for tile alignment, then one complete FA call per independent query/head. It does not zero-pad unmasked K/V or merge separately normalized outputs. Input replication is shape/layout copying; no tensor arithmetic is added.'}
(p/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(p/'DONE.json').write_text(json.dumps({'assigned':12,'completed':12,'decision_counts':{'要接':12,'不要接':0},'remaining_keys':[],'blocked_details':[],'finished_at':now(),'runtime_validation':False},ensure_ascii=False,indent=2)+'\n')
(p/'HEARTBEAT.json').write_text(json.dumps({'at':now(),'family':'complete','completed':12,'blockers':[]},ensure_ascii=False,indent=2)+'\n')
print('Wrote 12 decisions; AUDIT.json unresolved=[]')
