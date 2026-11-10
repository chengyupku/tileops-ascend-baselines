import json,pathlib,datetime,collections
p=pathlib.Path(__file__).parent
src=json.load(open(p/'SCOPE.json'))['rows']; root='/root/dyq/workspace/'
rows=[]; unresolved=[]
def ev(*a):return [root+x for x in a]
for r in src:
 b,o,c=r['key']; d=r['decision']; e=[]; why=''
 if o=='Conv3dFwdOp':
  unresolved.append({'key':r['key'],'previous_decision':d,'missing':'需主审明确vendored CATLASS完整示例是否算sgl baseline原版可用入口；不能沿用“本地原版不存在完整Conv3d”这一事实断言。','evidence':ev('ascend_baselines/sgl-kernel-npu/third_party/catlass/examples/24_conv_bias/README.md:10','ascend_baselines/sgl-kernel-npu/third_party/catlass/examples/24_conv_bias/conv_bias.cpp:187','ascend_baselines/sgl-kernel-npu/third_party/catlass/examples/python_extension/src/wrapper/conv.cpp:GetKernelInfo','TileOPs/benchmarks/ascend_all_pairs/sgl/convolution.py:30'),'details':'sgl自有python/csrc无Conv3d导出，原conv_nd的unfold+GEMMV确属新增分解；但third_party/catlass有真实Conv3d示例及wrapper，读取其README和host确认。示例固定half，另有shared_lib；若纳入，需要进一步确认对应case dtype、groups和layout入口，不能据README直接判全部要接。'})
  continue
 if b=='sgl-kernel-npu' and o=='KDAFwdOp':
  d='不要接'; why='合同显式use_qk_l2norm_in_kernel=true；chunk_kda_fwd完整签名仅有gate/state/output旗标，没有Q/K归一化，body直接转发torch.ops，host也不执行L2。另调l2norm_fwd是新增归约数学，非layout适配。raw-gate虽有原生use_gate_in_kernel仍不能补QK归一化。'
  if 'float16' in c and 'bfloat16' not in c:why+=' 此FP16 case还被host的BF16-only q/k/v检查拒绝，FP16→BF16不是无损扩宽。'
  e=ev('TileOPs/src/tileops/manifest/spec/linear_attention.yaml:272','ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk_kda_fwd.py:6','ascend_baselines/sgl-kernel-npu/csrc/chunk_kda_fwd/op_host/chunk_kda_fwd.cpp:340')
 elif o=='SSDChunkStateFwdOp':
  e=ev('TileOPs/src/tileops/manifest/spec/mamba.yaml:80','ascend_baselines/ops-transformer/experimental/mamba/mamba2_chunk_state/torch_interface.cpp:mambav2_chunk_state','ascend_baselines/ops-transformer/experimental/mamba/mamba2_chunk_state/op_kernel/CustVec.h:140')
  if 'seq-idx' in c or '1p3b' in c:
   d='不要接'; why='该workload含seq_idx，必须仅累计与chunk末token同序列的贡献；原mambav2_chunk_state只有dt_out,dacs,bt,xt四输入，Process_part1对所有token直接exp(last_dA-dA)*dt，没有序列标签或mask分支。外部乘mask为新增数学。'
  elif 'bfloat16' in c:
   d='不要接'; why='原完整host无条件将bt和xt转为at::kHalf，kernel固定half矩阵；BF16指数范围无法无损装入FP16，即使外部先转FP32也会在host转回half，不能满足BF16输入精度合同。'
  else:
   d='要接';why='FP16 x/Bmat、FP32 dt/dA、Q256/H48/P64/N128满足原例分块；host输出FP32[B,NC,H,N,P]，转置末两维即可。输入只需把序列reshape为chunk并把dt/dA转成[B,NC,Q,H]；Process_part1原生exp(dA_last-dA)*dt，无seq_idx需求。'
 elif o=='DeltaNetChunkFwdOp':
  d='不要接';why='合同返回o、FP32全chunk边界S、Aw、Au、w、u。chunk_gated_delta_rule_fwd虽能g=0/scale=1表达输出，但完整返回只有g,o,A,final_state,w,h,v_new，不返回u；h由k.new_empty分配成输入dtype，不是合同FP32边界状态，转FP32不能恢复精度。另调recompute_w_u或重建状态不属选择原版已返回结果。'
  e=ev('TileOPs/src/tileops/manifest/spec/linear_attention.yaml:432','ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py:chunk_gated_delta_rule_fwd','ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk_delta_h.py:241','ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/wy_fast.py:122')
 elif o=='MHADecodePagedWithKVCacheFwdOp':
  d='要接';why='合同仅返回o，无LSE或cache更新；按block_table重排完整K/V并裁到kv_lens，独立batch调用原aclnnFlashAttentionScore，FP16、headNum和scale=D^-0.5参数直接匹配，选attentionOut即可。'
  if 'speculative' in c:why+=' Q4/KV1000使用RIGHT_DOWN_CAUSAL模式并传其要求的压缩因果mask，保留右下对齐：第i个query可见到996+i；不能只设置sparseMode而省略attenMask。'
  else:why+=' S_q=1且is_causal默认false，全可见，无额外mask数学。'
  e=ev('TileOPs/src/tileops/manifest/spec/attention.yaml:332','ascend_baselines/ops-transformer/attention/flash_attention_score/op_api/aclnn_flash_attention_score.h:25','ascend_baselines/ops-transformer/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.cpp:191','ascend_baselines/ops-transformer/attention/flash_attention_score/docs/aclnnFlashAttentionScore.md:428')
 elif o=='MLAVarlenFwdOp':
  d='不要接';why='合同是每head独立k_nope和V的causal prefill，并返回FP32自然对数lse。旧paged MLA固定half、h_kv=1、KV同时作K和V且仅返回o；不能满足该合同。查普通varlen MHA/GQA也仅输出o；bwd示例的fwd虽输出lse，却明确T.log2(logsum)+max*log2(e)，需额外乘ln2才是合同lse，超出允许适配，且固定FP16。'
  e=ev('TileOPs/src/tileops/manifest/spec/attention.yaml:602','ascend_baselines/tilelang/examples/deepseek_mla/example_mla_decode_paged.py:16','ascend_baselines/tilelang/examples/flash_attention/example_mha_fwd_varlen.py:27','ascend_baselines/tilelang/examples/flash_attention/example_mha_bwd_bshd.py:77')
 elif o=='GeluTanhAndMulFwdOp':
  why='原BiasGateGelu测试reference明确approximate=tanh，kernel先Gelu再Mul，零bias可满足无bias合同；[2048,28672]列均分与输出[2048,14336]一致。';d='要接'
  if 'bfloat16' in c:d='不要接';why='原完整BiasGateGelu输入/输出/缓存都硬编码half，没有dtype参数；BF16→FP16不是无损扩宽，无法满足该BF16 case。原测试确为tanh GELU，排除依据是dtype而非激活名称。'
  e=ev('TileOPs/src/tileops/manifest/spec/elementwise_fused_gated.yaml:56','ascend_baselines/ops-transformer/experimental/moe/biasgategelu/BIAS_GATE_GELU.cpp:119','ascend_baselines/ops-transformer/experimental/moe/biasgategelu/tests/biasgategelu.py:26')
 elif o=='MoEGroupedGemmFwdOp' and b=='tilelang-ascend':
  if 'down' in c:
   unresolved.append({'key':r['key'],'previous_decision':d,'missing':'原grouped_gemm同dtype输出不足以排除无损BF16→FP32后调用原版泛型GEMM；需补证泛型dtype=float32的真实底层支持。','evidence':ev('ascend_baselines/tilelang-ascend/examples_experiment/grouped_gemm/example_grouped_gemm_fwd.py:45','ascend_baselines/tilelang-ascend/examples/gemm/example_gemm_pto_developer.py:25'),'details':'后者参数dtype同时控制A/B/C、累加float，表面允许FP32；未编译且禁止编译，不能凭参数宣称底层支持，也不能照旧以输出dtype固定为由排除。'})
   continue
  d='不要接';why='合同只需grouped GEMM后SwiGLU。原grouped_gemm仅线性输出；完整shared_expert_kernel/routed_expert_kernel内部虽有F.silu及乘法，却继续执行down projection并只返回最终MLP输出，不公开gate-up激活结果。另接swi_glu或改pipeline返回中间值为新增数学/改原版入口。'
  e=ev('TileOPs/src/tileops/manifest/spec/moe.yaml:300','ascend_baselines/tilelang-ascend/examples_experiment/grouped_gemm/example_grouped_gemm_fwd.py:45','ascend_baselines/tilelang-ascend/examples/fusedmoe/example_fusedmoe.py:957','ascend_baselines/tilelang-ascend/examples/fusedmoe/example_fusedmoe.py:1000')
 elif o=='DSAPagedFwdOp':
  d='不要接';why='合同cache每行656字节含512 E4M3、4个FP32分块scale、64 BF16 rope，返回o及FP32 lse。QuantSparseFlashMLA虽然原生输出lse，但q/KV为hifloat8且scale只有[1] per-tensor，q_d/kv_d=512；SparseFlashAttention普通QKV接口也不能直接解释该混合cache。外部反量化/拼接数学不能作layout适配。'
  e=ev('TileOPs/src/tileops/manifest/spec/attention.yaml:740','ascend_baselines/ops-transformer/attention/quant_sparse_flash_mla/docs/torchapi_quant_sparse_flash_mla.md:188','ascend_baselines/ops-transformer/attention/quant_sparse_flash_mla/docs/torchapi_quant_sparse_flash_mla.md:325','ascend_baselines/ops-transformer/attention/sparse_flash_attention/docs/aclnnSparseFlashAttentionV2.md:46')
 elif o=='GemmFP8FwdOp':
  d='不要接';why='该case有两个运行时FP32[1,1] scale。FP8数值本身可无损拓宽到FP16/BF16，但sgemmv_shrink唯一LoRA scale按half读取，且没有独立两FP32 scale入口；合成scale或事先缩放输入是新增数学。故不能由输入dtype拓宽解决scale精度合同。'
  e=ev('TileOPs/src/tileops/manifest/spec/gemm.yaml:62','ascend_baselines/sgl-kernel-npu/csrc/lora/op_host/sgemmv_shrink.cpp:49','ascend_baselines/sgl-kernel-npu/csrc/lora/op_kernel/sgemmv_shrink_kernel.cpp:58')
 elif o=='SinFwdOp':
  d='要接';why='原sin(M,N,block_M,block_N,dtype)完整kernel执行T.tile.sin后直接store；16384²可用64/128整除分块。采用FP32输入无损扩宽并dtype=float，选原例已验证的FP32路径，最终转回合同dtype，避免把BF16硬件支持当成未经证明前提。'
  e=ev('TileOPs/src/tileops/manifest/spec/elementwise_unary_math.yaml:SinFwdOp','ascend_baselines/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_elementwise.py:4050')
 elif o=='BmmFwdOp':
  d='要接';why='B8/M=N=K2048 FP16可无损扩为FP32；把A沿N广播、B沿M广播成逐输出点积输入，count_sum置1，完整ReduceMeanWithCount内部本来就执行sum(x*count/count_sum)，axes=K直接产出各点积。不外接乘法或分块归约；原L0完整入口分配输出并dispatch，支持950/FP32。'
  e=ev('TileOPs/src/tileops/manifest/spec/gemm_bmm.yaml:BmmFwdOp','ascend_baselines/ops-math/math/reduce_mean_with_count/op_api/reduce_mean_with_count.cpp:ReduceMeanWithCount','ascend_baselines/ops-math/math/reduce_mean_with_count/op_kernel/arch35/reduce_mean_with_count_dag.h:ReduceMeanWithCountDag','ascend_baselines/ops-math/math/reduce_mean_with_count/op_host/reduce_mean_with_count_def.cpp:23')
 elif o=='ProdFwdOp':
  d='要接';why='[64,32768]末轴prod可转置到[32768,64]，全零segment_ids、num_segments=1，由完整UnsortedSegmentProd直接完成全部32768项归约，输出[1,64] reshape为[64]；原API支持BF16，也可无损FP32扩宽计算再转回。'
  e=ev('TileOPs/src/tileops/manifest/spec/reduction.yaml:185','ascend_baselines/ops-nn/index/unsorted_segment_prod/op_graph/unsorted_segment_prod_proto.h:21','ascend_baselines/ops-nn/index/unsorted_segment_prod/op_host/config/ascend950/unsorted_segment_prod_binary.json:1')
 elif o=='TopKSelectFwdOp':
  d='不要接';why='合同每行65536个FP32分数选1024个全局索引；原MoeGatingTopKSoftmax最后维明确限2048。行batch可切分但归约轴不能切分后外部归并，否则引入新top-k算法，不能保持原版完整调用。'
  e=ev('TileOPs/src/tileops/manifest/spec/attention_indexing.yaml:48','ascend_baselines/ops-transformer/moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md:274')
 elif o=='MoEGroupedGemmFwdOp':
  d='要接';why='P8/E16/K2048/N1536，无activation；physical_psum仅划分独立expert行。按expert切片，M零填充到原例tile、B转置布局，调用gemm(dtype=bfloat16,out_dtype=bfloat16)再裁回有效行；完整原API无跨expert数学，零行也可跳过。'
  e=ev('TileOPs/src/tileops/manifest/spec/moe.yaml:331','ascend_baselines/tilelang/examples/ascend/example_gemm.py:11')
 elif o=='GemmINT8W8A8FwdOp':
  d='要接';why='M128/N6144/K4096均可整除32；原int8_gemm_kernel_corrected明确A/B int8、INT32累加、scale_a[M]/scale_b[N] FP32，kernel内部转换累加并乘两scale，out_dtype可选BF16。仅reshape合同[M,1]/[1,N] scales并传BF16输出，无bias需求。'
  e=ev('TileOPs/src/tileops/manifest/spec/gemm.yaml:244','ascend_baselines/tilelang-ascend/examples/deepseek_v4/int8_gemm.py:35')
 elif b=='vllm-ascend' and o=='KDAFwdOp':
  d='要接';why='16条8192 packed、H32/K=V128 BF16：原chunk_kda公开use_qk_l2norm_in_kernel，内部原生调用L2并默认scale=K^-0.5；cu_seqlens原生分隔状态。无initial_state时传零FP32状态，原状态V,K布局可转置为合同K,V，output_final_state=True原生返回FP32终态；无需新增激活。'
  e=ev('TileOPs/src/tileops/manifest/spec/linear_attention.yaml:342','ascend_baselines/vllm-ascend/vllm_ascend/ops/triton/kda/kda.py:1346','ascend_baselines/vllm-ascend/vllm_ascend/ops/triton/kda/chunk_delta_h.py:302')
 else:raise RuntimeError(r['key'])
 rows.append({'key':r['key'],'decision':d,'reason':c+'：'+why,'evidence':e,'review':'accepted' if d==r['decision'] else 'corrected'})
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (p/'decisions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 (p/'HEARTBEAT.json').write_text(json.dumps({'time':now,'completed':len(rows),'current_family':o,'unresolved':len(unresolved)},ensure_ascii=False))
a={'rows':rows,'unresolved':unresolved,'method':'仅CPU文本/JSON检查；没有导入项目模块、运行编译或NPU。未修改原分类文件。','counts':dict(collections.Counter(r['decision'] for r in rows))}
(p/'AUDIT.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
(p/'QUESTIONS.json').write_text(json.dumps({'unresolved':unresolved},ensure_ascii=False,indent=2)+'\n')
(p/'DONE.json').write_text(json.dumps({'assigned':len(src),'completed':len(rows),'decision_counts':a['counts'],'review_counts':dict(collections.Counter(r['review'] for r in rows)),'remaining_keys':[r['key'] for r in unresolved],'blocked_details':unresolved},ensure_ascii=False,indent=2)+'\n')
print(a['counts'],len(rows),len(unresolved),collections.Counter(r['review'] for r in rows))
