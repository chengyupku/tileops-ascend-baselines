"""Exact-key GQA adapter using the isolated original GraphCall vendor path."""
from pathlib import Path
import json, os, sys
HERE=Path(__file__).resolve().parent
RECIPE={"vendor":"/opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/vendors/kdapairs_transformer","runtime_deps":"/opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/runtime_deps/site-packages"}
KEYS=set(json.loads((HERE/"verified_keys.json").read_text())["GQADenseFwdOp"])
def _fallback(case):
 from benchmarks.ops.bench_ascend_classics import PROVIDERS as classic
 return classic["GQADenseFwdOp"]["ops-transformer"](case)
def _candidate(binding,run):
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,source_hashes
 return Candidate(Implementation(run),"ops-transformer::FlashAttentionScore via isolated original GraphCall","Original BSND grouped attention; final decode query sees all KV, keepProb1.",source_hashes(Path(__file__),HERE/"attention_graph_call.py"),limitations=("Requires pinned server2 original vendor and device launch.",),cleanup=binding.cleanup)
def gqa(case):
 if case.id not in KEYS:return _fallback(case)
 os.environ["TRANSFORMER_ATTENTION_VENDOR"] = RECIPE["vendor"];os.environ["ASCEND_CUSTOM_OPP_PATH"] = RECIPE["vendor"]
 if RECIPE["runtime_deps"] not in sys.path:sys.path.insert(0,RECIPE["runtime_deps"])
 import torch
 from .attention_graph_call import GraphCall
 from benchmarks.ascend_baselines import UnsupportedCase
 q,_,_,*extra=case.inputs;p=case.params
 if any(t is not None for t in extra) or q.shape[1]!=1 or p.get("softcap") is not None or p.get("pos_encoding_mode","none")!="none" or p.get("window_size_left",-1)!=-1 or p.get("window_size_right",-1)!=-1:raise UnsupportedCase("Exact GQA scope is full-context single-query decode.")
 binding=GraphCall("ops-transformer");scale=p.get("sm_scale") or q.shape[-1]**-.5
 def run(q,k,v,*extra):
  batch,queries,heads,_=q.shape;out=torch.empty_like(q);maximum=torch.empty((batch,heads,queries,8),device=q.device,dtype=torch.float32);total=torch.empty_like(maximum);unused=torch.empty((0,0,0,0),device=q.device,dtype=q.dtype)
  binding.call("FlashAttentionScore",[q.contiguous(),k.contiguous(),v.contiguous(),*([None]*16)],[maximum,total,unused,out],{"scale_value":float(scale),"keep_prob":1.0,"pre_tockens":2147483647,"next_tockens":2147483647,"head_num":heads,"input_layout":"BSND","inner_precise":0,"sparse_mode":0})
  return out
 return _candidate(binding,run)
