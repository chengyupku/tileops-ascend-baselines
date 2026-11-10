import pathlib,json,hashlib,difflib
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-20261026/transformer_attention'; old=W/'.cache/all-pairs/resume22_server2_transformer_attention01'; receipt=json.loads((old/'native_build/attention_vendor/installed_native_receipt.json').read_text()); vendor=receipt['installed_vendor']
(P/'VENDOR.json').write_text(json.dumps(receipt,indent=2))
source=W/'TileOPs/benchmarks/ascend_graph_call.py'; text=source.read_text(); start=text.index('        self.vendor ='); end=text.index('        if self.vendor.resolve()',start); new=text[:start]+'        self.vendor = Path(os.environ["TRANSFORMER_ATTENTION_VENDOR"])\n'+text[end:]; (P/'ascend_graph_call.py').write_text(new)
(P/'provider.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/benchmarks/ascend_graph_call.py',tofile='b/benchmarks/ascend_graph_call.py')))
source2=W/'TileOPs/benchmarks/ascend_transformer_graph.py'; (P/'ascend_transformer_graph.py').write_bytes(source2.read_bytes())
pins={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [source,source2,W/'TileOPs/scripts/ascend_pair_check.py',W/'TileOPs/workloads/attention/gqa/dense.py',W/'TileOPs/src/tileops/manifest/spec/attention.yaml']};(P/'SOURCE_PINS.json').write_text(json.dumps(pins,indent=2))
print(vendor)
