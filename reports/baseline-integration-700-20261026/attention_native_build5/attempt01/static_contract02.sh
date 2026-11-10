set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
S="$R/attempt02/stage/source"
python3 - "$S" <<'PY'
import ast,json,pathlib,hashlib
s=pathlib.Path(__import__('sys').argv[1]); p=s/'attention/flash_attn/torch_extension/flash_attn.py'
t=ast.parse(p.read_text())
for f in t.body:
 if isinstance(f,ast.FunctionDef) and f.name in ('flash_attn','flash_attn_metadata'):
  print(f.name,ast.unparse(f.args))
base=pathlib.Path('/opt/tileops-all-pairs/workspace')
for rel in ['TileOPs/src/tileops/manifest/spec/attention.yaml','TileOPs/workloads/attention/gqa/paged.py','TileOPs/workloads/attention/gqa/varlen.py','TileOPs/scripts/ascend_pair_check.py']:
 p=base/rel;print(rel,hashlib.sha256(p.read_bytes()).hexdigest())
PY
grep -n 'gpt-oss-sink-decode' /opt/tileops-all-pairs/workspace/TileOPs/src/tileops/manifest/spec/attention.yaml
sed -n '245,289p' "$S/third_party/cann-cmake/third_party/protobuf.cmake"
tail -10 "$R/attempt02/native_build.log"
