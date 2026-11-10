set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
python3 - "$R" <<'PY'
import pathlib,json,hashlib,tarfile,sys
r=pathlib.Path(sys.argv[1]); s=r/'attempt03/stage/source'
rows={}; mismatches=[]
with tarfile.open(r/'attempt01/original_source.tar') as t:
 for m in t:
  if not m.isfile():continue
  rel=m.name.removeprefix('./'); expected=hashlib.sha256(t.extractfile(m).read()).hexdigest();rows[rel]=expected
  p=s/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:mismatches.append(rel)
(r/'ORIGINAL_SOURCE_PINS.json').write_text(json.dumps(dict(files=rows,mismatches=mismatches),indent=2))
print('frozen files',len(rows),'mismatches',mismatches)
PY
for n in runtime opbase hcomm ge-executor metadef ge-compiler asc-devkit bisheng-compiler; do cat "/usr/local/Ascend/cann-9.1.0/share/info/$n/version.info" | head -4; done
sha256sum "$R/attempt03/stage/source/build.sh" "$R/attempt03/stage/source/attention/flash_attn/torch_extension/flash_attn.py"
tail -10 "$R/attempt03/native_build.log"
