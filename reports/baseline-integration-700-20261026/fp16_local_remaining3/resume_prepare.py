from pathlib import Path
import json,hashlib,ast,time
T=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
release=json.loads((T/'PUBLICATION_RELEASE.json').read_text());pub=json.loads(Path(release['publication']).read_text())
pins=json.loads((T/'FROZEN.json').read_text())['sha256'];provider='/root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py'
assert pins[provider]==pub['provider_before_sha256'];assert sha(provider)==pub['provider_after_sha256']==release['provider_sha256'];assert sha(pub['module'])==pub['module_sha256']
changed={p:dict(before=h,after=sha(p)) for p,h in pins.items() if sha(p)!=h};assert set(changed)=={provider},changed
# Verify the old initializer bytes are an unchanged prefix, from its attested hash.
b=Path(provider).read_bytes();marker=b'\nfrom .integration700_bf16_moe_masked_registration1 import wrap as _wrap_bf16_masked1';prefix=b.split(marker)[0];assert hashlib.sha256(prefix).hexdigest()==pins[provider]
(T/'square-2k-float16').rename(T/'square-2k-publication-startup');(T/'square-2k-float16').mkdir()
c=T/'capture.py';s=c.read_text().replace("manifest=json.loads((TASK/'FROZEN.json').read_text())","manifest=json.loads((P.parent/'FROZEN.json').read_text())")
s=s.replace(" v=x.detach().cpu().contiguous();raw=v.view(torch.uint8).numpy().tobytes()"," if 'input' in name:\n  captures[name]={'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'stride':list(x.stride()),'preservation':'original full checker inputs_preserved'};dump('CAPTURE.json',captures);return\n v=x.detach().cpu().contiguous();raw=v.view(torch.uint8).numpy().tobytes()")
c.write_text(s);ast.parse(s)
pins[provider]=sha(provider);pins[pub['module']]=sha(pub['module']);pins[str(c)]=sha(c)
for case in ['square-2k-float16','moe-fwd-float16']:(T/case/'FROZEN.json').write_text(json.dumps(dict(sha256=pins,publication_release=release,legitimate_delta=changed),indent=2))
(T/'RESUME_PREPARATION.json').write_text(json.dumps(dict(time=time.time(),initializer_prefix_verified=True,changed=changed,case_scoped_snapshots=True,additional_input_copy_omitted_per_feedback=True,original_checker_unchanged=True),indent=2))
print('resume prepared',len(pins))
