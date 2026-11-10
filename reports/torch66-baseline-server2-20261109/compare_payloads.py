import pathlib,json
P=pathlib.Path(__file__).resolve().parent
rows=[]
for i in range(11,42):
 a=P/'runs'/f'{i:02d}'/'v1';b=P/'runs'/f'{i:02d}'/'v3'
 if not (a/'payload.json').exists() or not (b/'payload.json').exists():continue
 def norm(f):
  x=json.load(open(f))
  for s in x['storages']:
   for part in s['files']:part.pop('path',None)
  return x
 equal=norm(a/'payload.json')==norm(b/'payload.json')
 av=json.load(open(a/'prepare.json'))['verification'];bv=json.load(open(b/'prepare.json'))['verification']
 rows.append({'ordinal':i,'identical_entire_cpu_inputs_expected_auxiliary_sha256':equal,'identical_verification_description':av==bv})
print(json.dumps({'method':'Compare compressed lossless full-storage SHA256, shapes/dtypes/strides/alias tree, excluding only storage location. Original per-case source v3 vs v1 preparation. No NPU required.','rows':rows},indent=2))
