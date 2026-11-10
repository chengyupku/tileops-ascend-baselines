import pathlib,subprocess,tarfile,io,json,datetime
P=pathlib.Path(__file__).resolve().parent
ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-661495']
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_attention'
# Fetch only receipts/logs, avoiding compiler outputs and cache blobs.
cmd="cd "+R+" && find runs shared_cache/TILELANG_CACHE_DIR -type f \\( -name '*.json' -o -name 'guard.log' -o -name '*.cpp' -o -name '*.cc' -o -name '*.cu' \\) -print0 | tar --null -T - -czf -"
data=subprocess.check_output(ssh+[cmd]);tarfile.open(fileobj=io.BytesIO(data),mode='r:gz').extractall(P,filter='data')
print('copied',len(data),'bytes')
