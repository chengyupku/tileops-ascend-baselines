set -e
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01
# Resource-only launcher: upstream prepare.sh independently requests 2*host CPUs.
# Keep original build/kernel/compiler files unchanged; cap GNU make jobs in PATH.
cat > "$B/stage/bin/make" <<'PY'
#!/opt/tileops-ci/runtime/bin/python3
import os,sys
args=[];skip=False
for a in sys.argv[1:]:
 if skip: skip=False; continue
 if a in ('-j','--jobs'): skip=True; continue
 if a.startswith('-j') or a.startswith('--jobs='): continue
 args.append(a)
os.execv('/usr/bin/make',['/usr/bin/make',*args,'-j8'])
PY
chmod +x "$B/stage/bin/make"
tail -25 "$B/native_build.log"
