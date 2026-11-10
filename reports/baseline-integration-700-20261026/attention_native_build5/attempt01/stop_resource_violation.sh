ps -eo pid,ppid,args | grep -E 'make -j[0-9]+ prepare_build|timeout --signal=TERM --kill-after=10 1800 bash build.sh' | grep -v grep
# Stop only this attempt's observed timeout supervisor; it forwards TERM to its own group.
if test -r /proc/2668466/cmdline && tr '\0' ' ' < /proc/2668466/cmdline | grep -q 'vendor_name=attention_native_build5'; then kill -TERM 2668466; fi
