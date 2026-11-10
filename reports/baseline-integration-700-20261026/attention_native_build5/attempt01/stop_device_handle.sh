# Exact current attempt timeout pgid was observed in compiler_status; no other tasks touched.
ps -p 2676912 -o pid,pgid,args
if test -r /proc/2676912/cmdline && tr '\0' ' ' < /proc/2676912/cmdline | grep -q 'vendor_name=attention_native_build5'; then kill -TERM 2676912; fi
