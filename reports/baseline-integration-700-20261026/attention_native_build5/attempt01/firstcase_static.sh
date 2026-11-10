S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01/stage/source
sed -n '110,150p' "$S/attention/flash_attn/torch_extension/flash_attn.py"
sed -n '1,135p' "$S/attention/flash_attn_metadata/torch_extension/flash_attn_metadata.py"
grep -nE '^def |mask|win_|layout_' "$S/attention/flash_attn_metadata/torch_extension/flash_attn_metadata.py" | tail -40
ps -eo pid,ppid,etimes,comm,args | grep -E 'attention_native_build5|git clone|json-populate' | grep -v grep | tail -12
