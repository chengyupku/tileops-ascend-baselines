S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
find "$S/attention/flash_attn" -maxdepth 3 -type f -name '*.py' | head -15
grep -R -n -E '2048|triu|tril|win_left' "$S/attention/flash_attn/examples" --include='*.py' | head -35
