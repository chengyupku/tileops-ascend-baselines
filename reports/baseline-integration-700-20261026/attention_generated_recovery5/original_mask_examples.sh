S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
grep -R -n -E 'triu|tril|2048|mask_mode' "$S/attention/flash_attn/tests/assets" --include='*.py' | head -45
sed -n '247,315p' "$S/attention/flash_attn/torch_extension/flash_attn.py"
