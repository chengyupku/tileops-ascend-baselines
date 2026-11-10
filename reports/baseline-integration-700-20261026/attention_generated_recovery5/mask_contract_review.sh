S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
grep -n -E '2048|triu|tril|win_left|attn_mask' "$S/attention/flash_attn/tests/pytests/test_flash_attn.py" | head -40
grep -n -E '^def |mask_mode|win_left|attn_mask' "$S/attention/flash_attn/torch_extension/flash_attn.py" | head -65
