grep -nE 'ccache|makefile|make_file|tmp_aiv|exists' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/asc_op_compiler/ascendc_compile_utils.py | head -60
grep -nE 'ccache|makefile|make_file|tmp_aiv|exists' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/asc_op_compiler/ascendc_compile_gen_code.py | head -40
command -v ccache || true
