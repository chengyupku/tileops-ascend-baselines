sed -n '185,270p' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/asc_op_compiler/ascendc_compile_base.py
grep -R -n -E 'cpu_count|ThreadPool|ProcessPool|MAX.*THREAD|BUILD.*JOBS' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/ | head -50
