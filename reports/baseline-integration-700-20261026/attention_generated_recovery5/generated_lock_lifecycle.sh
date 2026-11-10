grep -R -n -E 'kernel_meta.lock|compile_stop.flag|kernel_meta_' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/asc_op_compiler --include='*.py' | head -35
