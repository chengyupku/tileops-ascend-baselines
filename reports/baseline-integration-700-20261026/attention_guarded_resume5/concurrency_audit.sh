find /usr/local/Ascend/cann-9.1.0 -path '*opc*' -name '*.py' | head -25
grep -R -n -E 'TILINGKEY_PAR_COMPILE|MAX_COMPILE_CORE_NUMBER|MAX.*JOBS|ASCEND.*PARALLEL' /usr/local/Ascend/cann-9.1.0/python/site-packages /usr/local/Ascend/cann-9.1.0/compiler/python/site-packages 2>/dev/null | head -50
