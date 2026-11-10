S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
grep -n -E 'logfile|LOG|/var|/etc|INSTALL_PATH|targetdir=|\.bashrc|set_env|ldconfig|python|aicpu' "$S/build/scripts/install.sh" | head -90
