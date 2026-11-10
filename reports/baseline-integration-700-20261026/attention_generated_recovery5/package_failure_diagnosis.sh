R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
for f in cmake_install.log makeself.log extract_package.log; do
 test ! -f "$R/$f" || tail -18 "$R/$f"
done
