command -v uv curl wget
ls /opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/build_deps
find /root/.cache/uv -maxdepth 3 -iname '*build*' -o -iname '*pip*' -o -iname '*numpy*' 2>/dev/null | head -30
find /usr/local/lib /opt -maxdepth 5 -path '*/pip/__main__.py' 2>/dev/null | head -10
