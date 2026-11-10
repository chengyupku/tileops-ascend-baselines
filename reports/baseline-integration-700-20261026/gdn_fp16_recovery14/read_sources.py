from pathlib import Path
import json,subprocess
R=Path('/opt/tileops-all-pairs/workspace');P=R/'.cache/all-pairs/resume22_server2_tilekernels_adaptive_norm02/sdk_parts/cann-950-ops-transformer';root=P/'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_kda_fwd'
x={str(f.relative_to(root)):f.read_text() for f in list(root.glob('*.h'))+list(root.glob('*.cpp'))+list((root/'arch35').glob('*.h'))};x['official_header.h']=(P/'aarch64-linux/include/aclnnop/level2/aclnn_chunk_kda_fwd.h').read_text();x['dynamic.py']=(P/'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/dynamic/chunk_kda_fwd.py').read_text();print(json.dumps(x))
