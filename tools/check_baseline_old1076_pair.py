"""Original full checker for classification, without performance-cache claims."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reports/baseline-cache-20261027/python'), str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src'), str(ROOT)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    task=json.loads(args.task.read_text())
    library,op,case_id=task['key']
    receipt=args.out/'correctness.json'
    result={'task_id':task['id'],'key':task['key'],'started_at':time.time(),
            'status':'correctness_not_verified','performance_valid':False,
            'cache_eligible':False,'purpose':'full correctness classification only'}
    try:
        path=ROOT/'TileOPs/scripts/cache_frozen_pair_check.py'
        spec=importlib.util.spec_from_file_location('old1076_original_checker',path)
        checker=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        checker.require_guard()
        sys.argv=[str(path),'--op',op,'--case-id',case_id,'--library',library,'--output',str(receipt)]
        if op in ('GDNFwdOp','KDAFwdOp'):
            sys.argv.append('--cpu-fp64-reference')
        rc=checker.main()
        check=json.loads(receipt.read_text())
        result.update(correctness_receipt=str(receipt),correctness_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        if rc or check.get('status')!='verified':
            result['error']=check.get('error',check.get('reason'))
            return 1
        snap=check['case_snapshot']
        assert snap['parameters']==task['case']['params'] and snap['declared_tensors']==task['case']['tensors']
        assert check['inputs_preserved'] and not check['provider_source_changed'] and not check['oracle_source_changed']
        verification=check['verification']
        assert verification['checked_outputs']==verification['total_outputs'] and verification['checked_outputs']>0 and not verification.get('unchecked_reason')
        for field in ('source_sha256','provider_source_sha256','oracle_source_sha256'):
            for name,digest in check.get(field,{}).items():
                source=Path(name)
                if not source.is_absolute():source=ROOT/source
                assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==digest, name
        result['status']='correctness_verified_uncached'
        return 0
    except Exception as exc:
        result.update(error=str(exc),traceback=traceback.format_exc())
        return 1
    finally:
        result['finished_at']=time.time()
        temporary=args.out/'measurement.tmp'
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        temporary.replace(args.out/'measurement.json')

if __name__=='__main__':
    raise SystemExit(main())
