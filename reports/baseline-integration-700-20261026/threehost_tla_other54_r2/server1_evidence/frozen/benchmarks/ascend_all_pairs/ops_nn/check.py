"""Use the unchanged classic checker with the isolated provider dictionary."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT.parent)]
spec = importlib.util.spec_from_file_location("classic_checker", ROOT / "scripts/ascend_classic_check.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.require_guard()
# Patch this process only; retain all checker oracle and input-protection rules.
from benchmarks.ops import bench_ascend_classics
from benchmarks.ascend_all_pairs.ops_nn import PROVIDERS
bench_ascend_classics.candidate_factories = lambda case: {"ops-nn": lambda: PROVIDERS[case._call.signature.name](case)}
raise SystemExit(checker.main())
