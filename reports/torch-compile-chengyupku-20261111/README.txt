Owner first-batch evidence review (NOT accepted leaderboard data)

3 cases measured as paired eager/compile with3warmup/15isolated1GiB-flushed windows each;90 downloaded windows independently replayed.9 further cases passed correctness admission;1 failed compile (All Cast float to bool). No new TileOPs speedup claim.

Start with COVERAGE.json, ENVIRONMENT.json, FLUSH.json, GUARD.json and OPEN_GAPS.json. WINDOWS.json contains every timing window and complete parsed tasks, including AICPU counts. ARCHIVES.json indexes retained private raw archives; no public raw channel yet.

runner.py is a parameterized draft, not the exact historical runner. It adds explicit paths/build identity and output-count assertions. Syntax and --help were checked without device imports; clean-checkout end-to-end replay remains pending. The actual frozen measurement core/parser were not modified. The provided script must run inside the existing host guard and share its lock; never create a substitute lock. Stage the frozen reports/torch-v010-20261106/source and tools under WORK/frozen; supply the frozen CLAIM, validated local flush directory/SHA/build label, existing guard lock and unique output directory. First run --windows1, then15 only after protocol review. Never publish private output paths or raw profiles.

Known review blockers are explicit: original wheel provenance, header identity detail, complete Add/Mul observation history, compiled artifact load-time binding, zero-contention proof, clean-checkout validation, archive access. Passing correctness and raw replay is not evidence for these missing properties. No website/registry/CI changes; do not register these rows as accepted.
