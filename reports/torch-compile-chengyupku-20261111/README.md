# Ascend950 torch.compile evidence

This draft is not an accepted leaderboard submission. Final acceptance remains
pending. The report is a historical snapshot, not a live queue/status dashboard.

## Files retained for review

- `STATUS.json`: one historical reviewed snapshot (164 physical cases / 4920
  windows; 163 full-output candidates / 4890 windows), separated by protocol.
- `ENVIRONMENT.json` and `FLUSH.json`: measurement environment and flush provenance.
- `MEASUREMENTS.csv`: only the original RMSNorm/Mul/Add three-case sample,
  eager and compile (six rows); **not** the full 164-case data set or final results.
- `EVIDENCE.json`: full evidence archive location, size and SHA256.

The archive preserves all 50 original report files byte for byte, with a
`FILES.json` member hash manifest. Full measurement/admission details, per-window
records, failures, historical runner sources and raw-data indexes are in its
`report/` directory. It was downloaded and verified before the report was
simplified. Private agent-hub access is required; no evidence archive is needed
to run the pipeline. Historical files retain their original names and time scopes.

## Limits that remain

Representative cases do not establish all-shape coverage. DeltaNetChunkFwd's
partial output coverage remains excluded. The alternate environment and original
wheel provenance are not fully accepted. Driver sampling does not prove
continuous absence of competing users; historical missing samples remain gaps.
Failures, timeouts and protocol variants are retained in the archive, not counted
as successful cases. No new performance claims are introduced by this cleanup.

The optional pipeline implementation, tests, pilot results and limitations are
in [tools/compile_pipeline](../../tools/compile_pipeline/README.md). Its 274
experimental windows are separate from the historical baseline counts above.
Later task progress belongs in [agent-hub #3](https://github.com/yyttt6/agent-hub/issues/3).
