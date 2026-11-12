# Cross-case AscendC compilation pipeline

Experimental opt-in runner. Two-case device smoke and three-case
serial/pipeline/serial controls passed on 950DT 32-core hardware. These records
remain experimental, not published baseline records or environment acceptance.

`staged_backend.py` defers the first AscendC kernel launch after graph generation.
Workers compile that case's actual generated sources into isolated artifacts.
The parent verifies binary hashes and loads those same artifacts when the case
becomes current. Device queries, graph preparation, autotuning, and execution
remain in the parent. The lookahead is bounded to one case; two CPU workers may
compile its kernels while the preceding case is measured on the NPU.

Run `run_validation.py` inside the existing device reservation and guard, with
`--root` pointing to the frozen runner workspace and `--claim` to its case plan.
Use `--mode serial` or `--mode pipeline`, `--indices 0,4,6`, and separate
`--output` directories. The default is 15 retained windows per variant; `--windows
1` is only a smoke check. Serial/pipeline/serial controls use independent cold
caches and the same frozen inputs, checker, flush, and isolated-window timer.

Admission requires complete output checks, negative controls, unchanged inputs,
no graph compilation during measurement, raw-profile replay, and independent
guard review. Artifact hashes, worker PID/build intervals, consumption counts,
and parent measurement intervals permit auditing real reuse and overlap.
Process-tree device-FD samples run every 50 ms; they are observations, not proof
that no transient device access occurred. Review guard driver samples as well.

Graphs without deferred AscendC kernels fail this validation rather than being
reported as successful overlap. Backend private interfaces are version-sensitive.
Do not apply this runner to arbitrary mutating/stateful workloads without a
separate input/reset protocol review. There is no automatic retry after a failed
case or guard rejection.

CPU unit checks: `python3 -m unittest discover -s tools/compile_pipeline`.


## Device validation (2026-11-12)

Frozen Mean/Amax/Amin `3d-multidim-float16`, isolated cold caches, two CPU
workers, unchanged isolated-window timer and reservation-driver-v1-diagnostic04:

| Arm | End-to-end seconds | Actual lookahead overlap seconds |
| --- | ---: | ---: |
| Serial A | 202.504 | 0 |
| Pipeline | 143.171 | 54.038 |
| Serial B | 199.512 | 0 |

The observed wall-time reduction against the serial mean is 28.77%. All six
compiled kernels per arm were consumed by the parent using the worker artifact
paths and verified binary hashes. Full-output checks and zero-output negative
controls passed, with no graph recompilation during measurement. Guard exits
were zero. All 270 control windows plus four smoke windows were independently
replayed with the frozen full32 parser; trace hashes, tasks, samples and means
matched. Worker PIDs did not appear in retained driver process samples; FD
sampling observed management nodes only.

Device-mean offsets against the serial-control mean range from -0.421% to
+1.633%. Some exceed their particular pair's control spread (Amax compiled
+0.427% versus 0.063%; Amin eager +1.633% versus 1.376%). Thus this demonstrates
functional integration and bounded throughput benefit, not statistical timing
equivalence or a universal speedup. Earlier-arm evidence was compressed during
the pipeline run; this additional host activity is another limitation of this
small pilot's end-to-end comparison. Keep bulk formal measurements on their
existing protocol pending broader validation/review.

Private raw archive SHA256:
`27ad49df82c4cebf549b51e2df900c3ee1cdd6deb7861af33fd43df4f464e5eb`.


## Case lifetime and observation failures

Completed cases release their device input and compiled-callable references after
measurement, including failure paths. Per-case measurement outputs and closures
are scoped to that case. The NPU allocator may retain reserved cache memory;
reference release is not a claim that reserved memory immediately drops.

Worker process/FD observation permission errors, malformed process data and
unexpected monitor-thread failures invalidate the experiment and are recorded in
`worker_observation_errors`. Normal ENOENT/ESRCH process-exit and FD-close races
are counted separately. Checks run before and after each timing window and after
worker shutdown, so a late failure cannot produce a successful final report.

Ten CPU tests cover the artifact path, repeated case reference release, exception
cleanup, permission failures, process-exit races and monitor-thread failure.
The device pilot above predates these lifecycle/observation fixes; it is not a
fresh device validation of the latest revision.
