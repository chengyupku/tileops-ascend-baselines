Torch.compile baseline candidate evidence — not accepted leaderboard data

Selected first-batch V2: 3 cases / 90 paired eager+compile windows. Subsequent batch: 8 measured candidates / 240 windows, plus Softmax checker-admission failure. Publisher independently replayed these windows and reviewed source/checker/guard evidence. Final acceptance and merge remain pending; all measurements retain alternate-environment status.

Read NEXT_BATCH_SUMMARY.json for the subsequent batch and NEXT_MANAGER_ASSOCIATION.json for observation limitations. Add BF16 has two unknown external manager-only identities and is not a qualified observation-complete baseline. All has a compile failure; Softmax accepts an incorrect all-zero output under its frozen checker, so no performance was measured. No fallback/reference/checker change was used.

runner.py is the parameterized audited runner. RUNNER_REPRODUCTION.txt describes the clean-source/fresh-cache pilot, which passed guard and independent two-window replay in the same identified environment. This is neither new-wheel environment reconstruction nor formal15-window performance. Original failed pilot attempts remain in private evidence.

AUDIT_V2/WINDOWS_V2 are selected first-batch detailed evidence; original V1 measurements are retained historical, not newly accepted. HEADERS_V2 is an include-root superset, not a compiler dependency trace. Compiled artifact load-time identity and original wheel provenance remain limited. Sampling cannot prove absolute absence of interference.

Raw evidence is exchanged only through the private Hub. Public files contain summaries, code, source/hash indexes and limitations. No website registration, baseline acceptance, or merge is implied. OPEN_GAPS.json is the current gap summary; older archive indexes describe their historical delivery state.
