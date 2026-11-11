Torch compile Ascend950 supplementary measurements

Current reviewed scope: see SUMMARY_CURRENT.json. Physical replay108 cases/3240 formal-length windows;107/3210 full-output candidates after excluding DeltaNetChunkFwd partial coverage. All await final acceptance. These are representative cases, not all shapes. No merge is claimed.

Original full-FD34/1020 and reservation-driver-v1 74/2220 remain separate. Driver-only sampling does not establish continuous absence of contention or full FD visibility. Independent GQABwd trace_autograd_ops=True1/30 is included in the latter and separately labeled;20/30 wrapper intervals have no driver sample. Pilots remain separate.

SUMMARY_CURRENT.json and OPEN_GAPS.json are authoritative for current conclusions. ATTEMPT_INDEX_CURRENT.json links individual available admission records to private Release members; BATCH_OUTCOMES_CURRENT.json preserves CPU blocks, timeouts and all batch exits. PRIVATE_RAW_INDEX_CURRENT.json identifies available uploaded assets by exact name/size/local SHA. Reviewers need private Hub access. Earlier summaries and receipts are historical, not current cumulative counts.

A successful reference1/1 check does not imply the entire operator contract is covered. DeltaNetChunkFwd lacks5 intermediates and is excluded. Nullable MoEShared None is a valid structural output, not blanket permission for1/2 numeric coverage; its scoped eager validation passed but compile failed.

Latest MLAVarlenBwd prepared-payload attempt verified pinned payload/proof and passed eager, then failed Python range specialization. It produced zero timing windows. Prior timeouts remain separate. Payload contents were not independently replayed by the publisher.
