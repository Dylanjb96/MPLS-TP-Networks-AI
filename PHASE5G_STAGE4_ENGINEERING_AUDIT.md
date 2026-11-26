# Phase 5G Stage 4 source-handoff and evaluator engineering audit (pre-final)

## Source handoff inspected

- The uploaded `Phase5G_Stage4_Source_Handoff.zip` contained **50** members and passed ZIP CRC validation.
- The handoff manifest listed **49 source/protocol file SHA-256 fingerprints**; **49/49 matched** the supplied file bytes.
- The source handoff declared **no model weights, training records or held-out data**, so these were not inferred to be part of the upload.
- The exact Stage 3 protocol SHA `6edad86fb2f49328784c5e80ef60d967413f6bf4f30f64debf57ad3381ce1d43` and frozen manifest SHA `cde024cf52dbb56fc55c6380d5d0dd9c00fc731a2eac627276b8b88c03b8f04b` were preserved.
- The previously uploaded `Phase5G_Stage3_Freeze_Evidence.zip` was supplied separately and used *only for development-source and checkpoint verification*. **No held-out Phase 5G topology/stream seeds were generated**.

## Implemented and locally verified

- A new `phase5g_stage4/` implementation uses only preexisting Stage 3-locked source files and frozen checkpoint weights. Stage 4 does not modify any previous phase modules.
- Preflight passed and validated **all 15 Stage 3-locked source hashes**, all frozen Stage 3 evidence file hashes, selected model identities and the protocol hash.
- **22 new tests passed**, checking exact stress formulas, held-out isolation, K2/6 neural masking, six-policy evaluation and independent oracle agreement, immutable cohort payloads, strict resumability and source lock, symbolic 2,880-row statistics fixture, t intervals and separate Holm families.
- **Eight complete development-only cells** (hierarchical/ladder × base/bandwidth stress/short-hold/K2, 60 nodes, topology 501, stream 74000) ran all **six** policies per cell: 8 cohorts and **48 policy simulations**. All independent replay checks passed. All eight result files were saved atomically, read back, verified, and resumed with unchanged hashes.
- An independent **500-node validation prefix regression** reused already-seen Stage 2 topology 701 / stream 76000, generated only the first **80** candidates from its recorded 2,000-request stream, and exactly matched all 80 chosen actions, feasibility counts, admits, and live ledger-state digests for **both** frozen GNN seed27 and frozen MLP seed17 against their Stage 3 validation traces. The separate complete 2,000-action replay remains a Windows requirement.
- Separate final-run CLI without `stage4_AUDIT_RELEASE.json` rejected access **before** building reserved final topologies or creating a final output directory; the guard was exercised locally.
- True path-level **running maximum scenario utilization** is captured by a second-implementation oracle replay using only the two subsets of scenario/arc reservations that can increase after an admitted service. This avoids incorrectly reporting the final-state utilization as the maximum achieved during the episode.
- The source-predeclared dynamic traffic mode fix is preserved. In K2 cells only two route candidates exist, while the GNN, new MLP and older PPO receive a fixed **six-slot feature mask** with four permanently invalid slots.
- The bandwidth shock is **ceil(1.5 × request bandwidth)** (integer Mbps); link capacities are unchanged. The short-hold shock is `expiry = arrival + max(1, floor(original_duration/2))`. These are **pre-heldout implementation details** matching the earlier Stage 1 stress generator.

## Pending Windows/integration gates (not claimed passed)

- The developer environment has not completed the **full 500-node development scale test** or **2,000-action selected Stage 2 validation replay**. Both must be run on the Windows workstation before an independent Stage 4 release.
- The user's full previous 316-test suite is not reproduced in this source-only handoff. **22 Stage 4 tests passed** in the independent source sandbox; Windows must check the complete existing project plus new tests.
- No real Phase 5G final seed 801–820 / 77000–77002 has been generated or evaluated by this package.
- The preliminary study implementation is **not yet an independently approved final runner**; its `final` path requires a separate hash-specific audit release after examination of development evidence.
- Full six-policy, 500-node execution time and storage are not extrapolated from the 60-node smoke; these must be assessed using the Windows 500-node development evidence.

## Locked statistical design

The full final report builder checks **2,880 unique policy outputs** (480 matched cohorts × six policies), cohort hashes and outcome sanity. Two primary hierarchical/base comparisons are independently topology-clustered and jointly Holm adjusted: **GNN–MLP** and **GNN–scenario-minimax**. Eight secondary comparisons receive a separate Holm family adjustment; stress and K2 sensitivity are labeled **exploratory**. Twenty topology clusters, each averaging three stream differences, are the statistical replication units. A fixed synthetic fixture exercises incomplete/duplicate/unmatched-grid rejection and Holm/t inference before final data access.

**No conclusion about GNN superiority or out-of-family generalization is possible from Stage 4 development checks.**
