# Phase 5B extension — prospective experiment protocol (draft before new test)

**Research setting.** Sequential protected LSP admission on the fixed synthetic
60-node dual-gateway topology (seed 42), headroom 10%, k=2 candidates, 100
requests per demand stream, Phase3 protection abstraction
`RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`. Same admission oracle, no
future-request knowledge or voluntary rejection. Historical Phase5 held-out
seeds 31000–31029 have already been inspected; **do not use them again for
selection or as new confirmatory evidence**.

**Motivation.** The original validation-selected DDQN admitted 1020/3000 on the
previously used held-out test, while greedy/CSPF/MILP V2 each admitted
1051/3000. This extension tests whether *prospectively tuned* DDQN and
**masked discrete SAC** improve performance under otherwise unchanged
conditions. The original findings remain published/reported intact. The
extension is exploratory and independently evaluated on NEW demand seeds.

**Algorithms.** Greedy, CSPF-style, online MILP V2, random feasible, original
DDQN checkpoint (agent seed 7, best episode 20), validation-tuned DDQN,
masked discrete SAC with twin critics and fixed alpha. SAC is adapted to
categorical actions, not continuous SAC. Feasible candidates are determined
by the same Phase3 oracle and are capped at two arrangements. No model
re-routes prior admissions or predicts future requests.

**Training/validation.** Train demand seeds 10000–10079, validate on
20000–20009. Use scripted parameter grids and a predefined two-stage tuning
budget: screen three configurations with agent seed 7 for up to 80 episodes,
then confirm the strongest configuration with seeds 7, 17, 27, up to 160
episodes each. Selection per family: highest *mean* validation
priority-weighted admission over the three initializations; tie break by
mean ordinary admission. Within selected family/configuration, select best
individual checkpoint by validation weighted admission, then ordinary
admission. No use of previous or new held-out seeds. Explicitly report
repeated-validation selection bias and per-model training compute.

**Fresh test.** Only after model selection and code hashes are frozen:
demand seeds **41000–41029**, 30 whole-stream replicates, 100 requests each,
7 algorithms, topology seed 42, headroom10, k=2, matched exact request hashes.
Run once. Do not retrain or reselect to exploit these results. Compare to
original DDQN on THE SAME new test, not directly across old/new seed means.

**Primary endpoint.** Priority-weighted admission per stream. Secondary:
ordinary admission per stream, admission differences (paired per demand seed),
selection latency, decision-time overhead, final reservation-state digest,
path metrics, failure-state capacity violations, forced-rejection count,
training efficiency, and proportion of decisions with >=2 feasible candidates.

**Statistical reporting.** Paired seed-level mean differences with two-sided
95% Student-t confidence intervals, plus distribution of seed differences.
A zero-width CI is not proof of equivalence. Multiple model comparisons
are exploratory, so discuss multiplicity rather than claiming confirmatory
statistical significance without a prespecified correction. No inference
about variability over physical topology, actual railway networks, or
standards-level protection switching.

**Possible outcomes.** Improved RL admission is not guaranteed. A two-route
candidate cap and forced-reject rule limit the agent's control. Same-result or
negative-result extensions are to be reported without retuning after test.

**Research boundary.** This study does not implement RFC 8227 MSRP (including
ring-protection signaling and timing) and cannot substantiate railway
operational safety. Supervisor methodology approval is still pending.
