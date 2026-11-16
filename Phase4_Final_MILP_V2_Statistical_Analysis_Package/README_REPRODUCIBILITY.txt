Phase 4 — MILP V2 Final Statistical Analysis (reproducibility notes)

Data: Greedy, CSPF-style, MILP V2, topology seed 42, two-candidate cap.
Included: 5 experiment families / 45 paired seed-conditions / 135 algorithm runs.
MILP V1 is not merged with MILP V2 in these results.

To reproduce locally (from the MPLS-AI project directory):

  python run_phase4_v2_final_statistics.py --root outputs\phase4 --out outputs\phase4_v2_statistics

Expected source files relative to --root:
  v2_final_60_headroom\phase4_runs.csv
  v2_final_100\phase4_runs.csv
  v2_final_150\phase4_runs.csv
  v2_final_load_25\phase4_runs.csv
  v2_final_load_100\phase4_runs.csv

The script uses Python standard library and SciPy (the project's MILP dependency).
The statistics are t-based seed-level intervals conditioned on the fixed network
realizations. They are not proof of algorithmic equivalence.
The 131 passing tests were reported by the user; this statistical program
separately checks the benchmark CSVs and does not rerun simulation tests.

Scope: RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP; no MSRP protocol signaling,
node or simultaneous failure testing, or protection switching timing.
Supervisor methodology/protection-model approval remains outstanding.

Source CSV SHA-256 checksums:
  60node_headroom: 36c5688b229e75bd208e936dbc46f9e1f2190eb7e1fa8c29464da5b800dc12e7  (phase4_runs(20261116-083701).csv)
  150node_scalability: e65aa6c0ac3093d39688d7909b879d1eae484767ed985988d3bcec9d51107f25  (phase4_runs(20261116-083718).csv)
  100node_scalability: 522a663034482ebf07b18eaa81f4926f3fda9008994afb3ecdcc8c2fd0ad8146  (phase4_runs(20261116-085052).csv)
  25request_load: 792c708cd6b71dcfa4def0fcb4cfbb978f438acce4db86b2faf1fad8be29849f  (phase4_runs(20261116-085125).csv)
  100request_load: e6b0719d8f9cae0ac94f731867025b69461aa8c48a3031e5dc00bd9ab8f7d6d0  (phase4_runs(20261116-085538).csv)
