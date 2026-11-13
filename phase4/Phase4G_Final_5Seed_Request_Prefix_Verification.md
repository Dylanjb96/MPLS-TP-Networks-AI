# Phase 4G — Final offered-load request-prefix audit

Audit date: 13 November 2026. Audit scope: uploaded CSV request records for five demand seeds.

| Seed | 25 rows | 50 rows | 100 rows | First 25 match | First 50 match |
|---|---:|---:|---:|---|---|
| 2027 | 25 | 50 | 100 | Yes | Yes |
| 2028 | 25 | 50 | 100 | Yes | Yes |
| 2029 | 25 | 50 | 100 | Yes | Yes |
| 2030 | 25 | 50 | 100 | Yes | Yes |
| 2031 | 25 | 50 | 100 | Yes | Yes |

**Result: PASS.** All five demand seeds (2027–2031) use exactly matching nested request prefixes for the 25-, 50-, and 100-request experimental conditions. Comparisons included every CSV field: request ID, endpoints, required bandwidth, service class, priority weight, endpoint category.

**Limitations:** This check establishes workload-stream identity only. It does not re-audit the routing decisions, solver optimality, or physical-link capacity reservations. Duplicate CSV uploads were not treated as independent observations.

**Separate remaining methodology decision:** Archive the original one-stage weighted-tie MILP as version 1. Consider a versioned lexicographic optimization rule for version 2, with new tests and matched benchmark reruns. Supervisor approval of the simplified protection surrogate remains outstanding.
