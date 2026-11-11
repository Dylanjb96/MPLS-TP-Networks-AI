#Phase 3 end-of-phase reproducibility/sensitivity runner.

#Copy this file into the root of your existing MPLS-AI project, then run:
  #python run_phase3_multiseed_validation.py

#Every (request-count, demand-seed, headroom) combination uses a unique
#output directory, protecting your previous experiment files. The selected
#protection model is the Phase 3 *surrogate*, not full MPLS-TP MSRP.

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from railway_tp.topology import TopologyConfig
from phase3.cli import run


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 multi-seed robustness pilot")
    parser.add_argument("--config", default="config/primary_60_dual.json")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--headrooms", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--out", default="outputs/phase3/multiseed_validation")
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()
    if args.requests < 1 or not args.seeds or not args.headrooms or args.max_candidates < 1:
        parser.error("requests, candidate limit, seeds and headrooms must be valid non-empty inputs")
    if any(h < 0 or h >= 100 for h in args.headrooms):
        parser.error("headrooms must be integers in 0..99")

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = TopologyConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for seed in args.seeds:
        for h in args.headrooms:
            # Include request length in directory to prevent collision with
            # previously generated files using the same topology/seed/headroom.
            scenario_dir = out_root / f"n{args.requests}_seed{seed}_headroom{h}_k{args.max_candidates}"
            summary = run(cfg, scenario_dir, requests=args.requests,
                          seed=seed, headroom=h, max_candidates=args.max_candidates)
            reasons = summary["reason_counts"]
            records.append({
                "nodes": cfg.nodes,
                "gateway_mode": cfg.gateway_mode,
                "topology_seed": cfg.seed,
                "demand_seed": seed,
                "requests": summary["request_count"],
                "headroom_percent": h,
                "max_candidates": args.max_candidates,
                "admitted": summary["admitted_count"],
                "rejected": summary["rejected_count"],
                "protected_admission_ratio": summary["protected_admission_ratio"],
                "weighted_protected_acceptance_ratio": summary["weighted_protected_acceptance_ratio"],
                "normal_capacity_rejections": reasons.get("NORMAL_CAPACITY_EXCEEDED", 0),
                "postfault_capacity_rejections": reasons.get("POSTFAULT_CAPACITY_EXCEEDED", 0),
                "candidate_absent_count": summary["candidate_absent_count"],
                "peak_utilization": summary["scenario_ledger"]["peak_utilization"],
                "output_dir": str(scenario_dir),
            })

    fields = list(records[0])
    write_csv(out_root / "multiseed_runs.csv", records, fields)
    grouped: list[dict] = []
    for h in sorted(set(args.headrooms)):
        subset = [r for r in records if r["headroom_percent"] == h]
        raw = [r["protected_admission_ratio"] for r in subset]
        weighted = [r["weighted_protected_acceptance_ratio"] for r in subset]
        grouped.append({
            "headroom_percent": h,
            "num_seeds": len(raw),
            "requests_per_seed": args.requests,
            "mean_protected_admission": round(statistics.mean(raw), 6),
            "sd_protected_admission": round(statistics.stdev(raw), 6) if len(raw) > 1 else "",
            "min_protected_admission": min(raw),
            "max_protected_admission": max(raw),
            "mean_weighted_admission": round(statistics.mean(weighted), 6),
            "sd_weighted_admission": round(statistics.stdev(weighted), 6) if len(weighted) > 1 else "",
        })
    write_csv(out_root / "multiseed_summary.csv", grouped, list(grouped[0]))
    provenance = {
        "scope": "synthetic ring-segment robust-admission surrogate, NOT actual MPLS-TP MSRP",
        "configuration": str(config_path),
        "nodes": cfg.nodes,
        "topology_seed": cfg.seed,
        "demand_seeds": args.seeds,
        "headrooms": args.headrooms,
        "requests_per_run": args.requests,
        "max_candidates": args.max_candidates,
        "runs": len(records),
        "outputs": ["multiseed_runs.csv", "multiseed_summary.csv"],
        "interpretation": "Pilot descriptive analysis; seeds held identical across headroom settings. No operator claims.",
    }
    (out_root / "multiseed_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("\nFINISHED:", len(records), "scenarios")
    print("Detailed:", out_root / "multiseed_runs.csv")
    print("Aggregate:", out_root / "multiseed_summary.csv")
    print("Note: this pilot does not prove protection-protocol conformance.")


if __name__ == "__main__":
    main()