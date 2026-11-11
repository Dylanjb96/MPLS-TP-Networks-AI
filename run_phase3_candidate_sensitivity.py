#Phase 3 candidate-cap pilot for the documented single-link-fault surrogate.

#Copy into the *root* of the existing MPLS-AI project and run:
  #python run_phase3_candidate_sensitivity.py

#This calls the project's existing Phase 3 CLI implementation; it neither
#changes phase3/model.py nor implements a new optimization algorithm.

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from phase3.cli import run
from phase3.model import ProtectionSimulator, SCHEME
from railway_tp.requests import generate_requests
from railway_tp.topology import TopologyConfig, make_topology


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_summary(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 6),
        "sd": round(statistics.stdev(values), 6) if len(values) > 1 else 0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 candidate limit sensitivity; NOT MSRP")
    parser.add_argument("--config", default="config/primary_60_dual.json")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--headroom", type=int, default=10)
    parser.add_argument("--candidate-limits", nargs="+", type=int, default=[1, 2, 4, 8, 12, 24])
    parser.add_argument("--out", default="outputs/phase3/candidate_limit_sensitivity")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing outputs of the same experiment")
    args = parser.parse_args()
    if args.requests < 1 or not args.seeds or not args.candidate_limits:
        parser.error("requests, seeds and candidate limits must be nonempty")
    if not 0 <= args.headroom < 100:
        parser.error("headroom must be between 0 and 99")
    if any(k < 1 for k in args.candidate_limits):
        parser.error("all candidate limits must be >= 1")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.candidate_limits)) != len(args.candidate_limits):
        parser.error("seeds and candidate limits must not have duplicates")

    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = TopologyConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    limits = sorted(args.candidate_limits)
    seeds = sorted(args.seeds)
    if not args.overwrite:
        collision = [str(out / f"n{args.requests}_seed{seed}_headroom{args.headroom}_k{k}")
                     for seed in seeds for k in limits
                     if (out / f"n{args.requests}_seed{seed}_headroom{args.headroom}_k{k}").exists()]
        if collision:
            parser.error("existing experiment directory; use a different --out or --overwrite: " + collision[0])
    out.mkdir(parents=True, exist_ok=True)

    # The inventory is independent of the sequential admission state. It is
    # critical for interpreting whether any candidate cap actually binds.
    G, rings = make_topology(cfg)
    probe = ProtectionSimulator(G, rings, headroom_percent=args.headroom)
    inventory: list[dict] = []
    for seed in seeds:
        for req in generate_requests(G, args.requests, seed=seed):
            plans = probe.candidates(req, max_candidates=1_000_000)
            inventory.append({
                "topology_seed": cfg.seed, "demand_seed": seed, "request_id": req.request_id,
                "source": req.source, "destination": req.destination,
                "endpoint_category": req.endpoint_category,
                "full_policy_valid_candidate_count": len(plans),
                "candidate_limits_that_truncate": ";".join(str(k) for k in limits if k < len(plans)),
            })
    write_csv(out / "candidate_inventory.csv", inventory, list(inventory[0]))

    records: list[dict] = []
    for seed in seeds:
        for k in limits:
            directory = out / f"n{args.requests}_seed{seed}_headroom{args.headroom}_k{k}"
            print(f"\nRUN: seed={seed}, candidates={k}, headroom={args.headroom}%")
            summary = run(cfg, directory, requests=args.requests, seed=seed,
                          headroom=args.headroom, max_candidates=k)
            reasons = summary["reason_counts"]
            admitted = summary["admitted_count"]
            row = {
                "nodes": cfg.nodes, "topology_seed": cfg.seed,
                "demand_seed": seed, "headroom_percent": args.headroom,
                "max_candidates": k, "requests": summary["request_count"],
                "admitted": admitted, "rejected": summary["rejected_count"],
                "protected_admission_ratio": summary["protected_admission_ratio"],
                "weighted_protected_acceptance_ratio": summary["weighted_protected_acceptance_ratio"],
                "normal_capacity_rejections": reasons.get("NORMAL_CAPACITY_EXCEEDED", 0),
                "postfault_capacity_rejections": reasons.get("POSTFAULT_CAPACITY_EXCEEDED", 0),
                "candidate_absent_count": summary["candidate_absent_count"],
                "peak_utilization": summary["scenario_ledger"]["peak_utilization"],
                "output_dir": str(directory),
            }
            assert admitted + summary["rejected_count"] == summary["request_count"]
            records.append(row)
    write_csv(out / "candidate_limit_runs.csv", records, list(records[0]))

    hist = Counter(x["full_policy_valid_candidate_count"] for x in inventory)
    maximum = max(hist)
    aggregate: list[dict] = []
    for k in limits:
        subset = [x for x in records if x["max_candidates"] == k]
        ratios = [float(x["protected_admission_ratio"]) for x in subset]
        weighteds = [float(x["weighted_protected_acceptance_ratio"]) for x in subset]
        regular = as_summary(ratios)
        weighted = as_summary(weighteds)
        aggregate.append({
            "max_candidates": k, "num_seeds": len(subset),
            "requests_per_seed": args.requests,
            "mean_protected_admission": regular["mean"],
            "sd_protected_admission": regular["sd"],
            "min_protected_admission": regular["min"],
            "max_protected_admission": regular["max"],
            "mean_weighted_admission": weighted["mean"],
            "sd_weighted_admission": weighted["sd"],
            "requests_with_more_than_k_candidates": sum(x["full_policy_valid_candidate_count"] > k for x in inventory),
            "candidate_cap_binds_any_request": int(any(x["full_policy_valid_candidate_count"] > k for x in inventory)),
        })
    write_csv(out / "candidate_limit_summary.csv", aggregate, list(aggregate[0]))

    indexed = {(r["demand_seed"], r["max_candidates"]): r for r in records}
    changes: list[dict] = []
    for seed in seeds:
        for lo, hi in zip(limits, limits[1:]):
            old = indexed[(seed, lo)]; new = indexed[(seed, hi)]
            changes.append({
                "demand_seed": seed, "lower_k": lo, "higher_k": hi,
                "lower_admitted": old["admitted"], "higher_admitted": new["admitted"],
                "delta_admitted": new["admitted"] - old["admitted"],
                "delta_admission_ratio": round(float(new["protected_admission_ratio"]) -
                                               float(old["protected_admission_ratio"]), 6),
                "delta_weighted_ratio": round(float(new["weighted_protected_acceptance_ratio"]) -
                                              float(old["weighted_protected_acceptance_ratio"]), 6),
            })
    write_csv(out / "candidate_limit_paired_changes.csv", changes, list(changes[0]))

    if maximum <= min(limits):
        diagnosis = "NO_TESTED_CAP_BINDS: all tested candidate caps expose every available plan"
    elif maximum <= max(limits):
        diagnosis = f"CAPS_AT_OR_ABOVE_{maximum}_ARE_NONBINDING: comparisons above that cap cannot test candidate truncation"
    else:
        diagnosis = "SOME_CAPS_BIND: examine inventory and paired results"
    provenance = {
        "scope": "Simulation-only ring-segment scenario surrogate, not RFC 8227 MSRP",
        "model": SCHEME, "config_path": str(cfg_path), "config": asdict(cfg),
        "headroom_percent": args.headroom, "requests_per_run": args.requests,
        "seeds": seeds, "candidate_limits": limits, "num_runs": len(records),
        "full_candidate_count_histogram": {str(k): v for k, v in sorted(hist.items())},
        "max_available_candidates": maximum,
        "diagnosis": diagnosis,
        "note": "Different k may choose different early plans; aggregate admissions need not be monotonic."
    }
    (out / "candidate_limit_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("\nFINISHED:", len(records), "candidate-limit experiments")
    print("Candidate inventory:", out / "candidate_inventory.csv")
    print("Aggregate summary:", out / "candidate_limit_summary.csv")
    print("Paired changes:", out / "candidate_limit_paired_changes.csv")
    print("DIAGNOSIS:", diagnosis)


if __name__ == "__main__":
    main()
