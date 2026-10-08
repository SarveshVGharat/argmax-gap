#!/usr/bin/env python3
"""Compare available report point estimates and counts with the paper results.

Top-k and delta tolerances are percentage points; NLL/MRR/NDCG are raw units.
Bootstrap interval endpoints are recorded in the reference but are not checked:
their Monte Carlo variation is independent of point-estimate reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "maia3": "maia3base", "allie": "alliepolicyonlybase", "alliebase": "alliepolicyonlybase",
    "diagnosticoracle": "oraclediagnostic", "maxprobability": "rawmaxpselector",
    "minentropy": "minentropyselector", "maxmargin": "maxmarginselector",
    "crossmodellinear": "crossmodeltrainedselector", "crossmodel": "crossmodeltrainedselector",
    "maia3shortlistlinear": "maia3shortlistselector", "maia3shortlist": "maia3shortlistselector",
    "allieshortlistlinear": "allieshortlistselector", "allieshortlist": "allieshortlistselector",
    "rank2gate": "rank2correctiongate", "rank2": "rank2correctiongate",
    "maia3refined": "maia3trustregionrefiner", "maia3refiner": "maia3trustregionrefiner",
    "allierefined": "allietrustregionrefiner", "allierefiner": "allietrustregionrefiner",
    "convex": "convexensemblealpha070", "convexensemble": "convexensemblealpha070",
    "geometric": "geometricensemblealpha070", "geometricensemble": "geometricensemblealpha070",
    "maia3selftop10linear": "maia3shortlistselector",
    "allieselftop5linear": "allieshortlistselector",
    "maia3rank2gate": "rank2correctiongate",
    "maia3selftop10xgboost": "maia3shortlistxgboost",
    "allieselftop5xgboost": "allieshortlistxgboost",
    "maia3timebucketcalibration": "maia3calibrated",
    "allietimebucketcalibration": "alliecalibrated",
    "ensembleconvex": "convexensemblealpha070",
    "ensemblegeometric": "geometricensemblealpha070",
}


def canonical(value: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    return ALIASES.get(key, key)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_number(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    number = float(value)
    return number


def compare(report: Path, reference: Path, require_all: bool = False,
            percent_tolerance: float = 0.001, nll_tolerance: float = 1e-5) -> dict:
    metrics_path, pairs_path = report / "metrics.csv", report / "paired_top1.csv"
    if not metrics_path.exists() and not pairs_path.exists():
        raise ValueError(f"No metrics.csv or paired_top1.csv in {report}")
    metric_rows = read_csv(metrics_path) if metrics_path.exists() else []
    pair_rows = read_csv(pairs_path) if pairs_path.exists() else []
    metrics = {canonical(row["method"]): row for row in metric_rows}
    pairs = {(canonical(row["method"]), canonical(row.get("reference") or "maia3")): row for row in pair_rows}
    if len(metrics) != len(metric_rows) or len(pairs) != len(pair_rows):
        raise ValueError("Duplicate method/reference entries in report")
    checks, missing = [], []
    base_top1 = finite_number(metrics.get("maia3base", {}).get("Top1"))

    def check(method, ref, field, expected, actual, tolerance):
        if actual is None:
            missing.append({"method": method, "reference": ref, "metric": field})
            return
        passed = math.isfinite(actual) and abs(actual - expected) <= tolerance
        checks.append({"method": method, "reference": ref, "metric": field,
                       "expected": expected, "actual": actual,
                       "tolerance": tolerance, "passed": passed})

    for row in read_csv(reference):
        name, ref = row["method"], row["reference"]
        method_key, ref_key = canonical(name), canonical(ref)
        metric, paired = metrics.get(method_key, {}), pairs.get((method_key, ref_key), {})
        if not metric and not paired:
            missing.append({"method": name, "reference": ref, "metric": "all"})
            continue
        check(name, ref, "rows", float(row["rows"]),
              finite_number(metric.get("rows") or paired.get("rows")), 0)
        if metric and paired:
            check(name, ref, "paired_rows", float(row["rows"]), finite_number(paired.get("rows")), 0)
        for field, actual_field in [(f"top{k}_percent", f"Top{k}") for k in [1, 3, 5, 10, 20]]:
            if not row[field]:
                continue
            actual = finite_number(metric.get(actual_field) or paired.get(actual_field))
            if actual is None and field == "top1_percent" and paired and base_top1 is not None and ref_key == "maia3base":
                delta = finite_number(paired.get("delta_pp"))
                if delta is not None:
                    actual = base_top1 + delta
            check(name, ref, field, float(row[field]), actual, percent_tolerance)
        for field, actual_field in [("nll", "NLL"), ("mrr", "MRR"), ("ndcg5", "NDCG@5")]:
            if row[field]:
                check(name, ref, field, float(row[field]), finite_number(metric.get(actual_field)),
                      nll_tolerance if field == "nll" else 1e-6)
        for field in ["rescues", "breaks", "delta_top1_pp"]:
            if row[field]:
                actual_field = "delta_pp" if field == "delta_top1_pp" else field
                check(name, ref, field, float(row[field]), finite_number(paired.get(actual_field)),
                      percent_tolerance if field == "delta_top1_pp" else 0)
        for field, actual_field in [("ece_all_percent", "ECE"), ("ece_disagreement_percent", "disagreement_ECE")]:
            if row[field]:
                actual = finite_number(metric.get(actual_field))
                check(name, ref, field, float(row[field]), 100 * actual if actual is not None else None, percent_tolerance)
    failures = [check for check in checks if not check["passed"]]
    return {"passed": bool(checks) and not failures and not (require_all and missing),
            "checks": len(checks), "failures": failures, "missing": missing,
            "require_all": require_all,
            "checked_fields": "rows, Top-k, NLL, MRR, NDCG@5, rescues, breaks, Top1 deltas, ECE",
            "unchecked_fields": "stochastic confidence intervals, p-values, learned thresholds and switch rates"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "reference" / "paper_results.csv")
    parser.add_argument("--require-all", action="store_true", help="Fail if any selected method or checkable metric is missing")
    parser.add_argument("--percent-tolerance", type=float, default=0.001)
    parser.add_argument("--nll-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, help="Optional JSON comparison report")
    args = parser.parse_args()
    if args.percent_tolerance < 0 or args.nll_tolerance < 0:
        parser.error("Tolerances must be nonnegative")
    result = compare(args.report, args.reference, args.require_all, args.percent_tolerance, args.nll_tolerance)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['checks']} checks, "
          f"{len(result['failures'])} mismatches, {len(result['missing'])} unavailable method/metric entries")
    for failure in result["failures"]:
        print(f"  {failure['method']} / {failure['metric']}: {failure['actual']} (expected {failure['expected']})")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
