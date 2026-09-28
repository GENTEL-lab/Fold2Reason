"""Aggregate complete, normalized General-10 base/adapted scores across seeds."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

BENCHMARKS = ("ftb_core", "spatialviz", "vsi", "graphqa_easy", "graphqa_hard",
              "bbh", "chembench", "chembench4k", "lab_bench", "scibench")


def aggregate(records):
    by_model = defaultdict(dict)
    for row in records:
        model, seed, benchmark = str(row["model"]), str(row["seed"]), row["benchmark"]
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown General-10 benchmark: {benchmark}")
        values = tuple(float(row[key]) for key in ("base", "adapted"))
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Scores must be finite fractions in [0, 1], not percentages.")
        key = seed, benchmark
        if key in by_model[model]:
            raise ValueError(f"Duplicate model/seed/benchmark: {model}/{seed}/{benchmark}")
        by_model[model][key] = values
    if not by_model:
        raise ValueError("No score records.")
    output = {}
    for model, rows in sorted(by_model.items()):
        seeds = sorted({s for s, _ in rows})
        if len(seeds) != 3:
            raise ValueError(f"{model}: expected three seeds, found {len(seeds)}")
        for seed in seeds:
            missing = set(BENCHMARKS) - {b for s,b in rows if s == seed}
            if missing:
                raise ValueError(f"{model}/{seed}: incomplete suite, missing {sorted(missing)}")
        result = {}
        for bench in BENCHMARKS:
            base = [rows[s,bench][0] for s in seeds]
            adapted = [rows[s,bench][1] for s in seeds]
            deltas = [100*(b-a) for a,b in zip(base,adapted)]
            result[bench] = {"base_percent":100*statistics.mean(base),
                             "adapted_percent":100*statistics.mean(adapted),
                             "delta_pp":statistics.mean(deltas),
                             "seed_sd_pp":statistics.stdev(deltas)}
        macro = [statistics.mean(100*(rows[s,b][1]-rows[s,b][0]) for b in BENCHMARKS) for s in seeds]
        output[model] = {"seeds":seeds,"benchmarks":result,"macro_delta_pp":statistics.mean(macro),
                         "macro_seed_sd_pp":statistics.stdev(macro),"macro_seed_deltas_pp":macro}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True,help="JSON array of model/seed/benchmark/base/adapted records")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=aggregate(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2)
        handle.write("\n")

