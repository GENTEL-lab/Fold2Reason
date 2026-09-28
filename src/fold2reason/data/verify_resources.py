"""Verify the full General-10 input counts with the actual evaluation loaders."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--text-only", action="store_true", help="Check seven text datasets only")
    args = parser.parse_args()
    from fold2reason.evaluation import general_text as text
    external = args.benchmark_root / "data/external"
    expected = {"bbh":6511, "chembench":2148, "chembench4k":4009,
                "graphqa_easy":21600, "graphqa_hard":21600, "lab_bench":1967, "scibench":580}
    output = {}
    for name, count in expected.items():
        loader = text.load_graphqa if name.startswith("graphqa_") else getattr(text, "load_" + name)
        rows, _ = loader(external / name, name) if name.startswith("graphqa_") else loader(external / name)
        if len(rows) != count or len({row["id"] for row in rows}) != count:
            raise ValueError(f"{name}: expected {count} unique evaluation rows, found {len(rows)}")
        output[name] = len(rows)
    if not args.text_only:
        from fold2reason.evaluation.multimodal import load_spatialviz_rows, load_vsi_rows
        output["spatialviz"] = len(load_spatialviz_rows(external / "spatialviz_bench"))
        output["vsi"] = len(load_vsi_rows(external / "vsi_bench"))
        if output["vsi"] != 5130:
            raise ValueError(f"VSI expected 5130 rows, found {output['vsi']}")
        from fold2reason.evaluation.ftb import load_benchmark
        rows = load_benchmark(args.benchmark_root / "data/generated/ftb_core/v1", examples_per_task=500)
        if len(rows) != 12000 or len({row["id"] for row in rows}) != 12000:
            raise ValueError("FTB-Core expected 12000 unique test + test_ood examples")
        output["ftb_core"] = len(rows)
    print(json.dumps({"counts": output, "total": sum(output.values()),
                      "media_paths_verified": not args.text_only}, indent=2))
