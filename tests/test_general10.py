import copy
import math
import unittest

from fold2reason.analysis.general10 import BENCHMARKS, aggregate


def records():
    return [{"model": "synthetic", "seed": seed, "benchmark": benchmark,
             "base": 0.4, "adapted": 0.4 + gain}
            for seed, gain in [(1, 0.01), (2, 0.02), (3, 0.03)] for benchmark in BENCHMARKS]


class General10Tests(unittest.TestCase):
    def test_paired_mean_and_seed_sd(self):
        result = aggregate(records())["synthetic"]
        self.assertAlmostEqual(result["macro_delta_pp"], 2.0)
        self.assertAlmostEqual(result["macro_seed_sd_pp"], 1.0)
        self.assertEqual(len(result["benchmarks"]), 10)
        for benchmark in result["benchmarks"].values():
            self.assertAlmostEqual(benchmark["base_percent"], 40.0)
            self.assertAlmostEqual(benchmark["adapted_percent"], 42.0)
            self.assertAlmostEqual(benchmark["seed_sd_pp"], 1.0)

    def test_reject_incomplete_suite(self):
        with self.assertRaisesRegex(ValueError, "incomplete suite"):
            aggregate(records()[:-1])

    def test_reject_missing_seed(self):
        with self.assertRaisesRegex(ValueError, "three seeds"):
            aggregate(records()[:20])

    def test_reject_duplicates(self):
        rows = records()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate(rows + [rows[0]])

    def test_reject_invalid_values(self):
        for value in (math.nan, math.inf, -0.1, 42):
            with self.subTest(value=value):
                rows = records()
                rows[0]["adapted"] = value
                with self.assertRaises(ValueError):
                    aggregate(rows)

    def test_reject_unknown_benchmark_and_empty_input(self):
        rows = records()
        rows[0]["benchmark"] = "not-in-general10"
        for value in (rows, []):
            with self.assertRaises(ValueError):
                aggregate(value)

    def test_model_groups_are_independent(self):
        rows = records()
        extra = copy.deepcopy(rows)
        for row in extra:
            row["model"] = "another-model"
            row["adapted"] = row["base"]
        result = aggregate(rows + extra)
        self.assertAlmostEqual(result["another-model"]["macro_delta_pp"], 0)
        self.assertAlmostEqual(result["synthetic"]["macro_delta_pp"], 2)


if __name__ == "__main__":
    unittest.main()
