import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fold2reason import cli

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_top_level_help(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(["--help"])
        self.assertIn("general10-text", output.getvalue())

    def test_config_aliases_and_overrides(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(["train", "full", "--config", str(ROOT / "configs/train/full.toml"),
                      "--dry-run", "--seed", "123"])
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["module"], "fold2reason.training.workspace")
        self.assertIn("--loss-relation", payload["arguments"])
        self.assertEqual(payload["arguments"][-2:], ["--seed", "123"])

    def test_reject_unknown_config_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text("[typo]\nseed = 123\n")
            with self.assertRaises(ValueError):
                cli.config_arguments(path)

    def test_reject_unresolved_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text('[arguments]\nmodel = "${FOLD2REASON_MISSING_TEST_VARIABLE}"\n')
            with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
                cli.config_arguments(path)

    def test_all_backend_help_without_loading_models(self):
        for group, command in cli.COMMANDS:
            with self.subTest(command=(group, command)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    cli.main([group, command, "--help"])
                self.assertEqual(caught.exception.code, 0)

    def test_all_recipe_arguments_are_accepted(self):
        recipes = {
            "train/scaling.toml": "training.workspace",
            "train/full.toml": "training.workspace",
            "train/corpus_only.toml": "training.pure_lora",
            "train/decoder.toml": "training.geometry",
            "eval/general10_text.toml": "evaluation.general_text",
            "eval/ftb.toml": "evaluation.ftb",
            "eval/spatial.toml": "evaluation.spatial",
        }
        for config, module_path in recipes.items():
            with self.subTest(config=config):
                module = importlib.import_module("fold2reason." + module_path)
                # Capture only the parser, without running a model or reading data.
                with patch("argparse.ArgumentParser.parse_args", lambda parser: parser):
                    parser = module.parse_args()
                args = [cli.ALIASES.get(v, v) for v in cli.config_arguments(ROOT / "configs" / config)]
                for action in parser._actions:
                    if action.required and not any(flag in args for flag in action.option_strings):
                        args.extend([action.option_strings[0], "artifacts/test-placeholder"])
                parsed = parser.parse_args(args)
                if config == "eval/general10_text.toml":
                    self.assertEqual(parsed.max_examples_per_dataset, 0)
                    self.assertEqual(len(parsed.datasets), 7)

    def test_unknown_command(self):
        with self.assertRaises(SystemExit):
            cli.main(["train", "typo"])


if __name__ == "__main__":
    unittest.main()
