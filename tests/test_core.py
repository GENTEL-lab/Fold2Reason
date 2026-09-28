import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fold2reason.data.folding_corpus import recompute_spatial_answer, stable_seed
from fold2reason.evaluation.geometry import kabsch_align, lddt_ca
from fold2reason.models.workspace import SpatialWorkspace
from fold2reason.training.pure_lora import PureLoraRelationCEModel


class DummyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(6, 8))

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits.unsqueeze(0))


class CoreTests(unittest.TestCase):
    def test_selected_answer_loss_uses_causal_shift(self):
        torch.manual_seed(1)
        language_model = DummyLM()
        model = PureLoraRelationCEModel(language_model)
        labels = torch.tensor([-100, -100, 3, -100, 5, -100])
        actual = model(torch.arange(6), labels)
        expected = F.cross_entropy(language_model.logits[[1, 3]], torch.tensor([3, 5]))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(language_model.logits.grad[[0, 2, 4, 5]].abs().sum().item(), 0)
        self.assertGreater(language_model.logits.grad[[1, 3]].abs().sum().item(), 0)

    def test_empty_supervision_is_rejected(self):
        model = PureLoraRelationCEModel(DummyLM())
        with self.assertRaisesRegex(RuntimeError, "no supervised answer"):
            model(torch.arange(6), torch.full((6,), -100))

    def test_workspace_shapes_and_gradient_flow(self):
        torch.manual_seed(7)
        model = SpatialWorkspace(hidden_size=32, width=16, memory_tokens=4,
                                 fingerprint_dim=8, max_sparse_pairs=64)
        hidden = torch.randn(12, 32, requires_grad=True)
        output = model(hidden)
        self.assertEqual(output["entity_memory"].shape, (12, 32))
        self.assertEqual(output["memory_tokens"].shape, (4, 32))
        self.assertEqual(output["retrieval_embedding"].shape, (8,))
        self.assertLessEqual(len(output["workspace_pair_i"]), 64)
        torch.testing.assert_close(output["memory_attention"].sum(-1), torch.ones(4))
        loss = output["entity_memory"].square().mean() + output["memory_tokens"].square().mean()
        loss.backward()
        for parameter in (hidden, model.entity_down.weight, model.pair_mlp[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_structural_metrics_are_rigid_transform_invariant(self):
        rng = np.random.default_rng(7)
        target = rng.normal(size=(16, 3))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        predicted = target @ rotation + np.array([3, -2, 5])
        aligned, distances = kabsch_align(predicted, target)
        np.testing.assert_allclose(aligned, target, atol=1e-6)
        np.testing.assert_allclose(distances, 0, atol=1e-6)
        self.assertAlmostEqual(lddt_ca(predicted, target), 1.0)

    def test_synthetic_foldingcorpus_examples(self):
        path = Path(__file__).resolve().parents[1] / "examples/toy_protein.json"
        example = json.loads(path.read_text())
        ca = torch.tensor(example["ca_coordinates"], dtype=torch.float32)
        row = {"id": example["id"], "target_coords": ca[:, None, :].repeat(1, 4, 1)}
        for question in example["questions"]:
            self.assertEqual(recompute_spatial_answer(row, question), question["answer"])

    def test_data_seed_is_deterministic(self):
        self.assertEqual(stable_seed("protein", 7), stable_seed("protein", 7))
        self.assertNotEqual(stable_seed("protein", 7), stable_seed("protein", 8))


if __name__ == "__main__":
    unittest.main()
