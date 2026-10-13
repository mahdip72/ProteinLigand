"""CPU regression tests for both training entrypoints, without loading encoders.

Run with: python -m unittest discover -s tests -p test_padding_mask.py -v
Dependencies: torch, numpy, torchmetrics, scikit-learn.
The actual loss and loop function bodies are loaded via AST to avoid importing
Hugging Face models, datasets, or downloading checkpoints during these tests.
"""
import ast
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / "single_ligand/train.py", ROOT / "zero_shot/multi_ligand_train.py"]
torch.set_num_threads(1)


def load_functions(path):
    names = {"calculate_loss", "training_loop", "validation_loop", "evaluation_loop"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, np=np, F=F, torchmetrics=torchmetrics,
                     defaultdict=defaultdict, sk_f1=f1_score,
                     tqdm=SimpleNamespace(tqdm=lambda items, **kw: items),
                     autocast=lambda **kw: nullcontext(),
                     apply_random_masking=lambda inputs, **kw: inputs)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def config():
    return SimpleNamespace(
        use_positive_weighting_for_binding_sites=True, use_dataset_weighting=True,
        ligands=["A", "B"], dataset_format="BioLiP",
        dataset_weight={"A": 2.0, "B": 0.5}, pos_dataset_weight={"A": 3.0, "B": 2.0},
        train_settings=SimpleNamespace(mixed_precision="no", num_epochs=1),
        mask_prob=0, mask_token_id=0)


def inputs(padding=0):
    logits = torch.tensor([[2., -2., 2.], [2., -2., 9.]])
    labels = torch.tensor([[1, 0, 0], [1, 1, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    if padding:
        logits = F.pad(logits, (0, padding), value=9.)
        labels = F.pad(labels, (0, padding), value=0)
        mask = F.pad(mask, (0, padding), value=0)
    return logits.unsqueeze(-1).requires_grad_(), labels, mask


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.detach().clone())

    def forward(self, **kwargs):
        return self.logits


class Loader(list):
    dataset = SimpleNamespace(get_amino_acid_token_ids=lambda: [1, 2])


class PaddingMaskTests(unittest.TestCase):
    def test_stage1_dynamic_weight_is_uncapped(self):
        loss_fn = load_functions(PATHS[0])["calculate_loss"]
        previous_weight = None
        # Valid negative/positive ratios of 9 and 4 give EMA weights 9 and 8.5.
        for negatives, expected_weight in [(9, 9.), (4, 8.5)]:
            labels = torch.tensor([[1] + [0] * negatives + [0] * 3])
            mask = torch.tensor([[1] * (negatives + 1) + [0] * 3])
            logits = torch.zeros((*labels.shape, 1), requires_grad=True)
            loss, previous_weight = loss_fn(logits, labels, attention_mask=mask,
                smoothed_pos_weight=previous_weight, device="cpu")
            torch.testing.assert_close(torch.as_tensor(previous_weight), torch.tensor(expected_weight))
            valid = mask.bool()
            expected_loss = F.binary_cross_entropy_with_logits(
                logits[..., 0][valid], labels[valid].float(),
                pos_weight=torch.tensor(expected_weight))
            torch.testing.assert_close(loss, expected_loss)

    def test_loss_and_gradients_ignore_padding(self):
        for path in PATHS:
            loss_fn = load_functions(path)["calculate_loss"]
            for focal in [False, True]:
                for smoothing in [0., 0.1]:
                    for previous_weight in [None, 2.]:
                        with self.subTest(path=path, focal=focal, smoothing=smoothing, weight=previous_weight):
                            results = []
                            for padding in [0, 4]:
                                logits, labels, mask = inputs(padding)
                                # Excluded positions must not contaminate BCE or gradients.
                                with torch.no_grad():
                                    logits[..., 0][~mask.bool()] = float("nan")
                                labels[~mask.bool()] = 1
                                kwargs = dict(configs=config(), ligands=["A", "B"]) if path == PATHS[1] else {}
                                loss, weight = loss_fn(logits, labels, attention_mask=mask,
                                    device="cpu", use_focal_loss=focal, label_smoothing=smoothing,
                                    smoothed_pos_weight=previous_weight, **kwargs)
                                self.assertTrue(torch.isfinite(loss))
                                loss.backward()
                                self.assertTrue(torch.isfinite(logits.grad).all())
                                self.assertEqual(torch.count_nonzero(logits.grad[..., 0][~mask.bool()]).item(), 0)
                                results.append((loss.detach(), weight, logits.grad[..., 0][mask.bool()]))
                            torch.testing.assert_close(results[0][0], results[1][0])
                            self.assertEqual(results[0][1], results[1][1])
                            torch.testing.assert_close(results[0][2], results[1][2])

    def test_expected_valid_residue_loss_and_weights(self):
        for path in PATHS:
            for focal in [False, True]:
                for smoothing in [0., 0.1]:
                    with self.subTest(path=path, focal=focal, smoothing=smoothing):
                        logits, labels, mask = inputs(4)
                        valid = mask.bool()
                        kwargs = dict(configs=config(), ligands=["A", "B"]) if path == PATHS[1] else {}
                        loss, weight = load_functions(path)["calculate_loss"](
                            logits, labels, attention_mask=mask, device="cpu",
                            use_focal_loss=focal, label_smoothing=smoothing, **kwargs)
                        target = labels[valid].float() * (1 - smoothing)
                        target += smoothing * (0.01 if path == PATHS[0] else 0.5)
                        pos_weight = torch.tensor(2 / 3) if path == PATHS[0] else None
                        bce = F.binary_cross_entropy_with_logits(logits[..., 0][valid], target,
                                                                  pos_weight=pos_weight, reduction="none")
                        if focal:
                            bce = (1 - torch.exp(-bce)) ** 2 * bce
                        if path == PATHS[1]:
                            bce = bce * torch.tensor([6., 2., 2., 1., 1.])
                        else:
                            self.assertAlmostEqual(float(weight), 2 / 3, places=6)
                        torch.testing.assert_close(loss, bce.mean())

    def test_empty_mask_and_wrong_shape_fail_clearly(self):
        for path in PATHS:
            fn = load_functions(path)["calculate_loss"]
            logits, labels, mask = inputs()
            for invalid in [torch.zeros_like(mask), torch.ones(1, 6)]:
                with self.subTest(path=path, mask=invalid.shape):
                    with self.assertRaises(ValueError):
                        fn(logits, labels, attention_mask=invalid, device="cpu", configs=config())

    def test_unpadded_loss_and_optional_weights(self):
        logits = torch.tensor([[[2.], [-2.]], [[-1.], [1.]]])
        labels = torch.tensor([[1, 0], [0, 1]])
        mask = torch.ones_like(labels)
        bce = F.binary_cross_entropy_with_logits(logits[..., 0], labels.float(), reduction="none")
        stage1_loss, weight = load_functions(PATHS[0])["calculate_loss"](
            logits, labels, attention_mask=mask, device="cpu")
        torch.testing.assert_close(stage1_loss, bce.mean())
        self.assertEqual(float(weight), 1.)
        for positive_weighting in [False, True]:
            for dataset_weighting in [False, True]:
                with self.subTest(positive=positive_weighting, dataset=dataset_weighting):
                    cfg = config()
                    cfg.use_positive_weighting_for_binding_sites = positive_weighting
                    cfg.use_dataset_weighting = dataset_weighting
                    loss, _ = load_functions(PATHS[1])["calculate_loss"](
                        logits, labels, attention_mask=mask, configs=cfg, ligands=["A", "B"])
                    weights = torch.ones_like(bce)
                    if positive_weighting:
                        weights *= torch.tensor([[3., 1.], [1., 2.]])
                    if dataset_weighting:
                        weights *= torch.tensor([[2.], [0.5]])
                    torch.testing.assert_close(loss, (weights * bce).mean())

    def test_all_six_loops_use_only_valid_residues(self):
        for path in PATHS:
            functions = load_functions(path)
            for loop in ["training_loop", "validation_loop", "evaluation_loop"]:
                with self.subTest(path=path, loop=loop):
                    results = []
                    for padding in [0, 4]:
                        logits, labels, mask = inputs(padding)
                        batch = dict(input_ids=torch.ones_like(labels), labels=labels,
                                     attention_mask=mask, ligand=["A", "B"], ligand_idx=torch.tensor([0, 1]))
                        model = FixedLogits(logits)
                        kwargs = dict(configs=config()) if path == PATHS[1] else {}
                        with redirect_stdout(io.StringIO()):
                            if loop == "training_loop":
                                result = functions[loop](model, Loader([batch]),
                                    torch.optim.SGD(model.parameters(), lr=0.), epoch=0,
                                    device=torch.device("cpu"), scaler=torch.amp.GradScaler("cpu", enabled=False),
                                    scheduler=None, **kwargs)
                                self.assertEqual(torch.count_nonzero(model.logits.grad[..., 0][~mask.bool()]).item(), 0)
                            elif loop == "validation_loop":
                                result = functions[loop](model, Loader([batch]), epoch=0,
                                    device=torch.device("cpu"), **kwargs)
                            else:
                                result = functions[loop](model, Loader([batch]), device=torch.device("cpu"),
                                    log_confidences=True, **kwargs)
                        if isinstance(result, dict):
                            self.assertEqual([result[k] for k in ["true_positives", "false_positives",
                                                                  "true_negatives", "false_negatives"]], [2, 1, 1, 1])
                            self.assertEqual(len(result["incorrect_confidence_scores"]), 2)
                            results.append([v for k, v in result.items() if k != "incorrect_confidence_scores"])
                            self.assertAlmostEqual(result["f1_score"], 2 / 3, places=6)
                        else:
                            results.append(result)
                            self.assertAlmostEqual(result[1], 2 / 3, places=6)
                    np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
