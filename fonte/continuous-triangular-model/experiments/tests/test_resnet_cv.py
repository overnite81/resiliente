import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.run_resnet_cv import (
    METRICS,
    MODELS,
    make_grouped_folds,
    summarize_folds,
    write_summary,
)


class GroupedFoldTests(unittest.TestCase):
    def test_supported_backbones_cover_requested_architectures(self):
        self.assertEqual(MODELS, ("resnet18", "resnet34", "efficientnet_b0"))

    def test_five_folds_keep_records_separate_and_both_classes(self):
        groups = np.repeat([f"record-{i}" for i in range(10)], 4)
        labels = np.tile([0, 0, 1, 1], 10)
        folds = make_grouped_folds(labels, groups, n_splits=5, seed=7)
        self.assertEqual(len(folds), 5)
        for train_idx, valid_idx in folds:
            self.assertFalse(set(groups[train_idx]) & set(groups[valid_idx]))
            self.assertEqual(set(labels[train_idx]), {0, 1})
            self.assertEqual(set(labels[valid_idx]), {0, 1})

    def test_fold_summary_reports_sample_sd_and_t_interval(self):
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        rows = [{metric: value for metric in METRICS} for value in values]
        summary = summarize_folds(rows)
        accuracy = next(row for row in summary if row["metric"] == "accuracy")
        self.assertEqual(accuracy["n_folds"], 5)
        self.assertAlmostEqual(accuracy["mean"], 0.6)
        self.assertAlmostEqual(accuracy["std"], np.std(values, ddof=1))
        self.assertLess(accuracy["ci95_low"], accuracy["mean"])
        self.assertGreater(accuracy["ci95_high"], accuracy["mean"])

    def test_summary_keeps_model_representation_pairs_separate(self):
        rows = []
        for model, representation, values in (
            ("resnet18", "wavelet", [0.2, 0.4]),
            ("efficientnet_b0", "wavelet", [0.6, 0.8]),
        ):
            for value in values:
                rows.append({
                    "model": model,
                    "representation": representation,
                    **{metric: value for metric in METRICS},
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            write_summary(path, rows)
            with path.open(newline="") as stream:
                summaries = list(csv.DictReader(stream))
        accuracy_rows = [row for row in summaries if row["metric"] == "accuracy"]
        means = {(row["model"], row["representation"]): float(row["mean"])
                 for row in accuracy_rows}
        self.assertAlmostEqual(means["resnet18", "wavelet"], 0.3)
        self.assertAlmostEqual(means["efficientnet_b0", "wavelet"], 0.7)


if __name__ == "__main__":
    unittest.main()
