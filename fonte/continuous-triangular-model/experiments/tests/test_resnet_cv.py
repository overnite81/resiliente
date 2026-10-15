import unittest

import numpy as np

from experiments.run_resnet_cv import METRICS, make_grouped_folds, summarize_folds


class GroupedFoldTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
