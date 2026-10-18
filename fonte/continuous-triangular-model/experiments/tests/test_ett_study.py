import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.run_ett_study import (
    chronological_origins,
    encode_window,
    even_subsample,
    load_ett_csv,
    materialize_targets,
    score_forecast,
)


class ETTStudyTests(unittest.TestCase):
    def test_chronological_splits_keep_forecast_targets_in_their_period(self):
        length, sequence, horizon = 1000, 40, 12
        splits, train_end, validation_end = chronological_origins(
            length, sequence, horizon, 0.6, 0.2, 5
        )
        for start in splits["train"]:
            self.assertLess(start + sequence + horizon, train_end + 1)
        for start in splits["validation"]:
            self.assertGreaterEqual(start + sequence, train_end)
            self.assertLess(start + sequence + horizon, validation_end + 1)
        for start in splits["test"]:
            self.assertGreaterEqual(start + sequence, validation_end)
            self.assertLess(start + sequence + horizon, length + 1)

    def test_loader_uses_date_order_and_identifies_target_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ett.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("date", "HUFL", "OT"))
                for index in range(20):
                    writer.writerow((f"2020-01-{index + 1:02d}", index, index * 2))
            dates, features, values, target_index = load_ett_csv(path, "OT")
        self.assertEqual(len(dates), 20)
        self.assertEqual(features, ["HUFL", "OT"])
        self.assertEqual(target_index, 1)
        np.testing.assert_allclose(values[:, 1], np.arange(20) * 2)

    def test_all_encodings_return_finite_square_maps(self):
        time = np.linspace(0, 8 * np.pi, 64)
        window = np.column_stack((np.sin(time), np.cos(time)))
        for representation in ("raw", "stft", "wavelet", "ctm"):
            image = encode_window(
                window, representation, image_size=32, encoding_samples=32,
                ctm_span=8, wavelet_scales=8, ctm_measure="amplitude",
            )
            self.assertEqual(image.shape, (32, 32))
            self.assertTrue(np.isfinite(image).all())

    def test_uniform_train_subsampling_and_multi_horizon_targets(self):
        selected = even_subsample(np.arange(100), 10)
        self.assertEqual(len(selected), 10)
        self.assertTrue(np.all(np.diff(selected) > 0))
        values = np.arange(60, dtype=np.float32).reshape(30, 2)
        targets = materialize_targets(values, np.array([0, 5]), 10, 4, 1)
        np.testing.assert_array_equal(targets[0], [21, 23, 25, 27])
        np.testing.assert_array_equal(targets[1], [31, 33, 35, 37])

    def test_regression_score_metrics(self):
        scores = score_forecast([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(scores["mae"], 1 / 3)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(1 / 3))


if __name__ == "__main__":
    unittest.main()
