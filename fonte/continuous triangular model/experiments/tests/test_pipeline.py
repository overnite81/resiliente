import unittest

import numpy as np

from experiments.data import extract_windows_from_signal
from experiments.representations import extract_representations


class DataTests(unittest.TestCase):
    def test_window_extraction_preserves_group_and_labels(self):
        signal = np.linspace(-1.0, 1.0, 1000)
        dataset = extract_windows_from_signal(
            signal,
            annotation_samples=[100, 500, 900],
            annotation_symbols=["N", "V", "x"],
            record_id="record-100",
            window_samples=100,
        )
        self.assertEqual(dataset.signals.shape, (2, 100))
        np.testing.assert_array_equal(dataset.labels, [0, 1])
        self.assertEqual(set(dataset.groups), {"record-100"})
        np.testing.assert_allclose(dataset.signals.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(dataset.signals.std(axis=1), 1.0, atol=1e-12)


class RepresentationTests(unittest.TestCase):
    def test_all_representations_are_finite_and_one_dimensional(self):
        signal = np.sin(np.linspace(0, 12 * np.pi, 360))
        representations = extract_representations(signal, ctm_span=32)
        self.assertEqual(set(representations), {"raw", "wavelet", "stft", "ctm"})
        self.assertEqual(representations["ctm"].shape, (496,))
        for values in representations.values():
            self.assertEqual(values.ndim, 1)
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
