import unittest

import numpy as np

from experiments.image_representations import REPRESENTATIONS, encode_image


class ImageRepresentationTests(unittest.TestCase):
    def test_encoders_return_finite_square_float32_images(self):
        signal = np.sin(np.linspace(0, 12 * np.pi, 360)).astype(np.float64)
        for representation in REPRESENTATIONS:
            with self.subTest(representation=representation):
                image = encode_image(signal, representation, image_size=64, ctm_span=32)
                self.assertEqual(image.shape, (64, 64))
                self.assertEqual(image.dtype, np.float32)
                self.assertTrue(np.isfinite(image).all())
                self.assertGreaterEqual(float(image.min()), 0.0)
                self.assertLessEqual(float(image.max()), 1.0)

    def test_ctm_has_centered_triangular_support_before_resize(self):
        signal = np.linspace(-1.0, 1.0, 100)
        image = encode_image(signal, "ctm", image_size=64, ctm_span=32)
        self.assertGreater(np.count_nonzero(image), 0)
        self.assertLess(np.count_nonzero(image), image.size)


if __name__ == "__main__":
    unittest.main()
