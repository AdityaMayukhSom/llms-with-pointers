import math
import unittest

import torch

from src.utils import JensenShannonUtils


class TestJensenShannonUtils(unittest.TestCase):
    def setUp(self):
        self.js_utils = JensenShannonUtils()

    def test_jsd_divergence_between_distributions(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], dtype=torch.float32)
        Q = torch.tensor([[0.30, 0.50, 0.20]], dtype=torch.float32)
        expected_js_divergence = math.pow(0.0508, 2)
        js_divergence = self.js_utils.compute(P, Q, return_value="divergence", is_probability=True).item()
        self.assertAlmostEqual(js_divergence, expected_js_divergence, places=4)

        P = torch.tensor([[0.216, 0.432, 0.288, 0.064]], dtype=torch.float32)
        Q = torch.tensor([[0.1, 0.3, 0.4, 0.2]], dtype=torch.float32)
        expected_js_divergence = 0.039847
        js_divergence = self.js_utils.compute(P, Q, return_value="divergence", is_probability=True).item()
        self.assertAlmostEqual(js_divergence, expected_js_divergence, places=2)

    def test_jsd_distance_between_distributions(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], dtype=torch.float32)
        Q = torch.tensor([[0.30, 0.50, 0.20]], dtype=torch.float32)
        expected_js_distance = 0.050803
        js_distance = self.js_utils.compute(P, Q, return_value="distance", is_probability=True).item()
        self.assertAlmostEqual(js_distance, expected_js_distance, places=3)

    def test_jsd_between_identical_distributions(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], dtype=torch.float32)
        Q = torch.tensor([[0.36, 0.48, 0.16]], dtype=torch.float32)
        expected_jsd = 0.0
        jsd_divergence = self.js_utils.compute(P, Q, return_value="divergence", is_probability=True).item()
        jsd_distance = self.js_utils.compute(P, Q, return_value="distance", is_probability=True).item()
        self.assertAlmostEqual(jsd_divergence, expected_jsd, places=0)
        self.assertAlmostEqual(jsd_distance, expected_jsd, places=0)


if __name__ == "__main__":
    unittest.main()
