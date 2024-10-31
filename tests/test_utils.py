import math
import unittest

import torch

from src.utils import DivergenceUtils


class TestDivergenceUtils(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.divergence_utils = DivergenceUtils()

    def test_kullback_leibler_divergence(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], device=self.device)
        Q = torch.tensor([[0.333_333, 0.333_333, 0.333_333]], device=self.device)

        observed_pq = self.divergence_utils.kullback_leibler(P, Q)
        expected_pq = torch.tensor([[0.085_299]], device=self.device)
        torch.testing.assert_close(observed_pq, expected_pq)

        observed_qp = self.divergence_utils.kullback_leibler(Q, P)
        expected_qp = torch.tensor([[0.097_455]], device=self.device)
        torch.testing.assert_close(observed_qp, expected_qp)

    def test_kullback_leibler_divergence_on_jensen_shannon_example(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], device=self.device)
        Q = torch.tensor([[0.30, 0.50, 0.20]], device=self.device)
        M = torch.tensor([[0.33, 0.49, 0.18]], device=self.device)

        observed_pq = self.divergence_utils.kullback_leibler(P, Q)
        expected_pq = torch.tensor([[0.010_338_235]], device=self.device)
        torch.testing.assert_close(observed_pq, expected_pq)

        observed_qp = self.divergence_utils.kullback_leibler(Q, P)
        expected_qp = torch.tensor([[0.010_343_239]], device=self.device)
        torch.testing.assert_close(observed_qp, expected_qp)

        observed_pm = self.divergence_utils.kullback_leibler(P, M)
        expected_pm = torch.tensor([[0.002_581_552]], device=self.device)
        torch.testing.assert_close(observed_pm, expected_pm)

        observed_qm = self.divergence_utils.kullback_leibler(Q, M)
        expected_qm = torch.tensor([[0.002_580_402]], device=self.device)
        torch.testing.assert_close(observed_qm, expected_qm)

    def test_kullback_leibler_divergence_between_identical_distributions(self):
        P = torch.tensor([[0.2, 0.1, 0.3, 0.4]], device=self.device)
        expected_pp = torch.tensor([[0.0]], device=self.device)
        observed_pp = self.divergence_utils.kullback_leibler(P, P)
        torch.testing.assert_close(observed_pp, expected_pp)

    def test_jensen_shannon_divergence(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], device=self.device)
        Q = torch.tensor([[0.30, 0.50, 0.20]], device=self.device)

        observed = self.divergence_utils.jensen_shannon(P, Q, return_value="divergence")
        expected = torch.tensor([[0.002_580_977]], device=self.device)
        torch.testing.assert_close(observed, expected, msg="3 value error")

        P = torch.tensor([[0.216, 0.432, 0.288, 0.064]], device=self.device)
        Q = torch.tensor([[0.2, 0.4, 0.3, 0.1]], device=self.device)

        observed = self.divergence_utils.jensen_shannon(P, Q, return_value="divergence")
        expected = torch.tensor([[0.002_514_670]], device=self.device)
        torch.testing.assert_close(observed, expected, msg="4 value error")

    def test_jensen_shannon_divergence_cumulative_property(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], device=self.device)
        Q = torch.tensor([[0.30, 0.50, 0.20]], device=self.device)

        observed_pq = self.divergence_utils.jensen_shannon(P, Q, return_value="divergence")
        observed_qp = self.divergence_utils.jensen_shannon(Q, P, return_value="divergence")

        torch.testing.assert_close(observed_pq, observed_qp)

    def test_jensen_shannon_divergence_for_batch(self):
        P = torch.tensor([[0.36, 0.48, 0.16], [0.216, 0.432, 0.288]], device=self.device)
        Q = torch.tensor([[0.30, 0.50, 0.20], [0.2, 0.4, 0.4]], device=self.device)

        observed = self.divergence_utils.jensen_shannon(P, Q, return_value="divergence")
        expected = torch.tensor([[0.002_580_64], [0.005_040_140]], device=self.device)
        torch.testing.assert_close(observed, expected)

    def test_jensen_shannon_distance(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], device=self.device)
        Q = torch.tensor([[0.30, 0.50, 0.20]], device=self.device)
        actual = torch.tensor([[0.050_803_321]], device=self.device)
        observed = self.divergence_utils.jensen_shannon(P, Q, return_value="distance")
        torch.testing.assert_close(observed, actual)

    def test_jensen_shannon_divergence_and_distance_between_identical_distributions(self):
        P = torch.tensor([[0.36, 0.48, 0.16]], device=self.device)
        Q = torch.tensor([[0.36, 0.48, 0.16]], device=self.device)

        zero_tensor = torch.zeros([1, 1], device=self.device)

        jsd_divergence = self.divergence_utils.jensen_shannon(P, Q, return_value="divergence")
        jsd_distance = self.divergence_utils.jensen_shannon(P, Q, return_value="distance")

        torch.testing.assert_close(jsd_divergence, zero_tensor)
        torch.testing.assert_close(jsd_distance, zero_tensor)


if __name__ == "__main__":
    unittest.main()
