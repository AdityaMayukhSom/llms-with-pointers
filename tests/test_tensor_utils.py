import sys
import unittest
from io import StringIO

import torch

from src.utils import TensorUtils


class TestTensorUtils(unittest.TestCase):

    def setUp(self):
        # Redirect stdout to capture print statements
        self.held_stdout = StringIO()
        self.default_stdout = sys.stdout
        sys.stdout = self.held_stdout
        TensorUtils.set_debug_mode(True)

    def tearDown(self):
        # Reset stdout
        sys.stdout = self.default_stdout

    def test_inspect_details_list(self):
        tensor_list = [torch.tensor(0.5), torch.tensor(1.0)]
        TensorUtils.inspect_details(tensor_list, "test_list")
        output = self.held_stdout.getvalue().strip()
        expected_output = "test_list type is `list`, len is 2 elem type is Tensor"
        self.assertIn(expected_output, output)

    def test_inspect_details_empty_list(self):
        empty_list = []
        TensorUtils.inspect_details(empty_list, "empty_list")
        output = self.held_stdout.getvalue().strip()
        expected_output = "empty_list type is `list`, len is 0 does not contain any element"
        self.assertIn(expected_output, output)

    def test_inspect_details_non_tensor(self):
        non_tensor = "string"
        TensorUtils.inspect_details(non_tensor, "non_tensor")
        output = self.held_stdout.getvalue().strip()
        expected_output = "non_tensor not of type torch.Tensor, type is `str`"
        self.assertIn(expected_output, output)

    def test_nan_count(self):
        tensor_with_nan = torch.tensor([float("nan"), 1.0, 2.0, float("nan")])
        nan_count = TensorUtils.nan_count(tensor_with_nan, "tensor_with_nan")
        output = self.held_stdout.getvalue().strip()
        expected_output = "nan count tensor_with_nan 2"
        self.assertIn(expected_output, output)
        self.assertEqual(nan_count, 2)

    def test_inf_count(self):
        tensor_with_inf = torch.tensor([float("inf"), 1.0, 2.0, float("-inf")])
        inf_count = TensorUtils.inf_count(tensor_with_inf, "tensor_with_inf")
        output = self.held_stdout.getvalue().strip()
        expected_output = "inf count tensor_with_inf 2"
        self.assertIn(expected_output, output)
        self.assertEqual(inf_count, 2)

    def test_inspect_min_max(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        TensorUtils.inspect_min_max(tensor, "test_min_max", dim=0)
        output = self.held_stdout.getvalue().strip()
        expected_min_output = "test_min_max minimum value [[1.0, 2.0]]"
        expected_max_output = "test_min_max maximum value [[3.0, 4.0]]"
        self.assertIn(expected_min_output, output)
        self.assertIn(expected_max_output, output)

    def test_count_nan_and_inf(self):
        tensor_with_nan_inf = torch.tensor([float("nan"), float("inf"), 1.0, float("-inf"), 2.0, float("nan")])
        nan_count, inf_count = TensorUtils.count_nan_and_inf(tensor_with_nan_inf, "tensor_with_nan_inf")
        output = self.held_stdout.getvalue().strip()
        expected_nan_output = "nan count tensor_with_nan_inf 2"
        expected_inf_output = "inf count tensor_with_nan_inf 2"
        self.assertIn(expected_nan_output, output)
        self.assertIn(expected_inf_output, output)
        self.assertEqual(nan_count, 2)
        self.assertEqual(inf_count, 2)


if __name__ == "__main__":
    unittest.main()
