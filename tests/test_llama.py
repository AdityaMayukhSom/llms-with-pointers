import unittest

import torch

from src.llama import PointerGeneratorLlamaForCausalLM


class TestPointerGeneratorLlama(unittest.TestCase):
    def test_project_attention_on_vocab(self):
        batch_size, vocab_size = 2, 10

        input_ids = torch.tensor([[1, 5, 3], [7, 2, 6]])
        attention = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.6]])

        # Expected output shape
        expected_output_shape = (batch_size, vocab_size)
        expected = torch.zeros(expected_output_shape, dtype=torch.float32)

        expected[0, [1, 5, 3]] = torch.tensor([0.2, 0.3, 0.5])
        expected[1, [7, 2, 6]] = torch.tensor([0.1, 0.4, 0.6])

        observed = PointerGeneratorLlamaForCausalLM.project_attention_on_vocab(vocab_size, input_ids, attention)

        self.assertEqual(observed.shape, expected_output_shape, "attention projection shape mismatch")
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
