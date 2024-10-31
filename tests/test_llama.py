import unittest

import torch

from src.llama import PointerGeneratorLlamaForCausalLM


class TestPGNConfig(unittest.TestCase):
    def test_project_attention_on_vocab(self):
        batch_size = 2
        vocab_size = 10

        input_ids = torch.tensor([[1, 5, 3], [7, 2, 6]])
        attention = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.6]])

        # Expected output shape
        expected_shape = (batch_size, vocab_size)

        result = PointerGeneratorLlamaForCausalLM.project_attention_on_vocab(vocab_size, input_ids, attention)

        self.assertEqual(result.shape, expected_shape, "attention projection shape mismatch")

        # Check correct assignment
        self.assertAlmostEqual(result[0, 1].item(), 0.2, places=4)
        self.assertAlmostEqual(result[0, 5].item(), 0.3, places=4)
        self.assertAlmostEqual(result[0, 3].item(), 0.5, places=4)
        self.assertAlmostEqual(result[1, 7].item(), 0.1, places=4)
        self.assertAlmostEqual(result[1, 2].item(), 0.4, places=4)
        self.assertAlmostEqual(result[1, 6].item(), 0.6, places=4)

        # Check other positions are zero
        for i in range(batch_size):
            for j in range(vocab_size):
                if j not in input_ids[i]:
                    self.assertAlmostEqual(result[i, j].item(), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
