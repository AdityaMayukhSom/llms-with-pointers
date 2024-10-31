import unittest

import torch

from src.utils import PointerGeneratorLlamaUtils


class TestPointerGeneratorLlamaUtils(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pgl_utils = PointerGeneratorLlamaUtils()

    def test_project_attention_on_vocab(self):
        batch_size, vocab_size = 2, 10

        input_ids = torch.tensor([[1, 5, 3], [7, 2, 6]])
        attention = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.6]])

        # Expected output shape
        expected_output_shape = (batch_size, vocab_size)
        expected = torch.zeros(expected_output_shape, dtype=torch.float32)

        expected[0, [1, 5, 3]] = torch.tensor([0.2, 0.3, 0.5])
        expected[1, [7, 2, 6]] = torch.tensor([0.1, 0.4, 0.6])

        observed = self.pgl_utils.project_attention_on_vocab(vocab_size, input_ids, attention)

        self.assertEqual(observed.shape, expected_output_shape, "attention projection shape mismatch")
        torch.testing.assert_close(observed, expected)

    def test_non_input_prompt_mask_creation(self):
        batch_size, init_tok_cnt, cur_len = 3, 4, 7

        newly_generated_tokens = cur_len - init_tok_cnt
        mask_list = [[1] * init_tok_cnt + [0] * newly_generated_tokens] * batch_size
        expected = torch.tensor(mask_list, dtype=torch.bool)

        observed = self.pgl_utils.create_non_input_prompt_mask(cur_len, init_tok_cnt, batch_size)
        torch.testing.assert_close(observed, expected)

    def test_mask_application(self):
        batch_size, init_tok_cnt, cur_len = 3, 4, 7
        mask = self.pgl_utils.create_non_input_prompt_mask(cur_len, init_tok_cnt, batch_size)

        # Create a mock tensor with values from 1 to cur_len, repeated for each batch
        input_tensor = torch.randn((batch_size, cur_len))

        # Apply the mask to the input tensor (e.g., zero out values in masked positions)
        masked_output = input_tensor * mask

        # Define the expected output where only the first `init_tok_cnt` values are preserved, others are zero
        expected_output = input_tensor.clone()
        expected_output[:, init_tok_cnt:] = 0  # Set values after `init_tok_cnt` to zero

        torch.testing.assert_close(masked_output, expected_output)


if __name__ == "__main__":
    unittest.main()
