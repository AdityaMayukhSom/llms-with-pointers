from inspect import cleandoc
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.utils import DivergenceUtils


class PointerGeneratorLlamaUtils:
    def __init__(self, *, num_hidden_layers: int, dola_candidate_indices: List[int]):
        if not num_hidden_layers:
            raise ValueError("Cannot have zero hidden layers, `num_hidden_layer` must be more than zero.")

        self.__num_hidden_layers = num_hidden_layers
        self.__divergence_utils = DivergenceUtils()

        # Declaration is done to satisfy the intellsense. Otherwise
        # this variable was not shown in intellisense suggestions
        self.__dola_candidate_indices: List[int] = []
        self.set_dola_candiate_indices(dola_candidate_indices)

    def set_dola_candiate_indices(self, dola_candidate_indices: List[int]):
        if not dola_candidate_indices:
            raise ValueError("Atleast one DoLA candidate layer index must be specified.")

        sorted_dola_candidate_indices = sorted(dola_candidate_indices)
        max_candidate_idx = sorted_dola_candidate_indices[-1]

        if max_candidate_idx >= self.__num_hidden_layers:
            err_msg = f"""
            Invalid DoLA candidate layer indices: 
            Expected values in the range {0} to {self.__num_hidden_layers - 1}.
            Found an index exceeding the limit: {max_candidate_idx}. 
            """
            raise ValueError(cleandoc(err_msg))

        self.__dola_candidate_indices = sorted_dola_candidate_indices

    def reduce_multihead_attention(self, multihead_attention: torch.Tensor):
        return torch.mean(multihead_attention, dim=(1, 2), keepdim=False)

    def project_attention_on_vocab(
        self,
        vocab_size: int,
        input_ids: torch.FloatTensor | torch.LongTensor,
        reduced_attention: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Converts position wise attention values into token wise attention distribution. More formally, puts positional attention value into the correspnding id's index in a tensor with length as vocab length. All the remaining position's values are initialized with zero.

        Args:
            vocab_size:
                The total number of tokens present in the vocabulary on which to distribute the reduced_attention over.

            input_ids:
                This is required because the attention values provided by the model are for each individual position in the input. Hence we can index the `vocab_projection` with the combination for `batch_indices`and `input_ids` to assign `reduced_attention` value for that particular position in `vocab_projection`.

            reduced_attention:
                attention value for each position in the input. Must have same shape as `input_ids`.
        """
        assert reduced_attention.shape == input_ids.shape
        batch_size, seq_len = reduced_attention.shape
        proj_shape = (batch_size, vocab_size)
        vocab_projection = torch.zeros(proj_shape, device=reduced_attention.device, dtype=reduced_attention.dtype)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        vocab_projection[batch_indices, input_ids] = reduced_attention
        return vocab_projection

    def create_non_input_prompt_mask(
        self, current_length: int, initial_tokens_count: int, batch_size: int
    ) -> torch.Tensor:
        single_mask = torch.arange(current_length) < initial_tokens_count
        mask = single_mask.unsqueeze(0).expand(batch_size, -1)
        return mask

    def calc_copy_probability(self, *, scores: torch.Tensor):
        anchor_layer = scores[:, -1, :]
        candidate_layers = scores[:, self.__dola_candidate_indices, :]
        divergences = self.__divergence_utils.jensen_shannon(anchor_layer, candidate_layers)
        return torch.mean(divergences)

    def calc_final_distribution(
        self,
        *,
        input_ids: torch.LongTensor,
        logits: Tuple[torch.Tensor],
        attention: torch.FloatTensor | torch.HalfTensor,
        instrn_tok_cnt: int,
        prompt_tok_cnt: int,
    ):
        """
        Calculates the final probability distribution by combining the generated vocabulary distribution
        with the pointer attention distribution over the source document.

        Args:
            input_ids (torch.LongTensor):
                Token IDs of the input sequence upto currently generated token.

            logits (Tuple[torch.Tensor]):
                Logits from the model output for each layer about each token in the sequence, used to
                compute the vocabulary distribution.

            attention (torch.FloatTensor | torch.HalfTensor):
                Multi-head attention weights from the model output, with one attention value per head.
                This function reduces the attention values to match the shape of `input_ids`, allowing
                projection over the vocabulary before modifying the generated distribution.

            instrn_tok_cnt (int):
                Number of tokens in the instruction part of the input sequence; these tokens are ignored
                in the pointer distribution.

            prompt_tok_cnt (int):
                Number of tokens in the entire prompt (including instruction tokens); tokens beyond this
                count are considered part of the generated sequence and excluded from the pointer
                distribution.

        Returns:
            torch.FloatTensor: The final probability distribution for the next token, combining the
            original model's vocabulary distribution with the pointer-based distribution to emphasize
            source document content.

        See:
            * :func:`PointerGeneratorLlamaForCausalLM.project_attention_on_vocab`
        """
        logits = torch.cat(logits, dim=0)
        scores = F.softmax(logits, dim=-1)
        p_vocab = scores[:, -1, :]

        reduced_attn = self.reduce_multihead_attention(attention)

        batch_size, current_tokens_length = reduced_attn.size()
        masked_attn = reduced_attn.clone(memory_format=torch.contiguous_format)
        masked_attn[:, :instrn_tok_cnt, :] = 0
        masked_attn[:, prompt_tok_cnt:, :] = 0

        only_input_mask = self.create_non_input_prompt_mask(current_tokens_length, instrn_tok_cnt, batch_size)
        attn_projection = self.project_attention_on_vocab(self.vocab_size, input_ids, masked_attn)
        p_doc = attn_projection / torch.sum(attn_projection, dim=-1, keepdim=True)

        p_copy = self.calc_copy_probability(logits=logits)

        p_generation = p_copy * p_doc + (1 - p_copy) * p_vocab
        return p_generation
