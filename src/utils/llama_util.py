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

        # This declaration is done for satisfying the intellsense otherwise
        # this variable was not shown in intelliisense suggestions
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

    def reduce_multihead_attention(
        self,
        multihead_attention: torch.LongTensor | torch.FloatTensor | torch.IntTensor,
    ):
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

    def calc_copy_probability(self, *, logits: Tuple[torch.Tensor]):
        hidden_layer_scores = F.softmax(torch.tensor(logits), dim=-1)
        anchor_layer = hidden_layer_scores[:, -1, :]
        candidate_layers = hidden_layer_scores[:, self.__dola_candidate_indices, :]
        divergences = self.__divergence_utils.jensen_shannon(anchor_layer, candidate_layers)
        return torch.max(divergences)

    def calc_final_distribution(
        self,
        next_token_scores: torch.IntTensor | torch.FloatTensor | torch.LongTensor,
        input_ids: torch.LongTensor,
        attention: torch.FloatTensor | torch.HalfTensor,
        initial_tokens_count: int,
    ):
        """
        Calculates final distribution by modifying the generated probability distribution  with attention values used for pointing from the source text.

        Args:
            next_token_scores (torch.IntTensor | torch.FloatTensor | torch.LongTensor):
                _description_

            input_ids (torch.LongTensor):
                _description_

            attention (torch.FloatTensor | torch.HalfTensor):
                This is non-reduced raw attention with multiple head values, i.e. the attention value returned directly from the output of the model. This function also reduces the attention values to the same shape as `input_ids` to be projected over the vocabulary before modifying the generated distribution.

            initial_tokens_count (int):
                Number of initial tokens in the prompt.

        See :func:`PointerGeneratorLlamaForCausalLM.project_attention_on_vocab`
        """

        reduced_attn = self.reduce_multihead_attention(attention)
        batch_size, current_tokens_length = reduced_attn.size()

        only_input_mask = self.create_non_input_prompt_mask(current_tokens_length, initial_tokens_count, batch_size)
        attn_projection = self.project_attention_on_vocab(self.vocab_size, input_ids, reduced_attn * only_input_mask)

        # attn_projection = torch.softmax(attn_projection, dim=-1)
        # TensorUtils.log_details(attn_projection, "attn_projection")

        # TODO: add soft switch with p_gen as decoder only models do not have any generated context
        # It might have historical hidden states with
        p_gen = 0.5

        # normalize the output to range between zero to one
        # normalized_next_token_scores = torch.softmax(next_token_scores, dim=-1)
        generation_probability = p_gen * next_token_scores + (1 - p_gen) * attn_projection
        return generation_probability
