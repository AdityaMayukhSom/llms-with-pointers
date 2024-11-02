from inspect import cleandoc
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .divergence_utils import DivergenceUtils
from .tensor_utils import TensorUtils


class PointerGeneratorLlamaUtils:
    def __init__(self, *, vocab_size: int, num_hidden_layers: int, dola_candidate_indices: List[int]):
        if not num_hidden_layers:
            raise ValueError("Cannot have zero hidden layers, `num_hidden_layer` must be more than zero.")

        self.__vocab_size = vocab_size
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

    def _reduce_multihead_attn(self, multihead_attention: torch.Tensor):
        # Only take attention for the latest generated token.
        gen_token_attn = multihead_attention[:, :, -1, :]

        # Average out the attention over all the heads.
        avg = torch.mean(gen_token_attn, dim=1, keepdim=True)

        # Remove the second dimention, dim is passed so that it does not
        # squeeze any other dimention in case those turn out to be one.
        return torch.squeeze(avg, dim=1)

    def _proj_attn_on_vocab(
        self,
        vocab_size: int,
        input_ids: torch.FloatTensor | torch.LongTensor,
        masked_attn: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Converts position wise attention values into token wise attention distribution. More formally, puts positional attention value into the correspnding id's index in a tensor with length as vocab length. All the remaining position's values are initialized with zero.

        Args:
            vocab_size:
                The total number of tokens present in the vocabulary on which to distribute the reduced_attention over.

            input_ids:
                This is required because the attention values provided by the model are for each individual position in the input. Hence we can index the `vocab_projection` with the combination for `batch_indices`and `input_ids` to assign `reduced_attention` value for that particular position in `vocab_projection`.

            masked_attn:
                attention value for each position in the input. Must have same shape as `input_ids`. The attention values outside user prompt should be masked with zeroes (i.e. both the model generated tokens and instruction) so that model is unable to point to tokens in those parts of the prompt.
        """
        assert masked_attn.shape == input_ids.shape
        batch_size, seq_len = masked_attn.shape
        proj_shape = (batch_size, vocab_size)
        vocab_projection = torch.zeros(proj_shape, device=masked_attn.device, dtype=masked_attn.dtype)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        vocab_projection[batch_indices, input_ids] = masked_attn
        return vocab_projection

    def _calc_copy_proba(self, *, scores: torch.Tensor):
        # Second dimention have `:` because `dola_candiate_indices` will keep the dimention of
        # second layer in the `candiate_layers` too, but the third index is only `-1` without `:`
        # because we only care about the latest generated token and can discard previous tokens.
        anchor_layer = scores[:, -1:, -1, :]
        candidate_layers = scores[:, self.__dola_candidate_indices, -1, :]

        divergences = self.__divergence_utils.jensen_shannon(anchor_layer, candidate_layers)

        # Do not directly squeeze with keepdim=False as that leads to unexpected errors by
        # squeezing the batch dimention in case batch dimention is one which is most of the
        # times during evaluation, hence mean first with dimention, then selectively squeeze
        average_divergence = torch.mean(divergences, dim=1, keepdim=True).squeeze(dim=1)

        del divergences
        del anchor_layer
        del candidate_layers
        return average_divergence

    def calc_final_dist(
        self,
        *,
        input_ids: torch.LongTensor,
        logits: torch.Tensor,
        attentions: Tuple[torch.FloatTensor | torch.HalfTensor],
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
        scores = F.softmax(logits, dim=-1)
        p_vocab = scores[:, -1, -1, :]

        reduced_attn = self._reduce_multihead_attn(attentions[-1])

        masked_attn = reduced_attn.clone(memory_format=torch.contiguous_format)
        masked_attn[:, :instrn_tok_cnt] = 0
        masked_attn[:, prompt_tok_cnt:] = 0

        attn_proj = self._proj_attn_on_vocab(self.__vocab_size, input_ids, masked_attn)
        p_doc = attn_proj / torch.sum(attn_proj, dim=-1, keepdim=True)
        p_copy = self._calc_copy_proba(scores=scores)
        p_gen = p_copy * p_doc + (1 - p_copy) * p_vocab

        del p_doc
        del p_copy
        del p_vocab
        del masked_attn
        del attn_proj
        return p_gen
