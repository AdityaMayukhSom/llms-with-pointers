from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerationConfig,
)
from transformers.generation.streamers import BaseStreamer

from src.utils import DivergenceUtils, TensorUtils


class PointerGeneratorLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super(PointerGeneratorLlamaForCausalLM, self).__init__(*args, **kwargs)
        self.divergence_utils = DivergenceUtils()

    @staticmethod
    def reduce_multihead_attention(multihead_attention: torch.LongTensor | torch.FloatTensor | torch.IntTensor):
        return torch.mean(multihead_attention, dim=(1, 2), keepdim=False)

    @staticmethod
    def project_attention_on_vocab(
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
        vocab_projection = torch.zeros(
            (batch_size, vocab_size),
            device=reduced_attention.device,
            dtype=reduced_attention.dtype,
        )
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        vocab_projection[batch_indices, input_ids] = reduced_attention
        return vocab_projection

    @staticmethod
    def create_non_input_prompt_mask(current_length: int, initial_tokens_count: int, batch_size: int) -> torch.Tensor:
        single_mask = torch.arange(current_length) < initial_tokens_count
        mask = single_mask.unsqueeze(0).expand(batch_size, -1)
        return mask

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
        attn_projection = self.project_attention_on_vocab(self.vocab_size, input_ids, reduced_attn & only_input_mask)
        batch_size, current_tokens_length = reduced_attn.size()
        only_input_mask = self.create_non_input_prompt_mask(current_tokens_length, initial_tokens_count, batch_size)

        # attn_projection = torch.softmax(attn_projection, dim=-1)
        # TensorUtils.log_details(attn_projection, "attn_projection")

        # TODO: add soft switch with p_gen as decoder only models do not have any generated context
        # It might have historical hidden states with
        p_gen = 0.5

        # normalize the output to range between zero to one
        # normalized_next_token_scores = torch.softmax(next_token_scores, dim=-1)
        generation_probability = p_gen * next_token_scores + (1 - p_gen) * attn_projection
        return generation_probability

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        initial_tokens_count = input_ids.size(dim=-1)

        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs: Dict[str, Any] = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large
            # for first iteration (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            # last_hidden_layer_attn: torch.FloatTensor = outputs["attentions"][-1]
            # next_token_scores = self.calc_final_distribution(next_token_scores, input_ids, last_hidden_layer_attn)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
