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
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import LlamaConfig

from src.utils import PointerGeneratorLlamaUtils, TensorUtils


class PointerGeneratorLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super(PointerGeneratorLlamaForCausalLM, self).__init__(config)
        self._instrn_tok_cnt = 0
        self._num_hidden_layers = config.num_hidden_layers
        self._llama_utils = PointerGeneratorLlamaUtils(
            vocab_size=self.vocab_size,
            num_hidden_layers=config.num_hidden_layers,
            dola_candidate_indices=[16, 20, 24],
        )

    def set_instrn_tok_cnt(self, instrn_tok_cnt: int):
        self._instrn_tok_cnt = instrn_tok_cnt

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
            outputs: CausalLMOutputWithPast = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # The zeroth hidden state is the input embedding, we stack them using the
            # second dimention as first dimention is batch size hence layer_hidden_states
            # shape (batch_size, (decoder_hidden_layers + 1), generated_tokens, hidden_dim)
            layer_hidden_states = torch.stack(outputs.hidden_states, dim=1).clone()

            # Second dimention have `:` because `dola_candiate_indices` will keep the dimention of
            # second layer in the `candiate_layers` too, but the third index is only `-1` without `:`
            # because we only care about the latest generated token and can discard previous tokens.
            llama_hidden_state = layer_hidden_states[:, -1:, -1, :]
            dola_hidden_states = layer_hidden_states[:, self._llama_utils._dola_candidate_indices, -1, :]

            # layer_logits shape (batch_size, decoder_hidden_layers, generated_tokens, vocab_size)
            llama_logit = self.lm_head(llama_hidden_state)
            dola_logits = self.lm_head(dola_hidden_states)

            del layer_hidden_states

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large
            # for first iteration (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()

            next_token_logits = self._llama_utils.calc_final_dist(
                input_ids=input_ids,
                llama_logit=llama_logit,
                dola_logits=dola_logits,
                attentions=outputs.attentions,
                instrn_tok_cnt=self._instrn_tok_cnt,
                prompt_tok_cnt=initial_tokens_count,
            )

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # set all value set to -inf by logits_processor to zero
            next_token_scores[next_token_scores < 0] = 0

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

            # token selection
            if do_sample:
                # probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_tokens = torch.multinomial(next_token_scores, num_samples=1).squeeze(1)
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
            del next_token_logits
            del next_token_scores

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
