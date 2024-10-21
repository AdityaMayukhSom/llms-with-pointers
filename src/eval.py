import pprint

import torch
from loguru import logger
from transformers import LlamaForCausalLM, LogitsProcessorList
from transformers.generation import GenerateDecoderOnlyOutput

from src.config import ScriptArguments
from src.dataset import get_dataset
from src.model import create_and_prepare_model
from src.transform import batch_transform


def model_eval(config: ScriptArguments, device: torch.device):
    # TODO: FIX Manual Evaluation Code
    prompt = ["What is Google?"]

    model, tokenizer, _ = create_and_prepare_model(config, device=device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device=device)
    outputs: GenerateDecoderOnlyOutput = model.generate(
        **inputs,
        max_new_tokens=200,
        num_return_sequences=1,
        output_logits=True,
        output_scores=True,
        output_attentions=True,
        return_dict_in_generate=True,
    )

    logger.info("output runtime type: {}".format(type(outputs).__name__))
    logger.info(outputs.sequences.shape)

    logits = outputs.logits
    attentions = outputs.attentions

    # next_token_login = logits[:, -1, :]
    # final_dist = next_token_login + project_attention_on_vocab(attentions)
    # final_tokens = torch.argmax(final_dist)
    # final_words = tokenizer.decode(final_tokens, skip_special_tokens=True)
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    logger.info(text)
