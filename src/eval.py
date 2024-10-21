import torch
from loguru import logger
from src.config import ScriptArguments
from src.model import PointerGeneratorLlama, create_and_prepare_model
from transformers import LogitsProcessorList
from transformers import LlamaForCausalLM
from transformers.generation import GenerateDecoderOnlyOutput


def model_eval(config: ScriptArguments):
    # Characterizing Mechanisms for Factual Recall in Language Models, https://arxiv.org/pdf/2310.15910
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = ["What is Google?"]

    model, peft_config, tokenizer = create_and_prepare_model(config)
    model = model.to(device=device)
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
