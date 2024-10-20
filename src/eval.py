import torch
from loguru import logger
from src.config import ScriptArguments
from src.model import create_and_prepare_model
from transformers import GenerationConfig
from transformers import LlamaForCausalLM
from transformers import LogitsWarper
from transformers import LogitsProcessor
from transformers import LogitsProcessorList
from transformers.generation import GenerateDecoderOnlyOutput


def model_eval(config: ScriptArguments):
    # Characterizing Mechanisms for Factual Recall in Language Models
    # https://arxiv.org/pdf/2310.15910
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, peft_config, tokenizer = create_and_prepare_model(config)
    prompt = "What is Google?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device=device)
    outputs: GenerateDecoderOnlyOutput = model.generate(
        **inputs,
        max_new_tokens=1,
        num_return_sequences=1,
        output_logits=True,
        output_scores=True,
        output_attentions=True,
        return_dict_in_generate=True,
        logits_processor=LogitsProcessorList([]),
    )

    logger.info(outputs.sequences.shape)

    logits = outputs.logits
    attentions = outputs.attentions

    # next_token_login = logits[:, -1, :]

    # final_dist = next_token_login + project_attention_on_vocab(attentions)
    # final_tokens = torch.argmax(final_dist)
    # final_words = tokenizer.decode(final_tokens, skip_special_tokens=True)

    print(type(outputs).__name__)

    # print(outputs)
    # text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(text)
