import pprint

import torch
from loguru import logger
from transformers import LlamaForCausalLM, LogitsProcessorList
from transformers.generation import GenerateDecoderOnlyOutput

from src.config import ScriptArguments
from src.dataset import get_dataset
from src.model import create_and_prepare_model


def model_test(config: ScriptArguments, device: torch.device):
    # Characterizing Mechanisms for Factual Recall in Language Models, https://arxiv.org/pdf/2310.15910
    test_dataset = get_dataset(config)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2)
    model, peft_config, tokenizer = create_and_prepare_model(config, device=device)
    for sample in test_loader:
        articles = sample["article"]
        abstracts = sample["abstract"]
        inputs = tokenizer(articles, return_tensors="pt", padding=True, truncation=True).to(device=device)
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

        text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        logger.info(text)
        break
