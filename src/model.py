import textwrap

import torch
from loguru import logger
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig

from src.config import ScriptArguments
from src.llama import PointerGeneratorLlamaForCausalLM


def create_and_prepare_model(config: ScriptArguments, device: torch.device):
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)

    # commented qlora stuff
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    if torch.cuda.is_available() and compute_dtype == torch.float16 and config.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("~" * 80)
            logger.info("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            logger.info("~" * 80)

    device_map = {"": 0}

    model = PointerGeneratorLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        quantization_config=bnb_config,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    if not isinstance(model, PointerGeneratorLlamaForCausalLM):
        error_message = f"""
        Runtime type of model is `{type(model).__name__}`, 
        Type `PointerGeneratorLlamaForCausalLM` is required.
        """
        logger.error(textwrap.dedent(error_message))
        raise ValueError(textwrap.dedent(error_message))

    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
    )

    # For decoder only models, left padding is used.
    # Refer: https://discuss.huggingface.co/t/the-effect-of-padding-side/67188
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pretraining_tp = 1
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, peft_config
