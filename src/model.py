from typing import assert_type
import torch
from loguru import logger
from peft import LoraConfig

from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from src.config import ScriptArguments


def create_and_prepare_model(config: ScriptArguments):
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    # commented qlora stuff
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    if compute_dtype == torch.float16 and config.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("~" * 120)
            logger.info("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            logger.info("~" * 120)

    device_map = {"": 0}

    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        # quantization=bnb_config,
        device_map=device_map,
        use_auth_token=True,
    )

    # model.config.pretrained_tp = 1
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

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

    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, peft_config, tokenizer
