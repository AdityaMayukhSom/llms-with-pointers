import os

import torch
from loguru import logger
from peft.auto import AutoPeftModelForCausalLM
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora.config import LoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

from src.config import ScriptArguments
from src.dataset import get_dataset
from src.model import create_and_prepare_model


def model_train(config: ScriptArguments, device: torch.device):
    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        optim=config.optim,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to=config.reports_to,
    )

    logger.info("Creating and Preparing Model For Training.")
    model, tokenizer, peft_config = create_and_prepare_model(config, device=device)
    logger.success("Model Successfully Created For Training.")

    logger.info("Creating Train Dataset.")
    train_ds = get_dataset(
        data_filename="single/tfrecord/train.tfrecord",
        index_filename="single/tfindex/train.tfindex",
        base_data_directory=config.train_data_dir,
    )
    val_ds = get_dataset(
        data_filename="single/tfrecord/val.tfrecord",
        index_filename="single/tfindex/val.tfindex",
        base_data_directory=config.validation_data_dir,
    )
    logger.success("Train Dataset Successfully Created.")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config.packing,
    )

    trainer.train()

    if config.merge_and_push:
        output_dir = os.path.join(config.train_output_dir, "final_checkpoints")
        trainer.model.save_pretrained(output_dir)

        # Free memory from mergin weights
        del model
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model: PreTrainedModel = model.merge_and_unload()

        output_merged_dir = os.path.join(config.output_dir, "final_merged_checkpoints")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
