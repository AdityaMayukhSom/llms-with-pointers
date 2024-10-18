import os
import torch
from datasets import Dataset
from transformers import TrainingArguments
from src.args import script_args
from src.batcher import training_batch_generator
from src.model import create_and_prepare_model
from trl import SFTTrainer
from peft.auto import AutoPeftModelForCausalLM
from transformers import PreTrainedModel

torch.manual_seed(42)

training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_steps=script_args.warmup_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    report_to=script_args.reports_to,
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
tokenizer.padding_side = "right"


train_gen = Dataset.from_generator(training_batch_generator)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
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

    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoints")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
