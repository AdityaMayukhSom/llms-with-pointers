from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity
    and faetures are and what size of model you want to train.
    """

    mode: Optional[Literal["train", "test", "eval"]] = field(
        default="train",
        metadata={
            "help": "Whether to train, test or eval the model.",
        },
    )

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=17)
    learning_rate: Optional[float] = field(default=3e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.01)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[int] = field(default=0.0)
    lora_r: Optional[int] = field(default=16)
    max_seq_length: Optional[int] = field(default=256)

    model_name: Optional[str] = field(
        default="meta-llama/Llama-3.2-3B",
        metadata={
            "help": "The model that you want to train from HuggingFace Hub. E.g. GPT2, BERT, GPT2-XL etc.",
        },
    )

    local_rank: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Used for Multi GPU.",
        },
    )

    dataset_name: Optional[str] = field(
        default="tatsu-lab/alpaca",
        metadata={"help": "The preferenced dataset to use."},
    )

    use_4bit: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Activate 4 bit precision base model loading.",
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Activate nested quantization for 4bit base models.",
        },
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": "Compute dtype for 4 bit base model.",
        },
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={
            "help": "Quantization type, can be fp4 or nf4",
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enables fp16 training.",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enables bf16 training.",
        },
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use packing dataset creating.",
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enables gradient checkpointing.",
        },
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={
            "help": "The Optimizer to use.",
        },
    )
    lr_scheduler_type: Optional[str] = field(
        # default="cosine_with_warmup"
        default="cosine",
        metadata={
            "help": "Learning rate scheduler. Constant is a bit better than cosine, and has advantage for analysis.",
        },
    )
    max_steps: Optional[int] = field(
        default=1_000_000,
        metadata={
            "help": "How many optimizer update steps to take.",
        },
    )
    warmup_steps: Optional[int] = field(
        default=100,
        metadata={
            "help": "Number of steps to do a warmup for.",
        },
    )
    group_by_length: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length, Saves memory and speeds up training considerably.",
        },
    )
    save_steps: Optional[int] = field(
        default=200,
        metadata={
            "help": "Save checkpoint every X update steps.",
        },
    )
    logging_steps: Optional[int] = field(
        default=5,
        metadata={
            "help": "Log every X update steps.",
        },
    )
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Merge and push weights after training.",
        },
    )
    output_dir: Optional[str] = field(
        default="./results_packing",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written.",
        },
    )
    reports_to: Optional[
        Literal[
            "azure_ml",
            "clearml",
            "codecarbon",
            "comet_ml",
            "dagshub",
            "dvclive",
            "flyte",
            "mlflow",
            "neptune",
            "tensorboard",
            "wandb",
        ]
    ] = field(default="wandb")
