from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity
    and faetures are and what size of model you want to train.
    """

    mode: Literal["train", "test", "eval"] = field(
        metadata={
            "help": "Whether to train, test or eval the model.",
        },
    )

    model_name: str = field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        metadata={
            "help": "The model that you want to train from HuggingFace Hub. E.g. GPT2, BERT, GPT2-XL etc.",
        },
    )

    do_streaming_while_generating: bool = field(
        default=False,
        metadata={
            "help": "Whether to stream the model's generated output or not. Used only when `mode` is `eval` and `source` is `manual`."
        },
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Please provide the root directory where training data is kept according to README.md",
        },
    )

    max_result_writers: Optional[int] = field(
        default=64,
        metadata={
            "help": "Maximum processes or threads to write results.",
        },
    )

    train_checkpoints_dir: str = field(
        default="./results_packing",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written.",
        },
    )

    test_result_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to generate model test results. Must be provided in `test` mode",
        },
    )

    eval_source: Optional[Literal["manual", "file"]] = field(
        default=None,
        metadata={
            "help": "Specify the input source for the article. In 'eval' mode, source must be either 'manual' for direct input or 'file' for reading from a text file."
        },
    )

    eval_article_filepath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the text file containing the article. Optional, used when 'mode' is 'eval' and 'source' is set to 'file'. If not provided upfront, will be asked during program execution."
        },
    )

    eval_abstract_filepath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the text file containing the abstract. Optional, used when 'mode' is 'eval' and 'source' is set to 'file'. If not provided upfront, will be asked during program execution."
        },
    )

    per_device_train_batch_size: int = field(default=1)
    per_device_test_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=17)
    learning_rate: float = field(default=3e-4)
    max_grad_norm: float = field(default=0.3)
    weight_decay: int = field(default=0.01)
    lora_alpha: int = field(default=16)
    lora_dropout: int = field(default=0.0)
    lora_r: int = field(default=16)
    max_seq_length: int = field(default=256)

    local_rank: int = field(
        default=-1,
        metadata={
            "help": "Used for Multi GPU.",
        },
    )

    use_4bit: bool = field(
        default=True,
        metadata={
            "help": "Activate 4 bit precision base model loading.",
        },
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={
            "help": "Activate nested quantization for 4bit base models.",
        },
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "Compute dtype for 4 bit base model.",
        },
    )
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = field(
        default="nf4",
        metadata={
            "help": "Quantization type, can be fp4 or nf4",
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Enables fp16 training.",
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "Enables bf16 training.",
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Use packing dataset creating.",
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Enables gradient checkpointing.",
        },
    )
    optim: str = field(
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
    max_steps: int = field(
        default=1_000_000,
        metadata={
            "help": "How many optimizer update steps to take.",
        },
    )
    warmup_steps: int = field(
        default=100,
        metadata={
            "help": "Number of steps to do a warmup for.",
        },
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length, Saves memory and speeds up training considerably.",
        },
    )
    save_steps: int = field(
        default=200,
        metadata={
            "help": "Save checkpoint every X update steps.",
        },
    )
    logging_steps: int = field(
        default=5,
        metadata={
            "help": "Log every X update steps.",
        },
    )
    merge_and_push: bool = field(
        default=False,
        metadata={
            "help": "Merge and push weights after training.",
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

    # ~~~~~~~~~~~~~~~~~~~~~~ Generation Configurations ~~~~~~~~~~~~~~~~~~~~~~

    requested_max_words: int = field(
        default=80,
        metadata={
            "help": "This number specifies the target word count for the abstract that the model should aim to generate."
        },
    )

    max_tokens_to_generate_for_abstract: int = field(
        default=120,
        metadata={
            "help": "The maximum number of tokens the model is allowed to generate for the abstract. This value includes special tokens and may exceed the target word count."
        },
    )

    repetition_penalty: float = field(
        default=1.2,
        metadata={
            "help": "A penalty applied to discourage the model from repeating the same tokens during generation. Values greater than 1.0 increase the penalty."
        },
    )
