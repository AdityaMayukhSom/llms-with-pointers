import os
import torch
import huggingface_hub

from loguru import logger
from dotenv import load_dotenv
from transformers import HfArgumentParser

from src.train import model_train
from src.eval import model_eval
from src.config import ScriptArguments


if __name__ == "__main__":
    load_dotenv()
    torch.manual_seed(42)

    if torch.cuda.is_available():
        logger.info("GPU with CUDA available.")
    else:
        logger.info("Could not find any CUDA supported device.")

    __parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = __parser.parse_args_into_dataclasses()[0]

    huggingface_hub.login(os.getenv("HF_ACCESS_TOKEN"))

    match (script_args.mode):
        case "train":
            model_train(script_args)
        case "test":
            raise NotImplementedError("'test' functionality not yet implemented.")
        case "eval":
            model_eval(script_args)
        case _:
            raise NotImplementedError("mode can only be train, text or eval.")
