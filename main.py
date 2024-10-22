import os

import huggingface_hub
import torch
from dotenv import load_dotenv
from loguru import logger
from transformers import HfArgumentParser

from src.config import ScriptArguments
from src.eval import model_eval
from src.test import model_test
from src.train import model_train

if __name__ == "__main__":
    load_dotenv()
    torch.manual_seed(42)

    device: torch.device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU with CUDA available.")
    else:
        device = torch.device("cpu")
        logger.info("Could not find any CUDA supported device.")

    __parser = HfArgumentParser(ScriptArguments)
    config: ScriptArguments = __parser.parse_args_into_dataclasses()[0]

    huggingface_token = os.getenv("HF_ACCESS_TOKEN")

    if huggingface_token is None:
        huggingface_token = input("please enter huggingface token manually: ")

    huggingface_hub.login(huggingface_token)

    match (config.mode):
        case "train":
            model_train(config, device)
        case "test":
            model_test(config, device)
        case "eval":
            model_eval(config, device)
        case _:
            raise NotImplementedError("mode can only be train, text or eval.")
