import torch
from transformers import HfArgumentParser
from src.config import ScriptArguments

torch.manual_seed(42)

if __name__ == "__main__":
    __parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = __parser.parse_args_into_dataclasses()[0]
    match (script_args.mode):
        case "train":
            print("train")
        case "test":
            print("train")
        case "eval":
            print("train")
        case _:
            print("other")
