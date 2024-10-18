import torch
from transformers import HfArgumentParser
from src.train import train
from src.config import ScriptArguments

torch.manual_seed(42)

if __name__ == "__main__":
    __parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = __parser.parse_args_into_dataclasses()[0]
    match (script_args.mode):
        case "train":
            train(script_args)
        case "test":
            raise NotImplementedError("'test' functionality not yet implemented.")
        case "eval":
            raise NotImplementedError("'eval' functionality not yet implemented.")
        case _:
            raise NotImplementedError("mode can only be train, text or eval.")
