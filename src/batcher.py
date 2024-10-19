from datasets import load_dataset
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast
from src.config import ScriptArguments


def get_training_batch_generator(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, config: ScriptArguments):
    def training_batch_generator():
        ds = load_dataset(path=config.dataset_name, streaming=True, split="train")

        for sample in iter(ds):
            # Extract instructions and inputs from the samples
            instruction = str(sample["instruction"])
            input_text = str(sample["input"])
            output_text = str(sample["output"])
            formatted_prompt = None

            # "<|im_start|>user\n" + x["prompt"] + " <|im_end|>\n<|im_start|>assistant\n" + x["response"] + "<|im_end|>\n"

            if input_text is None or input_text == "":
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
                    f"Below is an instruction that describes a task.",
                    f"Write a response that appropriately completes the request.\n\n",
                    f"### Instruction:\n{instruction}\n\n",
                    f"### Response:\n",
                    f"<|eot_id|><|start_header_id|>asssitant<|end_header_id|>\n\n",
                    f"{str(output_text)}",
                    f"<|eot_id|><|end_of_text|>",
                )
            else:
                formatted_prompt = (
                    f"<|begin_of_text|>",
                    f"<|start_header_id|>user<|end_header_id|>\n\n",
                    f"Below is an instruction that describes a task. ",
                    f"Write a response that appropriately completes the request.\n\n",
                    f"### Instruction:\n{instruction}\n\n",
                    f"### Input:\n{input_text}\n\n",
                    f"### Response:\n",
                    f"<|eot_id|><|start_header_id|>asssitant<|end_header_id|>\n\n",
                    f"{str(output_text)}",
                    f"<|eot_id|><|end_of_text|>",
                )

            formatted_prompt = "".join(formatted_prompt)
            yield {
                "text": formatted_prompt,
            }

    return training_batch_generator
