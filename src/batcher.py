from datasets import load_dataset
from src.config import ScriptArguments


def training_batch_generator(config: ScriptArguments):
    ds = load_dataset(path=config.dataset_name, streaming=True, split="train")

    for sample in iter(ds):
        # Extract instructions and inputs from the samples
        instruction = str(sample["instruction"])
        input_text = str(sample["input"])
        output_text = str(sample["output"])
        formatted_prompt = None

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
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                f"<|eot_id|><|start_header_id|>asssitant<|end_header_id|>\n\n"
                f"{str(output_text)}"
                f"<|eot_id|><|end_of_text|>"
            )

        formatted_prompt = "".join(formatted_prompt)
        yield {"text": formatted_prompt}
