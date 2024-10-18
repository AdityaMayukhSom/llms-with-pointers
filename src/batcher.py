from datasets import load_dataset
from src.args import script_args

def get_batches_train():
    ds = load_dataset(script_args.dataset_name, streaming=True, split='train')

    for sample in iter(ds):
        # Extract instructions and inputs from the samples
        instruction = str(sample['instruction'])
        input_text = str(sample['input'])
        output_text = str(sample['output'])
        formatted_prompt = None

        if input_text is None or input_text == "":
            formatted_prompt = (f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Input:
{input_text}

### Response: 
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{str(output_text)}
<|eot_id|>
<|end_of_text|>
"""
            )

                    
