import torch
from transformers.generation import GenerateDecoderOnlyOutput, TextStreamer

from src.config import ScriptArguments
from src.dataset import generate_prompt_from_article
from src.model import create_and_prepare_model
from src.utils import ResultsUtils


def model_eval(config: ScriptArguments, device: torch.device):
    if config.mode == "eval" and config.eval_type is None:
        raise ValueError("In 'eval' mode, 'eval_type' must be provided.")

    article_filepath = config.eval_article_filepath
    abstract_filepath = config.eval_abstract_filepath
    result_utils = ResultsUtils()

    write_abstract_to_file: bool = False  # whether to write the generated abstract to the config path or not

    if config.eval_type == "manual":
        write_abstract_to_file = False
        article = input("Enter Article: ")

    elif config.eval_type == "file":
        write_abstract_to_file = True

        if article_filepath is None:
            article_filepath = input("Enter Article File Path: ")

        if abstract_filepath is None:
            abstract_filepath = input("Enter Abstract File Path: ")

        with open(article_filepath, "r", encoding="utf-8") as article_file:
            article = article_file.read()

    else:
        raise ValueError(f"`eval_type` can either be `file` or `manual`, but `{config.eval_type}` was provided.")

    model, tokenizer, _ = create_and_prepare_model(config, device=device)

    # trial over different batch size can be done by setting the dynamic batch size variable
    # this feature is only for evaluation during development and doesn't bring any functionality
    # as we are only saving the zeroth index of eval text, hence only one eval text will be saved
    # that is the expected behaviour as we are simply multiplying same input multiple times.
    dynamic_batch_size = 1

    prompts = [generate_prompt_from_article(article, max_words=config.requested_max_words)] * dynamic_batch_size

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device=device)

    streamer = None if write_abstract_to_file or not config.do_streaming_while_generating else TextStreamer(tokenizer)

    outputs: GenerateDecoderOnlyOutput = model.generate(
        **inputs,
        max_new_tokens=config.max_tokens_to_generate_for_abstract,
        num_return_sequences=1,
        output_logits=True,
        output_scores=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        repetition_penalty=config.repetition_penalty,
        streamer=streamer,
    )

    full_input_texts = tokenizer.batch_decode(
        inputs["input_ids"],
        skip_special_tokens=True,
    )

    full_output_texts = tokenizer.batch_decode(
        outputs.sequences,
        skip_special_tokens=True,
    )

    del inputs
    del outputs

    generated_abstracts = result_utils.parse_llm_outputs(full_input_texts, full_output_texts)

    if write_abstract_to_file and abstract_filepath is not None:
        with open(abstract_filepath, "w", encoding="utf-8") as abstract_file:
            abstract_file.write(generated_abstracts[0])

    if not write_abstract_to_file:
        print("~~~~~~~~ Article ~~~~~~~~")
        print(article)
        print("~~~~~~~~ Summary ~~~~~~~~")
        print(generated_abstracts[0])
