import torch
from transformers.generation import GenerateDecoderOnlyOutput, TextStreamer

from src.config import ScriptArguments
from src.dataset import generate_prompt_from_article
from src.model import create_and_prepare_model
from src.utils import extract_user_message


def model_eval(config: ScriptArguments, device: torch.device):
    if config.mode == "eval" and config.source is None:
        raise ValueError("In 'eval' mode, 'source' must be provided.")

    article_filepath = config.eval_article_filepath
    abstract_filepath = config.eval_abstract_filepath

    write_abstract_to_file: bool = False  # whether to write the generated abstract to the config path or not

    if config.eval_source == "manual":
        write_abstract_to_file = False
        article = input("Enter Article: ")

    elif config.eval_source == "file":
        write_abstract_to_file = True

        if article_filepath is None:
            article_filepath = input("Enter Article File Path: ")

        if abstract_filepath is None:
            abstract_filepath = input("Enter Abstract File Path: ")

        with open(article_filepath, "r", encoding="utf-8") as article_file:
            article = article_file.read()

    else:
        raise ValueError("could not find respective action to perform")

    model, tokenizer, _ = create_and_prepare_model(config, device=device)
    prompt = [generate_prompt_from_article(article, requested_max_words=config.requested_max_words)]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device=device)

    outputs: GenerateDecoderOnlyOutput = model.generate(
        **inputs,
        max_new_tokens=config.max_tokens_to_generate_for_abstract,
        num_return_sequences=1,
        output_logits=True,
        output_scores=True,
        output_attentions=True,
        return_dict_in_generate=True,
        streamer=(
            None if write_abstract_to_file or not config.do_streaming_while_generating else TextStreamer(tokenizer)
        ),
    )

    full_input_texts = tokenizer.batch_decode(
        inputs["input_ids"],
        skip_special_tokens=True,
    )

    full_output_texts = tokenizer.batch_decode(
        outputs.sequences,
        skip_special_tokens=True,
    )

    user_article = extract_user_message(prompt[0])
    generated_text = full_output_texts[0][len(full_input_texts[0]) :]

    if write_abstract_to_file and abstract_filepath is not None:
        with open(abstract_filepath, "w", encoding="utf-8") as abstract_file:
            abstract_file.write(generated_text)

    if not write_abstract_to_file:
        print("~~~~~~~~ Article ~~~~~~~~")
        print(user_article)
        print("~~~~~~~~ Summary ~~~~~~~~")
        print(generated_text)
