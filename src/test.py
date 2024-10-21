import torch
from loguru import logger
from transformers.generation import GenerateDecoderOnlyOutput, TextStreamer

from src.config import ScriptArguments
from src.dataset import DataPointKeys, get_dataset
from src.model import create_and_prepare_model
from src.transform import batch_transform


def model_test(config: ScriptArguments, device: torch.device):
    """
    Characterizing Mechanisms for Factual Recall in Language Models.
    `https://arxiv.org/pdf/2310.15910`
    """
    model, tokenizer, _ = create_and_prepare_model(config, device=device)
    streamer = TextStreamer(tokenizer)

    test_dataset = get_dataset(config)
    test_loader = torch.utils.data.DataLoader(
        shuffle=False,
        dataset=test_dataset,
        batch_size=config.per_device_test_batch_size,
        collate_fn=lambda x: batch_transform(x, requested_max_words=config.requested_max_words),
    )

    for sample in test_loader:
        articles = sample.get(DataPointKeys.ARTICLE)
        abstracts = sample.get(DataPointKeys.ABSTRACT)

        if articles is None:
            logger.error("Received `articles` as None")
            continue

        if abstracts is None:
            logger.error("Received `abstracts` as None")
            continue

        inputs = tokenizer(
            articles,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device=device)

        full_input_texts = tokenizer.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=True,
        )

        with torch.no_grad():
            outputs: GenerateDecoderOnlyOutput = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens_to_generate_for_abstract,
                num_return_sequences=1,
                output_logits=True,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
                repetition_penalty=config.repetition_penalty,
                streamer=streamer,
            )

        full_output_texts = tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )

        logger.info("output runtime type: {}".format(type(outputs).__name__))
        logger.info("output.sequences runtime shape: {}".format(outputs.sequences.shape))
        logger.info(outputs.sequences[:, inputs["input_ids"].shape[1] :].shape)

        # text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # print(full_output_texts[0])
        # print("~" * 120)

        for full_input, full_output in zip(full_input_texts, full_output_texts):
            generated_text = full_output[len(full_input) :].strip()
            print(generated_text)