import hashlib
import os

# from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
from loguru import logger
from transformers.generation import GenerateDecoderOnlyOutput, TextStreamer

from src.config import ScriptArguments
from src.constants import DataPointKeys
from src.dataset import batch_transform, get_dataset
from src.model import create_and_prepare_model
from src.utils import extract_user_message


def save_test_results(full_input_texts: List[str], full_output_texts: List[str], result_dir: str):

    for full_input, full_output in zip(full_input_texts, full_output_texts):
        article = extract_user_message(full_input)
        abstract_generated = full_output[len(full_input) :].strip()

        h = hashlib.sha256(usedforsecurity=False)
        h.update(article.encode(encoding="utf-8"))

        article_filename = "{}_article.txt".format(h.hexdigest())
        abstract_filename = "{}_abstract.txt".format(h.hexdigest())

        with open(os.path.join(result_dir, article_filename), "w") as article_file:
            article_file.write(article)

        with open(os.path.join(result_dir, abstract_filename), "w") as abstract_file:
            abstract_file.write(abstract_generated)


def model_test(config: ScriptArguments, device: torch.device):
    """
    Characterizing Mechanisms for Factual Recall in Language Models.
    `https://arxiv.org/pdf/2310.15910`

    TODO: Handle randomness in dataloader.
    https://pytorch.org/docs/stable/notes/randomness.html
    """

    if config.mode == "test" and config.data_dir is None:
        raise ValueError("Please provide a directory with the test data following the structure outlined in README.md.")

    if config.mode == "test" and config.test_result_dir is None:
        raise ValueError("Please specify a directory where the test results will be stored.")

    model, tokenizer, _ = create_and_prepare_model(config, device=device)
    streamer = TextStreamer(tokenizer) if config.do_streaming_while_generating else None

    test_dataset = get_dataset(
        data_filename="single/tfrecord/test.tfrecord",
        index_filename="single/tfindex/test.tfindex",
        base_data_directory=config.data_dir,
    )

    test_loader = torch.utils.data.DataLoader(
        shuffle=False,
        dataset=test_dataset,
        batch_size=config.per_device_test_batch_size,
        collate_fn=lambda x: batch_transform(x, requested_max_words=config.requested_max_words),
    )

    # thread_pool_executor = ThreadPoolExecutor(max_workers=config.max_writer_processes)

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

        save_test_results(full_input_texts, full_output_texts, config.test_result_dir)

        # thread_pool_executor.submit(
        #     save_test_results,
        #     full_input_texts,
        #     full_output_texts,
        #     config.test_result_dir,
        # )

        # logger.info("output runtime type: {}".format(type(outputs).__name__))
        # logger.info("output.sequences runtime shape: {}".format(outputs.sequences.shape))
        # logger.info(outputs.sequences[:, inputs["input_ids"].shape[1] :].shape)
        # text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # print(full_output_texts[0])
        # print("~" * 120)

        del inputs
        del outputs

    # thread_pool_executor.shutdown(wait=True, cancel_futures=False)
