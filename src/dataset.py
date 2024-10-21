import torch
from datasets import Dataset, IterableDataset, load_dataset
from loguru import logger
from torchdata.datapipes.iter import FileLister, FileOpener, TFRecordLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.config import ScriptArguments


def batch_generator(config: ScriptArguments):
    file_lister_dp = FileLister("./data/chunked", "test_*.tfrecords")
    file_opener_dp = FileOpener(file_lister_dp, mode="b")
    tfrecord_loader_dp = TFRecordLoader(file_opener_dp)
    # ds = load_dataset(path=config.dataset_name, streaming=True, split="train")

    for sample in tfrecord_loader_dp:
        logger.info(sample)
        # Extract instructions and inputs from the samples
        # instruction = str(sample["instruction"])
        # input_text = str(sample["input"])
        # output_text = str(sample["output"])
        # formatted_prompt = None

        # # "<|im_start|>user\n" + x["prompt"] + " <|im_end|>\n<|im_start|>assistant\n" + x["response"] + "<|im_end|>\n"
        # base_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n {system_message} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # if input_text is None or input_text == "":
        #     formatted_prompt = (
        #         f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        #         f"Below is an instruction that describes a task.",
        #         f"Write a response that appropriately completes the request.\n\n",
        #         f"### Instruction:\n{instruction}\n\n",
        #         f"### Response:\n",
        #         f"<|eot_id|><|start_header_id|>asssitant<|end_header_id|>\n\n",
        #         f"{str(output_text)}",
        #         f"<|eot_id|><|end_of_text|>",
        #     )
        # else:
        #     formatted_prompt = (
        #         f"<|begin_of_text|>",
        #         f"<|start_header_id|>user<|end_header_id|>\n\n",
        #         f"Below is an instruction that describes a task. ",
        #         f"Write a response that appropriately completes the request.\n\n",
        #         f"### Instruction:\n{instruction}\n\n",
        #         f"### Input:\n{input_text}\n\n",
        #         f"### Response:\n",
        #         f"<|eot_id|><|start_header_id|>asssitant<|end_header_id|>\n\n",
        #         f"{str(output_text)}",
        #         f"<|eot_id|><|end_of_text|>",
        #     )

        # formatted_prompt = "".join(formatted_prompt)
        yield {
            "text": "hello world",
        }


# def _parse_function(example_proto):
#     # Create a description of the features.
#     feature_description = {
#         "article": tf.io.FixedLenFeature([], tf.string, default_value=""),
#         "abstract": tf.io.FixedLenFeature([], tf.string, default_value=""),
#     }
#     # Parse the input `tf.Example` proto using the dictionary above.
#     parsed_example = tf.io.parse_single_example(example_proto, feature_description)
#     return parsed_example


# def example_generator(filenames, vocab, max_enc_len, max_dec_len, mode, batch_size):
#     raw_dataset = tf.data.TFRecordDataset(filenames)
#     parsed_dataset = raw_dataset.map(_parse_function)
#     if mode == "train":
#         parsed_dataset = parsed_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat()

#     for raw_record in parsed_dataset:

#         article = raw_record["article"].numpy().decode()  # type: ignore
#         abstract = raw_record["abstract"].numpy().decode()  # type: ignore

#         start_decoding = vocab.word_to_id(vocab.START_DECODING)
#         stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)

#         article_words = article.split()[:max_enc_len]
#         enc_len = len(article_words)
#         enc_input = [vocab.word_to_id(w) for w in article_words]
#         enc_input_extend_vocab, article_oovs = DataHelper.article_to_ids(article_words, vocab)

#         abstract_sentences = [sent.strip() for sent in DataHelper.abstract_to_sents(abstract)]
#         abstract = " ".join(abstract_sentences)
#         abstract_words = abstract.split()
#         abs_ids = [vocab.word_to_id(w) for w in abstract_words]
#         abs_ids_extend_vocab = DataHelper.abstract_to_ids(abstract_words, vocab, article_oovs)
#         dec_input, target = DataHelper.get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)
#         _, target = DataHelper.get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
#         dec_len = len(dec_input)

#         output = {
#             "enc_len": enc_len,
#             "enc_input": enc_input,
#             "enc_input_extend_vocab": enc_input_extend_vocab,
#             "article_oovs": article_oovs,
#             "dec_input": dec_input,
#             "target": target,
#             "dec_len": dec_len,
#             "article": article,
#             "abstract": abstract,
#             "abstract_sents": abstract_sentences,
#         }

#         if mode == "test" or mode == "eval":
#             for _ in range(batch_size):
#                 yield output
#         else:
#             yield output


# def old_batch_generator(generator, filenames,  batch_size: int):
#     dataset = tf.data.Dataset.from_generator(
#         lambda: generator(filenames, vocab, max_enc_len, max_dec_len, mode, batch_size),
#         output_types={
#             "enc_len": tf.int32,
#             "enc_input": tf.int32,
#             "enc_input_extend_vocab": tf.int32,
#             "article_oovs": tf.string,
#             "dec_input": tf.int32,
#             "target": tf.int32,
#             "dec_len": tf.int32,
#             "article": tf.string,
#             "abstract": tf.string,
#             "abstract_sents": tf.string,
#         },
#         output_shapes={
#             "enc_len": [],
#             "enc_input": [None],
#             "enc_input_extend_vocab": [None],
#             "article_oovs": [None],
#             "dec_input": [None],
#             "target": [None],
#             "dec_len": [],
#             "article": [],
#             "abstract": [],
#             "abstract_sents": [None],
#         },
#     )

#     def update(entry):
#         return (
#             {
#                 "enc_input": entry["enc_input"],
#                 "extended_enc_input": entry["enc_input_extend_vocab"],
#                 "article_oovs": entry["article_oovs"],
#                 "enc_len": entry["enc_len"],
#                 "article": entry["article"],
#             },
#             {
#                 "dec_input": entry["dec_input"],
#                 "dec_target": entry["target"],
#                 "dec_len": entry["dec_len"],
#                 "abstract": entry["abstract"],
#             },
#         )

#     dataset = dataset.map(update)
#     return dataset


def get_dataset(config: ScriptArguments):
    ds: Dataset | IterableDataset = Dataset.from_generator(lambda: batch_generator(config))
    return ds
