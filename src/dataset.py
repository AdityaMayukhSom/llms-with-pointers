import os
import re
from typing import Any, Callable, Dict, List

from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate

from src.constants import DataPointKeys

INSTRUCT_PROMPT_TEMPLATE = """\
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_message}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

SYSTEM_MESSAGE = """\
You are an expert in reading articles and coming up with brief, yet clear and informative abstracts.
Only return a generated abstract. Do not provide any explanations or unnecessary words. 
Do not provide anything which is not necessary or related to the abstract. Just provide the abstract."""

USER_MESSAGE_TEMPLATE = """\
Summarize this following article under {max_words} words:

{article}"""


def extract_user_message(text: str) -> str:
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def generate_prompt_from_article(article: str, requested_max_words: int):
    return INSTRUCT_PROMPT_TEMPLATE.format(
        system_message=SYSTEM_MESSAGE,
        user_message=USER_MESSAGE_TEMPLATE.format(
            article=article,
            max_words=requested_max_words,
        ),
    )


def datapoint_transform(datapoint: Dict[str, str]):
    return datapoint


def batch_transform(batch: List[Dict[str, Any]], requested_max_words: int) -> Dict[str, List]:
    # article_type = type(batch[0][DataPointKeys.ARTICLE]).__name__
    # logger.info("Type of `article` before transformation {}".format(article_type))

    for elem in batch:
        elem[DataPointKeys.ARTICLE] = bytes(elem[DataPointKeys.ARTICLE]).decode("utf-8")
        elem[DataPointKeys.ABSTRACT] = bytes(elem[DataPointKeys.ABSTRACT]).decode("utf-8")

        elem[DataPointKeys.ARTICLE] = generate_prompt_from_article(
            elem[DataPointKeys.ARTICLE],
            requested_max_words=requested_max_words,
        )

    # article_type = type(batch[0][DataPointKeys.ARTICLE]).__name__
    # logger.info("Type of `article` before transformation {}".format(article_type))

    return default_collate(batch)


def get_dataset(
    data_filename: str,
    index_filename: str,
    base_data_directory: str,
    transform_fn: Callable[[Dict], Any] | None = None,
) -> IterableDataset:
    if not os.path.exists(base_data_directory):
        raise FileNotFoundError("Path {} does not exist.")

    if not not os.path.isdir(base_data_directory):
        raise ValueError("Path {} is not a directory.")

    data_path = os.path.join(base_data_directory, data_filename)
    index_path = os.path.join(base_data_directory, index_filename)

    description = {
        DataPointKeys.ARTICLE: "byte",
        DataPointKeys.ABSTRACT: "byte",
    }

    dataset: IterableDataset = TFRecordDataset(
        data_path=data_path,
        index_path=index_path,
        description=description,
        transform=transform_fn,
    )

    return dataset
