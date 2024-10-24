import os
from typing import Any, Callable, Dict, List

from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate

from src.constants import (
    INSTRUCT_PROMPT_TEMPLATE,
    SYSTEM_MESSAGE,
    USER_MESSAGE_TEMPLATE,
    DataPointKeys,
)


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
        # The articles and abstracts loaded in the batch have `bytes` as their element
        # type, but the tokenizer needs `str` as the element type to work, hence we need
        # to convert each article and abstract into string before passing to tokenizer.
        elem[DataPointKeys.ARTICLE] = bytes(elem[DataPointKeys.ARTICLE]).decode("utf-8")
        elem[DataPointKeys.ABSTRACT] = bytes(elem[DataPointKeys.ABSTRACT]).decode("utf-8")

        # This key did not exist in the original `elem` in the datapoint, but is being added
        # so that article can be extracted without explicit parsing of generated outtput later.
        elem[DataPointKeys.PROMPT] = generate_prompt_from_article(
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
        raise FileNotFoundError("Path {} does not exist.".format(base_data_directory))

    if not os.path.isdir(base_data_directory):
        raise NotADirectoryError("Path {} is not a directory.".format(base_data_directory))

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
