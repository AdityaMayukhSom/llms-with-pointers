import enum
from typing import Any, Callable, Dict

from loguru import logger
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import IterableDataset

from src.config import ScriptArguments


@enum.unique
class DataPointKeys(enum.StrEnum):
    ARTICLE = "article"
    ABSTRACT = "abstract"


def get_dataset(
    config: ScriptArguments,
    transform_fn: Callable[[Dict], Any] | None = None,
) -> IterableDataset:
    # TODO: make the data path dynamic
    tfrecord_path = "./data/single/tfrecord/test.tfrecord"
    index_path = "./data/single/tfindex/test.tfindex"

    description = {
        DataPointKeys.ARTICLE: "byte",
        DataPointKeys.ABSTRACT: "byte",
    }

    dataset: IterableDataset = TFRecordDataset(
        data_path=tfrecord_path,
        index_path=index_path,
        description=description,
        transform=transform_fn,
    )

    return dataset
