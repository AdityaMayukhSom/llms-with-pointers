import pprint
from typing import Any, Callable, Dict

from loguru import logger
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import IterableDataset

from src.config import ScriptArguments


def get_dataset(config: ScriptArguments, transform_fn: Callable[[Dict], Any] | None = None) -> IterableDataset:
    tfrecord_path = "./data/single/tfrecord/test.tfrecord"
    index_path = "./data/single/tfindex/test.tfindex"
    description = {
        "article": "byte",
        "abstract": "byte",
    }
    dataset: IterableDataset = TFRecordDataset(tfrecord_path, index_path, description, transform=transform_fn)

    return dataset
