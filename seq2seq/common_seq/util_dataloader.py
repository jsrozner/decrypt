from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)

# todo: we should not doubly track the warmup state both in multitask trainer and in trainer


@dataclass
class DataLoaderConfig:
    """
    Config to be used for setting up a DataLoader
    """
    shuffle: bool = True
    batch_size: int = 64
    num_workers: int = 4

    use_json: bool = False


@dataclass
class DatasetConfig:
    """
    Config to be used for setting up DataSet
    """
    tokenizer: PreTrainedTokenizer
    max_examples: int = -1  # if not -1, will truncate
    # src_len: int = 100
    # tgt_len: int = 20


# support json encoding of dataclass
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif callable(o):
            return o.__name__
        return super().default(o)
