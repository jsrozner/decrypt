from typing import *

from transformers import PreTrainedTokenizerFast

collate_fn_type = Callable[[PreTrainedTokenizerFast, List[Dict]], Dict]
pretokenize_fn = Callable[[List[Dict]], Tuple[List, ...]]
