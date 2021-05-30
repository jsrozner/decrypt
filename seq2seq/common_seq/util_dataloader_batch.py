from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from common_seq.types import *
from common_seq.util_dataloader import DatasetConfig, DataLoaderConfig

log = logging.getLogger(__name__)


class ClueDataLoaderBatched(DataLoader):
    dataset: ClueDatasetBatched

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__post_init_check()

    # @property
    def num_examples(self):
        return len(self.dataset)

    def __post_init_check(self):
        for idx, batch in enumerate(self):
            if idx > 0:
                break
            inputs = batch["source_text"]
            targets = batch["target_text"]
            log.info(f'Dataloader:\n\t'
                     f'{inputs[0]} => {targets[0]}')


@dataclass
class DataSetEntry:
    src: str
    tgt: str
    idx: Optional[int]


class ClueDatasetBatched(Dataset):
    def __init__(self,
                 dataset_config: DatasetConfig,
                 data_dir: str,
                 type_path):
        """
        max_examples: if > 0 then will load only max_examples into the dataset
        """
        self.use_json = False
        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"
        log.info(f'Loading cluedatasetbatched of type {type_path}')

        self.example_path = Path(data_dir) / type_path
        self.max_examples = dataset_config.max_examples

        # metric scoring additions
        # self.orig_clue_idxs = None  # the original indices of the examples; set in _build

        # populated later
        self._len = None  # the total number of examples

        self.data_list: Optional[List[DataSetEntry]] = None

        if type_path == "train":  # hacky way to print only once per dataset, since we always have train
            try:
                with open(Path(data_dir) / "README.txt") as f:
                    log.info('For dataset, found readme: ')
                    log.info(f.readlines())
            except:
                log.info('No readme for dataset')
                pass
        self._build()  # fill inputs, targets, max_lens

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.data_list[index]

    def _build_from_json(self):
        path = self.example_path.with_suffix(".json")

        with open(path, 'r') as f:
            all_json = json.load(f)
            ex_ct = len(all_json)
            if self.max_examples > 0:
                ex_ct = min(self.max_examples, ex_ct)

            self._len = ex_ct
            self.data_list = all_json[:ex_ct]

    def _build(self):
        if os.path.isfile(self.example_path.with_suffix(".json")):
            # log.info('Json files found, so using them')
            self.use_json = True
            self._build_from_json()
        else:
            raise NotImplementedError(f'No json files found at {self.example_path}')

    @classmethod
    def from_config(cls, cfg: DatasetConfig,
                    data_dir: str,
                    type_path: str):
        return cls(dataset_config=cfg,
                   data_dir=data_dir,
                   type_path=type_path)

pretokenize_fn = Callable[[List[Dict]], Tuple[List,...]]

def default_pretokenize(batch_list: List[Dict]) -> Tuple[List,...]:
    src_text = [e['input'] for e in batch_list]
    tgt_text = [e['target'] for e in batch_list]
    idxs = [e['idx'] for e in batch_list]
    return src_text, tgt_text, idxs

def default_collate_fn_json(tokenizer: PreTrainedTokenizerFast, batch_list: List[Dict],
                            pre_tokenize_fn: pretokenize_fn = None) -> Dict:
    if pre_tokenize_fn is not None:
        src_text, tgt_text, idxs = pre_tokenize_fn(batch_list)
    else:
        src_text, tgt_text, idxs = default_pretokenize(batch_list)

    tokenized_inputs = tokenizer(src_text, padding='longest', return_tensors='pt')
    tokenized_outputs = tokenizer(tgt_text, padding='longest', return_tensors='pt')

    source_ids = tokenized_inputs["input_ids"]
    target_ids = tokenized_outputs["input_ids"]
    src_mask = tokenized_inputs["attention_mask"]      # might need to squeeze
    target_mask = tokenized_outputs["attention_mask"]  # might need to squeeze

    # We cast these to torch.long in preprocess batch in trainer (# todo: is this right?)
    ret = {"source_ids": source_ids,
           "source_mask": src_mask,
           "target_ids": target_ids,
           "target_mask": target_mask,
           "source_text": src_text,
           "target_text": tgt_text,
           "idxs": idxs}

    return ret

def _get_dataloader_from_dataset(
        tokenizer,
        dataset: ClueDatasetBatched,
        dl_config: DataLoaderConfig,
        inputted_collate_fn: collate_fn_type) \
    -> ClueDataLoaderBatched:
        # inputted_collate_fn: Optional[Callable[[PreTrainedTokenizerFast, List[DataSetEntry]], Dict]] = None) \

    # take care of currying the appropriate collation function
    if dl_config.use_json:
        default_coll = default_collate_fn_json
    else:
        raise NotImplemented
        # default_coll = default_collate_fn
    if inputted_collate_fn is not None:
        def curried_collate_fn(input_list) -> Dict:
            return inputted_collate_fn(tokenizer, input_list)
    else:
        def curried_collate_fn(input_list) -> Dict:
            return default_coll(tokenizer, input_list)
    collate_fn = curried_collate_fn

    dataloader = ClueDataLoaderBatched(dataset,
                                       batch_size=dl_config.batch_size,
                                       shuffle=dl_config.shuffle,
                                       num_workers=dl_config.num_workers,
                                       collate_fn=collate_fn)
    log.info(f'Dataloader loaded from dataset')
    return dataloader


def _get_dataloader_batched(
        tokenizer,
        dataset_config: DatasetConfig,
        dl_config: DataLoaderConfig,
        data_dir,
        type_path: str,
        label_fn: Optional[Callable] = None,
        clue_to_idx_map=None,
        inputted_collate_fn: Optional[Callable[[PreTrainedTokenizerFast, List[DataSetEntry]], Dict]] = None) \
        -> ClueDataLoaderBatched:

    if label_fn is not None:
        # needed because we need the token offsets
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # set up dataset
    data_set = ClueDatasetBatched(dataset_config,
                                  data_dir=data_dir,
                                  type_path=type_path)
    log.info(f'Dataset {type_path} loaded with size: {len(data_set)}')

    # setup dataloader
    return _get_dataloader_from_dataset(tokenizer, data_set, dl_config, inputted_collate_fn)


def get_dataloaders_batched(tokenizer,
                            dataset_config_train: DatasetConfig,
                            dataset_config_val: DatasetConfig,
                            dl_config_train: DataLoaderConfig,
                            dl_config_val: DataLoaderConfig,
                            data_dir,
                            label_fn: Optional[Callable] = None,
                            clue_to_idx_map: Optional[Dict[str, int]] = None,
                            collate_fns: Optional[List[Callable]] = None,
                            use_test_set: bool = False) -> Tuple[ClueDataLoaderBatched, ClueDataLoaderBatched]:
    """

    :param tokenizer:
    :param dataset_config:
    :param dl_config_train:
    :param dl_config_val:
    :param data_dir:
    :param label_fn:
    :param clue_to_idx_map:
    :param collate_fns: Two collate functions, one for each of the dataloaders
    :return:
    """
    if collate_fns is None:
        collate_fns = [None, None]
    assert len(collate_fns) == 2
    train_loader = _get_dataloader_batched(tokenizer,
                                           dataset_config_train,
                                           dl_config_train,
                                           data_dir,
                                           type_path="train",
                                           label_fn=label_fn,
                                           clue_to_idx_map=clue_to_idx_map,
                                           inputted_collate_fn=collate_fns[0])
    if use_test_set:
        val_path = "test"
    else:
        val_path = "val"
    eval_loader = _get_dataloader_batched(tokenizer,
                                          dataset_config_val,
                                          dl_config_val,
                                          data_dir,
                                          type_path=val_path,
                                          label_fn=label_fn,
                                          clue_to_idx_map=clue_to_idx_map,
                                          inputted_collate_fn=collate_fns[1])

    return train_loader, eval_loader
