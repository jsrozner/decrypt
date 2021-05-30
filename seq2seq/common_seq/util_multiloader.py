from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from .types import *
from .util_dataloader import DatasetConfig, DataLoaderConfig
from .util_dataloader_batch import ClueDataLoaderBatched, ClueDatasetBatched, _get_dataloader_from_dataset
from .util_metrics import MetricFcn
from .collate_fns import collate_fn_type

log = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """
    Used to create valloaderwithfunction
    """
    dir: str
    name: str
    val_fcn_list: Optional[List[MetricFcn]]  # if set, will split into val set
    collate_fn: Optional[collate_fn_type] = None

    # val_metrics_to_keep: Optional[List[List[str]]] = None

    # batch_size: int = 64
    # src_len: int = 20
    # tgt_len: int = 20


@dataclass
class ValLoaderWithFcn:
    """
    Created from a TaskConfig
    """
    name: str
    dataloader: ClueDataLoaderBatched
    val_fcn_list: Optional[List[Callable]]
    batch_size: int


@dataclass
class MultitaskConfig:
    """
    Set of TaskConfig
    Used to create a multitaskmanager
    """
    multitask_dir: str
    freq_list: List[int]  # should be one longer than the list of tasks
    reset: bool
    num_warmup: int

    tasks: List[TaskConfig]
    val_split_pct: float = 0.8


class MultitaskManager:
    def __init__(self, cfg: MultitaskConfig,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,           # use the same as the trainer that creates this
                 num_examples=-1,
                 use_json: bool = False,
                 collate_fn: Optional[Callable] = None):
        # check inputs
        assert len(cfg.tasks) + 1 == len(cfg.freq_list)
        self.cfg = cfg
        self.batch_size = batch_size
        self.use_json = use_json

        # used in prepare dataloaders; here for readability; all are the same
        # src len modified below in _prepare
        self.dataset_cfg = DatasetConfig(tokenizer=tokenizer,
                                         max_examples=num_examples)

        self.train_dls: Optional[List[ClueDataLoaderBatched]] = None
        self.val_dls: Optional[List[ValLoaderWithFcn]] = None
        self._prepare_dataloaders(collate_fn)

        self.train_multiloader = None

    @property
    def multitask_reset(self):
        return self.cfg.reset

    @property
    def multitask_warmup(self):
        return self.cfg.num_warmup

    def _prepare_dataloaders(self, collate_fn):
        """
        Initialization helper
        """
        train_dls: List[ClueDataLoaderBatched] = []
        val_dls: List[ValLoaderWithFcn] = []
        cfg = self.cfg

        folder = cfg.multitask_dir
        for task in cfg.tasks:
            path = Path(folder) / task.dir
            local_config = self.dataset_cfg
            if task.collate_fn is not None:
                assert collate_fn is None
                local_collate = task.collate_fn
                logging.info(f'For task {task.name}, using cfg-provided collate function')
            else:
                local_collate = collate_fn

            dl_list = self.dataloaders_from_path(
                dataset_cfg=local_config,
                dl_cfg=DataLoaderConfig(batch_size=self.batch_size,
                                        use_json=self.use_json),
                split_into_val=task.val_fcn_list is not None,
                data_dir=path,
                val_split_pct=cfg.val_split_pct,
                collate_fn=local_collate)

            # always produces a train
            train_dls.append(dl_list[0])

            # if cfg specifies that we validate, then also append the validation
            if task.val_fcn_list is not None:
                assert dl_list[1] is not None
                val_dls.append(ValLoaderWithFcn(
                    task.name, dl_list[1], task.val_fcn_list, self.batch_size))

        self.train_dls = train_dls
        self.val_dls = val_dls

    def get_train_multiloader(self,
                              primary_dl: ClueDataLoaderBatched,
                              warmup_iters: int) -> MultiTaskDataLoader:
        if self.train_multiloader is not None:
            raise ValueError('get_train_multiloader should be called only once')
        train_loader_list = [primary_dl] + self.train_dls
        self.train_multiloader = MultiTaskDataLoader(train_loader_list,
                                                     freq_list=self.cfg.freq_list,
                                                     warmup_iters=warmup_iters)
        return self.train_multiloader

    @classmethod
    def dataloaders_from_path(cls,
                              dataset_cfg: DatasetConfig,
                              dl_cfg: DataLoaderConfig,
                              split_into_val: bool,
                              data_dir: str,
                              val_split_pct: float,
                              collate_fn: collate_fn_type) -> List[ClueDataLoaderBatched]:
        """
        Produce 1 or two dataloaders (train/ val) from a path

        :param dataset_cfg:
        :param dl_cfg:
        :param split_into_val:
        :param data_dir:
        :param val_split_pct:
        :param collate_fn:
        :return:
        """
        if not dl_cfg.shuffle:
            raise NotImplemented('Does not support not shuffling the multitask datasets')

        # default to loading from type_path = train
        dataset = ClueDatasetBatched.from_config(dataset_cfg,
                                                 data_dir=data_dir,
                                                 type_path="train")
        if split_into_val:
            train_len = math.floor(val_split_pct * len(dataset))
            test_len = len(dataset) - train_len
            ds_list = torch.utils.data.random_split(dataset, [train_len, test_len],
                                                    generator=torch.Generator().manual_seed(42))
        else:
            ds_list = [dataset, None]

        # always a len 2 list; val is second one, potentially None
        output_list: List[Optional[ClueDataLoaderBatched]] = []
        for idx, ds in enumerate(ds_list):
            if ds is None:
                output_list.append(None)
                continue

            dl_cfg.shuffle = (True if idx == 0 else False)  # shuffle train loader only
            dl = _get_dataloader_from_dataset(dataset_cfg.tokenizer,
                                              ds,
                                              dl_cfg,
                                              inputted_collate_fn=collate_fn)
            output_list.append(dl)

        return output_list


class MultiTaskDataLoader:
    def __init__(self, dataloaders: List[ClueDataLoaderBatched],
                 freq_list: List[int],
                 warmup_iters=1):
        """
        Produces a single dataloader that can do multidataset iteration.

        When the system is in warmup, skips the first dataset and just loads datasets[1:]
        When the system is in normal train, yields batches from the given dataloaders at the freq_list rate (e.g. 20,2)
                will produce 20 batches from the main set and then 2 from the first multitrain set

        For warmup, always does batches in 1:1 ratio

        :param dataloaders:
        :param freq_list:
        :param warmup_iters:
        """
        log.info(f'Configuring multiloader with freqs {freq_list} batches')

        # todo: need to pass in batch sizes for accurate computations of num train
        self.warmup_iters = warmup_iters
        self.num_iters = 0          # iters so far; probably don't want to keep state in two places
                                    # but, we only increment if the dataloader is fully processed and otherwise
                                    # raise an exception, so this is probably safe enough

        # get batch_sizes
        self.dl_batch_sizes: List[int] = []
        self.dataloaders_warmup: List[ClueDataLoaderBatched] = []
        self.dataloaders_train: List[ClueDataLoaderBatched] = []
        self.warmup_freq_list: List[int] = []
        self.train_freq_list: List[int] = []

        self._setup(dataloaders, freq_list)

        # iterators (set by iter())
        # current DL we are on; will be None if we have not called iter()
        self.dataloader_idx: Optional[int] = None
        self.dataloader_iters: Optional[List] = None
        # how many batches we have processed in the current loader
        self.batch_ct: Optional[int] = None

    # self.dls, self.fl

    def _setup(self, dataloaders, freq_list):
        assert len(dataloaders) == len(freq_list)
        self.dataloaders_warmup = dataloaders[1:]
        # originally assumed all warmup the same; now we say do in same measure as the
        # frequency list
        # self.warmup_freq_list = [1] * len(self.dataloaders_warmup)
        self.warmup_freq_list = freq_list[1:]

        for dl, freq in zip(dataloaders, freq_list):
            if freq == 0:  # skip a dataloader if it has freq 0 (i.e. its just used in warmup
                raise NotImplemented
                # continue
            batch_size = len(next(iter(dl))["source_ids"])
            self.dl_batch_sizes.append(batch_size)
            self.dataloaders_train.append(dl)
            self.train_freq_list.append(freq)

        log.info(f'Finished setting up multiloader\n'
                 f'\t batch_sizes: {self.dl_batch_sizes}\n'
                 f'\t freq: {self.train_freq_list}')

    def __len__(self):
        raise NotImplemented

    # todo: these will be slightly off bc we don't do drop last - so batchsize multiplication
    # doesn't quite work
    # @property
    def num_examples(self) -> int:
        if self.num_iters < self.warmup_iters:
            # assumes all batch sizes are the same
            bs = self.dataloaders_warmup[0].batch_size

            # take ceiling bc we do take last batch
            batches_primary = self.dataloaders_warmup[0].num_examples() / bs
            actual_batches = math.ceil(batches_primary)

            # how many times we run through all dataloaders
            iters_primary = actual_batches // self.warmup_freq_list[0]

            total_batches_per_iter_non_primary = sum(self.warmup_freq_list[1:])
            total_batches_non_primary = total_batches_per_iter_non_primary * iters_primary

            # this will be slightly off depending on the final batch
            return total_batches_non_primary * bs + self.dataloaders_warmup[0].num_examples()

        else:
            bs_primary = self.dataloaders_train[0].batch_size
            bs_multi = self.dataloaders_warmup[0].batch_size

            # take ceiling bc we do take last batch
            batches_primary = self.dataloaders_train[0].num_examples() / bs_primary
            actual_batches = math.ceil(batches_primary)

            # how many times we run through all dataloaders
            iters_primary = actual_batches // self.train_freq_list[0]

            total_batches_per_iter_non_primary = sum(self.train_freq_list[1:])
            total_batches_non_primary = total_batches_per_iter_non_primary * iters_primary

            # this will be slightly off depending on the final batch
            return total_batches_non_primary * bs_multi + self.dataloaders_train[0].num_examples()

    def __next__(self):
        # Note that iter() is called to set this up - read the code there first
        if self.dataloader_idx is None:
            raise ValueError(f'Iterator not initialized. Call iter() first')

        # if we have processed all of a given DL, then increment the DL index
        if self.batch_ct == self.fl[self.dataloader_idx]:
            self.dataloader_idx = (self.dataloader_idx + 1) % len(self.dataloader_iters)
            self.batch_ct = 0

        # first DL is special - it limits what we do
        if self.dataloader_idx == 0:
            # if it raises StopIter, then pass it on since we are done
            try:
                # ret = (next(self.dataloader_iters[self.dataloader_idx]),
                #        self.dl_batch_sizes[self.dataloader_idx])
                ret = next(self.dataloader_iters[self.dataloader_idx])
            except StopIteration:
                self._end_iters()
                raise StopIteration
        # otherwise we wrap around and continue iterating through the dataset
        else:
            try:
                # ret = (next(self.dataloader_iters[self.dataloader_idx]), 0)
                ret = next(self.dataloader_iters[self.dataloader_idx])
            except StopIteration:
                # reset this iterator - we need to keep running it until we finish the first iterator
                self.dataloader_iters[self.dataloader_idx] = iter(self.dls[self.dataloader_idx])
                # continues from this iterator (this should never produce a stop)
                # todo: maybe should call self.next() rather than duplpicating here
                ret = next(self.dataloader_iters[self.dataloader_idx])

        self.batch_ct += 1
        return ret

    def _end_iters(self):
        self.dataloader_idx = None
        self.dataloader_iters = None
        self.batch_ct = None

    def __iter__(self):
        if self.dataloader_iters is not None:
            raise NotImplemented(f'Reinitializing loader that was not done')

        if self.num_iters < self.warmup_iters:
            self.dls = self.dataloaders_warmup
            self.fl = self.warmup_freq_list
        else:
            self.dls = self.dataloaders_train
            self.fl = self.train_freq_list

        self.dataloader_iters = list(map(iter, self.dls))
        self.dataloader_idx = 0  # start from the data
        self.batch_ct = 0

        # increment that we have done one epoch
        self.num_iters += 1

        return self