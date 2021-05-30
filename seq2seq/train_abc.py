"""
Overview

Provides:
- ABC Trainer
- T5Trainer that overrides ABC Trainer (other T5 trainers should subclass this one).
    T5trainer does not implement val_step() so it cannot be instantiated

roughly, _ methods need to be overwritten; other methods can be overwritten

Control Flow for ABC Trainer:
init:
    - setup_model_and_device
        #subclass: _setup_model_and_tokenizer
        gets available devices
        loads checkpoint
        adds special tokens
    - setup_dataloaders
        #subclass _get_dataloaders
        #optional #subclass _verify_data_loader_sizes (if verify_fits_in_mem)
        setes num_train, num_val, optim_sets, total_train
    - #subclass _setup_optim_sched
    - verify_and_log_trainer_info

run:
    for warmup_epochs, epochs:
        #subclass train_step # both for warmup and non-warmup
        val_stepsave_callback
        save_callback

    train (overridden


All to overwrite:
- _setup_model_and_tokenizer
- _get_dataloaders
- _setup_optim_sched


RunHelper:
    Can be overridden in other files that overwrite T5Trainer

TrainInfo: Tracks the trainer's current state


Note:
    - Because this is setup for multitask train, logging epochs will always start at 10 for primary training
    for consistency

"""
import json
import logging
import os
import socket
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict, NoReturn, Set, List, Union, Callable

import numpy as np
import torch
import wandb
from overrides import EnforceOverrides, final, overrides
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import (
    PreTrainedModel, PreTrainedTokenizer,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer, T5TokenizerFast,
)

from common_seq import (
    util as util,
    util_checkpoint, util_dataloader,
    util_multiloader
)
from common_seq.util import (
    ProcessedBatch, PerBatchValStep,
)
from common_seq.util_checkpoint import CheckpointDict
from common_seq.util_dataloader_batch import ClueDataLoaderBatched, get_dataloaders_batched
from common_seq.util_metrics import MetricsPredsWrapper, MetricFcn
from common_seq.util_multiloader import MultiTaskDataLoader

log = logging.getLogger(__name__)       # todo: change to logging.getLogger(__name__)

# semi-hacky we leave space for 10 warmup steps (so that graphs line up)
k_max_warmup_epochs = 10                # primary train will always start at epoch == k_max_warmup_epochs + 1

# Other, okay
k_set_opt_to_none = True                # whether to reset grad params to None vs 0 when resetting opt; this is
                                        # huggingface default
k_use_step_for_logging = True           # originally was using epoch

# to run dataparallel, set this flag and pass --multi_gpu=#
# This setting is not recommended because it will train more slowly.
# flag is added so that user must specify twice to use multiple gpus through dataparallel
k_data_parallel = False

# Hacky:
# in the case that multitask needs to be resumed, one can set this flag
# to resume from multitask, you need to already be passing --resume_train and --multitask
# to *actually* resume, you must then also pass --hacky since the resume functionality for multitask
# was hacked together.
# you must also set the following constant to the number of warmup epochs already done
# todo(cleanup so this isn't necessary)
k_hard_reset_warmup_iters_done = None

class RunHelper:
    """
    Object that the Trainer has to get access to various non-trainer objects
    """

    def __init__(self,
                 dir,
                 metrics: List[Tuple[str, bool]],
                 config=None):
        """

        :param dir:
        :param metrics:
        :param config: wandb config object
        :param labels_files_map:
        """
        self.global_record_dir = dir

        # self.tbx = SummaryWriter(self.global_record_dir, flush_secs=5)
        if config.multitask:
            assert 'multisave' in [x[0] for x in metrics], 'multitask but not saving multisave metric'
        self.metrics_to_track = metrics
        self.ckpt_saver = util_checkpoint.CheckpointSaver(save_dir=self.global_record_dir,
                                                          metrics_to_track=metrics)

        # special for computing extra labels
        self.clue_to_idx_map: Optional[Dict[str, int]] = None
        self.eval_labels: Optional[List[str, Set[int]]] = None


class TrainInfo:
    """
    Tracks the trainer's state (epoch, steps, num warmupsteps = warmup epochs)
    """

    def __init__(self,
                 multitask_mgr: Optional[util_multiloader.MultitaskManager] = None):
        # state tracking
        self._epoch: int = 0                 # current epoch; epochs < k_max_warmup are warmup
                                             # we start at == 0 since train() will increment to 1 at start
        self._all_step: int = 0              # num training examples seen
        self._step_after_warmup: int = 0     # num training ex seen after warmup
        self.metric_best = None              # for early stopping        # todo: make property; type

        self._num_warmup_epochs: Optional[int] = None   # none if there is no multitask; could be 0 if there are no warmup epochs
        self.__init_epoch(multitask_mgr)

    def __init_epoch(self, multitask_mgr):
        """
        will initialize either to 0 (warmup) or k_max_warmup

        increment_epoch() will be called before actually training which will give us + 1 so that we start at either
        1 (warmup) or k_max_warmup + 1
        """
        if multitask_mgr is not None:
            self._num_warmup_epochs = multitask_mgr.multitask_warmup
            if self._num_warmup_epochs > k_max_warmup_epochs:
                raise NotImplemented
            # self._epoch = -self._num_warmup_epochs
        else:   # now armup
            # self._num_warmup_epochs = 0         # default to no warmup
            self._epoch = k_max_warmup_epochs   # starts at k_max_warmup for primary train (+1 will be added in increment)

    def is_warmup(self) -> bool:
        # todo: just add a "has warmup" flag
        if self._num_warmup_epochs is None:     # no warmup
            # assert self._epoch > k_max_warmup_epochs
            return False

        # otherwise, could be in warmup
        return self._epoch < k_max_warmup_epochs

    def warmup_remaining(self) -> int:
        # warmup remaining cannot be negative
        return max(self._num_warmup_epochs - self._epoch, 0)

    @property
    def epoch(self):
        return self._epoch

    @property
    def step(self):
        return self._all_step

    def increment_steps(self, batch_num: int):
        """
        Increment current step trackers by batch_num examples
        :param batch_num:
        """
        self._all_step += batch_num
        if not self.is_warmup:
            self._step_after_warmup += batch_num

    def increment_epoch(self) -> bool:
        """

        :return: whether it was the final warmup epoch (i.e. primrary training beginning)
        """
        orig = self._epoch

        self._epoch += 1
        was_last_warmup = False

        # if we are finishing warmup (i.e.
        if self._num_warmup_epochs is not None and self._epoch == self._num_warmup_epochs + 1:
            self._epoch = k_max_warmup_epochs + 1       # primary train starts here
            was_last_warmup = True

        assert self._epoch > orig

        return was_last_warmup

    def resume(self, epoch: int,
               step: int):
        self._epoch = epoch
        self._all_step = step
        log.warning('Train info resumed but did not set all_step_after_warmup')

    # # always leave space for warmup runs (so that primary logging aligns)
    # def epoch_for_logging(self) -> int:
    #     if self._epoch <= 0:
    #         return self._num_warmup_epochs + self._epoch
    #     else:
    #         return k_max_warmup_epochs + self._epoch


class Trainer(EnforceOverrides, metaclass=ABCMeta):
    def __init__(self, config: wandb.Config,
                 rh: RunHelper,
                 aux_config: Optional[Dict] = None):
        self.config = config          # wandb Config
        self.aux_config = aux_config  # aux dictionary
        self.rh = rh

        # legacy: still allow val_fcn_list to be created in val_step()
        self.val_fcn_list: Optional[List[MetricFcn]] = None      # should be set in the implementing subclass (in init)
        self.init_val_fcns()

        # pytorch trainer objects
        self.device: Optional[torch.device] = None
        self.gpu_ids: Optional[List[int]] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        # self.model_parallel = None
        self.model_is_parallelized: bool = False
        self.setup_model_and_device()       # populate 3 above; potentially add special tokens

        self.train_loader: Optional[Union[ClueDataLoaderBatched, MultiTaskDataLoader]] = None
        self.dev_loader: Optional[ClueDataLoaderBatched] = None
        self.multitask_manager: Optional[util_multiloader.MultitaskManager] = None  # set if we are doing multitask
        self.setup_dataloaders()

        self.optimizer: Optional[Adafactor] = None
        self.scheduler: Optional[lr_scheduler] = None
        self._setup_optim_sched()

        # trainer state (must go here since ref'd by verify_and_log)
        self.state = TrainInfo(multitask_mgr=self.multitask_manager)

        self.verify_and_log_trainer_info()

        # if we're resuming
        if self.config.ckpt_path:
            if not self.config.no_train:
                assert self.config.resume_train is not None
            # if resume train, train state, optim, and scheduler will be changed
            self.load_from_ckpt(resume_train=self.config.resume_train)

        # todo misc attributes
        # metrics to track is stored in rh.checkpointsaver

    @abstractmethod
    def init_val_fcns(self):
        pass

    @final
    def load_from_ckpt(self, resume_train=False):
        # todo: print where the config dictionaries differ
        # loads model state
        log.info(f'Loading checkpoint: {self.config.ckpt_path}')
        ckpt_dict: CheckpointDict = util_checkpoint.load_ckpt(self.config.ckpt_path,
                                                              self.model,
                                                              map_location=self.device)
        if not self.config.no_train and resume_train:
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            if self.scheduler is not None:
                raise NotImplemented('Resume not implemented for adam')

            # todo: should also set other properties of state
            self.state.resume(epoch=ckpt_dict['epoch'],
                              step=ckpt_dict['step'])

            # if we're resuming with multitask
            if self.config.multitask:
                # todo(hacK): this needs to be cleaned up
                # so that we can actually resume multitask
                # multitask state needs to be moved into trainer state
                # shoudl also check for equivalence of the other params
                # todo: remove this after verifying that everything is okay

                assert self.config.hacky
                assert isinstance(k_hard_reset_warmup_iters_done, int) and k_hard_reset_warmup_iters_done > 0
                total_warmup_todo = self.multitask_manager.multitask_warmup
                total_warmup_done = k_hard_reset_warmup_iters_done
                warmup_remaining = total_warmup_todo - total_warmup_done
                assert warmup_remaining == self.state.warmup_remaining()

                # before fixing epoch
                # self.state.epoch -= warmup_remaining      # will be incremented before running first epoch in run()

                # trainloader is a multiloader
                # reset its state correctly
                self.train_loader.num_iters = k_hard_reset_warmup_iters_done

                # # josh hack 04/14/2021
                # self.state.resume(epoch=0,                  # go back to epoch 0
                #                   step=ckpt_dict['step'])
                # self._setup_optim_sched()       # reset

                log.info(f'Set up at epoch {self.state.epoch}, with {self.train_loader.warmup_iters}'
                         f' total warmup, and {self.train_loader.num_iters} already done, ie'
                         f'{self.state.warmup_remaining()} warmup todo')

    @final
    def run(self):
        # if not training
        if self.config.no_train:
            log.info(f'arg no_train given. Just doing single validation. Setting to epoch == 1')
            self.state.increment_epoch()        # set to epoch == 1
            assert self.state.epoch == k_max_warmup_epochs + 1
            self.val_only()
            return

        log.warning(
            f'For actual train, epochs start at {k_max_warmup_epochs + 1}')
        # main training; includes warmup
        while self.state.epoch < self.config.num_epochs + k_max_warmup_epochs:
            was_last_warmup = self.state.increment_epoch()
            if was_last_warmup and self.multitask_manager.multitask_reset:
                log.info('Final warmup epoch done. Resetting optimizer')
                self._setup_optim_sched()

            self.train_step()

            # Validate; this will do both multitask and normal validation
            all_metrics = self.val_step()
            metrics_dict, preds = self.metrics_list_to_dict(all_metrics)
            # will have multisave appended if it is multitasking
            self.save_callback(metrics_dict, preds)
            if self.early_stopping_callback(metrics_dict):
                break

        # e.g., final eval
        self.post_run()

    def val_only(self):
        metrics: List[MetricsPredsWrapper] = self.val_step()
        metrics_dict, preds = self.metrics_list_to_dict(metrics)
        self.save_callback(metrics_dict, preds)

    # does not have to be implemented by subclasses
    def post_run(self):
        pass

    @final
    def setup_model_and_device(self):
        # device will be cuda:0
        self.device, self.gpu_ids = util.get_available_devices(assert_cuda=True)
        assert str(self.device) == "cuda:0", f'{self.device} != cuda:0'

        if len(self.gpu_ids) > 1 or self.config.multi_gpu is not None or k_data_parallel:
            logging.info(f'{len(self.gpu_ids)}, {self.config.multi_gpu}, {k_data_parallel}')
            assert k_data_parallel
            assert len(self.gpu_ids) == self.config.multi_gpu

        self._setup_model_and_tokenizer()  # implemented by subclasses

        if self.config.add_special_tokens:
            util.add_special_tokens(self.model, self.tokenizer)  # adds <SEP>

        self.model_to_device()

    # todo: we might be able to omit this and just have it in the self.train() function
    @final
    def model_to_device(self):
        if k_data_parallel:
            log.info('Using dataparallel')
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
            self.model_is_parallelized = True

        self.model.to(self.device)


    @abstractmethod
    def _setup_model_and_tokenizer(self):
        """
        Should load and make any tweaks (e.g. vocab changes) the following
        - self.model
        - self.tokenizer
    
        Called by setup_model_and device()
        """
        pass


    @abstractmethod
    def setup_dataloaders(self):
        pass

    @abstractmethod
    def _get_dataloaders(self):
        """
        Should set
        - train_loader
        - dev_loader
        """
        pass

    @abstractmethod
    def _setup_optim_sched(self):
        """
        Should set
        - optimizer
        - scheduler (optional)
        """
        pass

    def verify_and_log_trainer_info(self):
        # verify that the metric we want to log is valid
        log.info('Verifying that all metrics are OK. The outputs here are NOT from the model that was passed if'
                 'one was passed')
        metrics_dict, _ = self.metrics_list_to_dict(self.val_step(trial_run=True))
        for m in self.rh.metrics_to_track:
            if m[0] in ['epoch']:      # these won't be in the normal metrics returned
                continue
            assert m[0] in metrics_dict, f'{m} not in {metrics_dict}'

        log.info(f'Tracking metrics {self.rh.metrics_to_track} all verified')
        
        # verify everything else
        assert all(map(lambda x: x is not None,
                       [self.config, self.model, self.tokenizer, self.device, self.optimizer,
                        self.train_loader, self.dev_loader]))

        # validation freq
        if self.config.val_freq is not None:
            assert self.config.val_freq * 1000 < self.config.num_train
            # we log as {epoch}.{intermed/100} so max is 100 99
            assert self.config.num_train / (self.config.val_freq * 1000) < 100

        log_string = '\n' \
                      f'total_train_steps (num_train_ex * epochs): {self.config.total_train}\n' \
                      f'machine: {socket.gethostname()}\n' \
                      f'num_train: {self.config.num_train}\n' \
                      f'num_val: {self.config.num_val}'
                      # log_string += f'total_optim_steps: {self.config.total_optim_steps}\n' \

        # can't use json for first config dict because not of type dic
        for k, v in sorted(self.config.items(), key=lambda x: x[0]):
            log_string += f'{k}: {v}\n'
        if self.aux_config:
            log_string += "multitask:\n"
            log_string += json.dumps(self.aux_config, sort_keys=True, indent=2,
                                     cls=util_dataloader.EnhancedJSONEncoder)
        else:
            log_string += "No aux config (e.g. multitask) given"
        log_string += "\n"
        log.info(log_string)

    @abstractmethod
    def _batch_to_objects(self, batch) -> ProcessedBatch:
        pass

    def val_end_epoch(self,
                      metrics_all_accum: Union[List[MetricsPredsWrapper], MetricsPredsWrapper],
                      num_val=None):
        if isinstance(metrics_all_accum, MetricsPredsWrapper):
            metrics_all_accum = [metrics_all_accum]

        for m_dict in metrics_all_accum:
            # get_all_metrics will already have a <val_label>:<set_label>:
            for k, v, orig_v in m_dict.get_all_metrics(num_val):
                # Log val and avg val
                log.info(f'{k}: {orig_v:05.2f}\t avg: {v:05.4f}')
                # util.log_scalar(f'{k}', v/self.config.num_val, self.state.epoch, tbx=self.rh.tbx)
                util.log_wandb_new({f'{k}': v},
                                   step=self.state.step,
                                   epoch=self.state.epoch,
                                   use_step_for_logging=k_use_step_for_logging)

    @abstractmethod
    def model_forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor, tgt_ids: torch.Tensor) -> \
        Tuple[torch.Tensor, Dict]:
        pass

    @abstractmethod
    def train_step(self) -> NoReturn:
        # will generally need to call model_forward method
        pass

    @abstractmethod
    def _generate_outputs_greedy(self, src_ids, src_mask, skip_special_tokens=True) -> Tuple:
        pass

    @abstractmethod
    def _generate_outputs_sampled(self, src_ids, src_mask, batch_size) -> List:
        pass

    def get_valstepdict_for_batch(self,
                                  pbatch: ProcessedBatch,
                                  do_sample: bool,
                                  do_generate: bool = True) -> PerBatchValStep:
        # evaluation for loss fcn
        perbatch_valstep = PerBatchValStep()
        loss, _ = self.model_forward(pbatch.src_ids,
                                     pbatch.src_mask,
                                     pbatch.tgt_ids)  # loss, logits, but don't need logits
        if k_data_parallel:
            loss = loss.mean()
        perbatch_valstep.loss_val = loss.detach().item()

        if do_generate:
            outputs_decoded_greedy, generated_ids_greedy = \
                self._generate_outputs_greedy(pbatch.src_ids, pbatch.src_mask)
            perbatch_valstep.outputs_greedy = outputs_decoded_greedy
            perbatch_valstep.outputs_greedy_ids = generated_ids_greedy

        if do_sample:
            outputs_decoded_sampled = \
                self._generate_outputs_sampled(pbatch.src_ids, pbatch.src_mask, pbatch.batch_size)
            perbatch_valstep.outputs_sampled = outputs_decoded_sampled

        return perbatch_valstep

    def val_step(self, trial_run: bool = False) -> List[MetricsPredsWrapper]:
        """
        :param trial_run: whether this is an initial check run - only one batch will be computed
        :return:
        """
        log.info(f'Evaluating at all_step {self.state.step} (epoch={self.state.epoch})...')
        self.eval()
        # self.model.eval()  # put model in eval mode

        # accumulate all metrics over all of the val_dls
        all_metrics_wrappers: List[MetricsPredsWrapper] = []

        # if not self.state.epoch > 0 or trial_run:     # not warmup
        if not self.state.is_warmup() or trial_run:
            log.info(f'Primary eval; epoch: {self.state.epoch}')
            metrics_accum = self.validate_val_loader(self.dev_loader,
                                                     self.val_fcn_list,
                                                     trial_run,
                                                     label='dev',
                                                     do_print=True)
            all_metrics_wrappers.append(metrics_accum)

        # always do multitask
        if self.config.multitask:
            log.info(f'Multitask eval; epoch: {self.state.epoch}')
            for val in self.multitask_manager.val_dls:
                log.info(f'Validating DL {val.name}')
                metrics_accum = self.validate_val_loader(val.dataloader,
                                                         val.val_fcn_list,
                                                         trial_run=trial_run,
                                                         label=f'multi/{val.name}',
                                                         do_print=False)
                # we don't save predictions from the multiloaders
                metrics_accum.preds = None
                all_metrics_wrappers.append(metrics_accum)

        assert len(all_metrics_wrappers) > 0, 'Val step called with invalid params'

        if not trial_run:
            self.val_end_epoch(all_metrics_wrappers, num_val=None)  # use the avg divisor as set in the constructor
        return all_metrics_wrappers

    def validate_val_loader(self, val_loader: ClueDataLoaderBatched,
                            val_fcn: List[Callable],
                            trial_run: bool,
                            label: str,
                            do_print: bool):
        metrics_all_accum: MetricsPredsWrapper = MetricsPredsWrapper(label=label,
                                                                     avg_divisor=self.dev_loader.num_examples())
        loss_meter = util.AverageMeter()  # NLL (default metric for model) (reset each time)

        # todo: should total be num_examples or num_val
        with torch.no_grad(), \
            tqdm(total=val_loader.num_examples()) as progress_bar:
            for batch_num, batch in enumerate(val_loader):
                # run a single batch and then return
                if trial_run and batch_num > 0:
                    break

                pbatch = self._batch_to_objects(batch)
                valstepbatch = self.get_valstepdict_for_batch(pbatch, do_sample=self.config.do_sample)

                # update metrics and predictions tracking
                metrics_all_accum.update_for_batch(val_fcn,
                                                   valstepbatch,
                                                   pbatch,
                                                   metric_label='')

                loss_meter.update(valstepbatch.loss_val,
                                  pbatch.batch_size)
                progress_bar.update(pbatch.batch_size)
                progress_bar.set_postfix(NLL=loss_meter.avg)

                # On first batch print one batch of generations for qualitative assessment
                if do_print and batch_num == 0:
                    for idx, orig_input, orig_target, output_greedy, *other in metrics_all_accum.preds[:1]:
                        log.info(f'\n idx: {idx}'
                                 f'\nSource: {orig_input}\n '
                                 f'\tTarget: {orig_target}\n'
                                 f'\t Actual: {output_greedy}\n')

        # append the NLL to the metrics
        metrics_all_accum.add_val('NLL', loss_meter.avg, avg=False, label='')

        return metrics_all_accum

    ##
    # Other helper functions
    ###
    def metrics_list_to_dict(self, metrics_wrappers: List[MetricsPredsWrapper]) -> Tuple[Dict, List]:
        all_metrics_dict = dict()

        # we should have only a single set of preds; this is hacky. we set all multiloader
        # preds to None during val_step() which is the only time this MetricsPredswrappers are produced
        preds = None
        for m in metrics_wrappers:
            all_metrics_dict.update(m.get_all_metrics_dict())
            if m.preds is not None:
                assert preds is None
                preds = m.preds

        # could also do this; but then change the code in util_checkpoitn
        # all_metrics_dict.update(dict(epoch=self.state.epoch))
        # todo: hacky
        # if self.config.multitask and self.state.epoch <= 0:
        if self.config.multitask and self.state.is_warmup():
            all_metrics_dict.update(dict(multisave=self.state.epoch))

        return all_metrics_dict, preds

    def save_callback(self, metrics_dict, preds, intermed_epoch=None):
        # save_metrics = metrics.get_all_metrics_dict()
        # save_preds = metrics.preds
        if intermed_epoch is not None:
            save_epoch = self.state.epoch + intermed_epoch / 100
        else:
            save_epoch = self.state.epoch
        self.rh.ckpt_saver.save_if_best(save_epoch,
                                        self,
                                        # metric_dict=save_metrics,
                                        metric_dict=metrics_dict,
                                        # preds=save_preds,
                                        preds=preds,
                                        save_model=self.config.do_save)

    # todo(wrong): support max/min metrics
    def early_stopping_callback(self, metrics_dict: Dict):
        if not self.config.early_stopping:
            return False
        if self.config.early_stopping not in metrics_dict:
            log.warning(f'Early stopping but metric {self.config.early_stopping} not found')
            return False
        curr_metric = metrics_dict[self.config.early_stopping]
        if self.state.metric_best is not None:
            if self.state.metric_best < curr_metric:
                log.info(f"Early stopping: prev {self.state.metric_best}\t current: {curr_metric}")
                return True
            else:
                log.info(f"Not stopping: prev {self.state.metric_best}\t current: {curr_metric}")
        # otherwise store new best
        self.state.metric_best = curr_metric
        return False

    def make_ckpt_dict(self) -> CheckpointDict:
        self.model.cpu()      # todo(parallel): verify this isn't necessary for save
        model_for_ckpt = self._model_for_ckpt()

        sched = None
        if self.scheduler is not None:
            sched = self.scheduler.state_dict()

        ckpt_dict: CheckpointDict = {
            # 'model_state': self.model.state_dict(),
            'model_state': model_for_ckpt.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': sched,
            'config': dict(self.config.items()),  # todo: fix this (so that it can be reloaded)
            'step': self.state.step,
            'epoch': self.state.epoch
        }
        # was needed when we did self.model.cpu()
        self.model.to(self.device)
        return ckpt_dict

    def _model_for_ckpt(self):
        #todo(parallel): verify don't need cpu
        if k_data_parallel:
            return self.model.module
        else:
            return self.model
        # model_for_save = self.model.cpu()
        # return model_for_save

    def eval(self):
        if k_data_parallel and self.model_is_parallelized:
            self.model = self.model.module
            self.model_is_parallelized = False
            # self.model.to(self.device)      # todo(parallel): do we need this?
        self.model.eval()  # put model in eval mode

    def train(self):
        if k_data_parallel and not self.model_is_parallelized:
            self.model = DataParallel(self.model, self.gpu_ids)
            self.model_is_parallelized = True
            #self.model.to(self.device)      # todo(parallel): do we need this?
        self.model.train()



class T5Trainer(Trainer, metaclass=ABCMeta):
    """
    This class cannot be instantiated, see methods that still need to be implemented below

    Overrides:
            _setup_model_and_tokenizer
            _get_dataloaders
            _setup_optim_sched
            _batch_to_objects
            _train_step
            model_forward

    Still to be implemented in subclasses:
            val_update_metrics_and_preds
            val_step
    """

    def __init__(self, config: wandb.Config, rh: RunHelper, **kwargs):
        super().__init__(config, rh, **kwargs)

    @overrides
    def _setup_model_and_tokenizer(self):
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        if self.config.fast_tokenizer:
            self.tokenizer = T5TokenizerFast.from_pretrained(self.config.model_name)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)

    ###
    # Dataloader setup functions
    ###
    def setup_dataloaders_no_multi(self, batched_collate_fcns: Optional[List[Callable]]=None):
        self._get_dataloaders(batched_collate_fns=batched_collate_fcns)

        # reset in case we used the -1 flag for all data
        # num_train, num_val = len(self.train_loader.dataset), len(self.dev_loader.dataset)
        num_train, num_val = self.train_loader.num_examples(), self.dev_loader.num_examples()
        self.config.update(
            {'num_train': num_train,
             'num_val': num_val,

             # num times that optim.step() will be called
             # todo: is total_optim_steps this correct for multiloader? (we don't need it)
             # 'total_optim_steps': ((num_train // self.config.batch_size) * self.config.num_epochs),
             'total_train': num_val * self.config.num_epochs},
            allow_val_change=True)

    def setup_dataloaders_multi(self, batched_collate_fcns: Optional[List[Callable]]=None,
                                multitask_collate_fn: Optional[Callable] = None):
        log.info('Setting up for multitask')

        # primary train and dev loaders
        self._get_dataloaders(batched_collate_fns=batched_collate_fcns)     # this populates self.train_loader and self.dev_loader
        train_loader1: ClueDataLoaderBatched = self.train_loader
        dev_loader: ClueDataLoaderBatched = self.dev_loader

        # load the multitask trainers
        multi_cfg: util_multiloader.MultitaskConfig = self.aux_config['multitask_config']
        self.multitask_manager = \
            util_multiloader.MultitaskManager(multi_cfg,
                                              tokenizer=self.tokenizer,
                                              batch_size=self.config.batch_size,
                                              num_examples=self.config.multitask_num,
                                              use_json=self.config.use_json,
                                              collate_fn=multitask_collate_fn)
        train_loader_multi = self.multitask_manager.get_train_multiloader(train_loader1,
                                                                          self.multitask_manager.multitask_warmup)

        # reset in case we used the -1 flag for all data
        # todo: these numbers might not make sense in the multi case
        # note that num_train is the number of true train examples, not multi
        # num_train, num_val = len(train_loader) * self.config.batch_size, \
        #                      len(dev_loader) * self.config.batch_size
        num_train, num_val = train_loader1.num_examples(), dev_loader.num_examples()
        self.config.update(
            {'num_train': num_train,
             'num_val': num_val,

             # num times that optim.step() will be called
             # todo: this is wrong with multiloader
             # 'total_optim_steps': ((num_train // self.config.batch_size) * self.config.num_epochs),
             'total_train': (num_val * self.config.num_epochs)},
            allow_val_change=True)

        self.train_loader: MultiTaskDataLoader = train_loader_multi

    @overrides
    def setup_dataloaders(self):
        if not self.config.multitask:
            self.setup_dataloaders_no_multi()
        else:
            self.setup_dataloaders_multi()

    @overrides
    @final
    def _get_dataloaders(self, batched_collate_fns=None):
        if self.config.batched_dl:
            # todo: using only one of the num_train/num_val params
            ds_config_train = util_dataloader.DatasetConfig(tokenizer=self.tokenizer,
                                                      max_examples=self.config.num_train)
            ds_config_val = util_dataloader.DatasetConfig(tokenizer=self.tokenizer,
                                                      max_examples=self.config.num_val)

            dl_config_train = util_dataloader.DataLoaderConfig(shuffle=True,
                                                               batch_size=self.config.batch_size,
                                                               num_workers=self.config.num_workers,
                                                               use_json=self.config.use_json)
            dl_config_val = util_dataloader.DataLoaderConfig(shuffle=False,
                                                             batch_size=self.config.batch_size,
                                                             num_workers=self.config.num_workers,
                                                             use_json=self.config.use_json)
            train_loader, dev_loader = \
                get_dataloaders_batched(self.tokenizer,
                                        dataset_config_train=ds_config_train,
                                        dataset_config_val=ds_config_val,
                                        dl_config_train=dl_config_train,
                                        dl_config_val=dl_config_val,
                                        data_dir=self.config.data_dir,
                                        label_fn=None,
                                        clue_to_idx_map=self.rh.clue_to_idx_map,
                                        collate_fns=batched_collate_fns,
                                        use_test_set=self.config.test)

        else:
            raise NotImplemented('non-batched loading is deprecated')

        self.train_loader, self.dev_loader = train_loader, dev_loader

    @overrides
    def _setup_optim_sched(self):
        """Assumes we are using adafactor. See huggingface transformers readme"""
        if not self.config.ada:
            raise NotImplemented
        # beta1 = 0 (beta1 defaults to 0 -> don't use first moment)
        # clipping threshold = default
        # per the instructions in transformers/optimization.py (for adafactor); other params are the defaults

        # this is a constant LR optimizer.
        # this is worse than the second one below (except for T5 large)
        if self.config.ada_constant:
            log.info('Using constant lr for adafactor')
            optimizer = Adafactor(self.model.parameters(),
                                  lr=1e-3,
                                  relative_step=False,
                                  warmup_init=False,
                                  scale_parameter=False)
        else:
            optimizer = Adafactor(self.model.parameters(),
                                  relative_step=True,
                                  warmup_init=True)

        # worst (i.e. don't use this)
        # optimizer = Adafactor(self.model.parameters(),
        #                       relative_step=True,
        #                       warmup_init=True,
        #                       scale_parameter=False)

        # adafactor does not use scheduler (but ADAM does)
        scheduler = None

        self.optimizer, self.scheduler = optimizer, scheduler

    @overrides
    def _batch_to_objects(self, batch) -> ProcessedBatch:
        ret = ProcessedBatch(
            src_ids=batch["source_ids"].to(self.device, dtype=torch.long),
            src_mask=batch["source_mask"].to(self.device, dtype=torch.long),
            tgt_ids=batch["target_ids"].to(self.device, dtype=torch.long),
            orig_text_input=batch["source_text"],
            orig_text_output=batch["target_text"],
            batch_size=len(batch["source_ids"]),
            idxs=batch.get('idxs', None)
        )
        return ret

    @overrides
    def model_forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor, tgt_ids: torch.Tensor,
                      **model_kwargs) -> Tuple[torch.Tensor, Dict]:
        # padded ids (pad=0) are set to -100, which means ignore for loss calculation
        label_ids = tgt_ids.clone().detach()
        label_ids[label_ids[:, :] == 0] = -100
        label_ids.to(self.device)  # todo: necessary?
        # when we call model() with labels, they will be
        # - automatically right shifted
        # - prepended by BOS (pad token)
        # - any token that was -100 will be masked_fill_ to <pad>
        out_dict = self.model(src_ids,
                              attention_mask=src_mask,
                              labels=label_ids,
                              return_dict=True,
                              **model_kwargs)
        return out_dict['loss'], out_dict

    def train_backward(self, batch_idx: int, loss: torch.Tensor):
        # see https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
        # for grad_accum details
        loss = loss / self.config.grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % self.config.grad_accum_steps == 0:
            # if self.config.max_grad_norm is not None:
            #     # todo
            #     # if self.config.ada:
            #     #     raise NotImplemented('max grad is ignored when using ada')
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=k_set_opt_to_none)

            if self.scheduler is not None:
                self.scheduler.step()  # don't need to pass step to scheduler


    @overrides
    def train_step(self):
        if self.state.is_warmup():
            log.info(f'Training warmup={self.state.epoch}...')
        else:
            log.info(f'Training epoch={self.state.epoch}...')

        self.train()
        # self.model.train()

        # todo: assumes the batch size is the same across loaders
        total_steps = 0
        val_step = 0
        with torch.enable_grad(), \
            tqdm(total=self.train_loader.num_examples()) as progress_bar:
            self.optimizer.zero_grad(set_to_none=k_set_opt_to_none)  # probably not necessary, added with grad accum
            for batch_num, batch in enumerate(self.train_loader):
                # intermediate validations; do here so that we don't double validate the end
                # intermed val only when not warming up, and when we have passed the step for validation

                # todo: save vs eval
                # todo: don't duplicate this code
                if self.config.val_freq is not None and \
                    not total_steps // (self.config.val_freq * 1000) - val_step == 0:
                    # not self.state.is_warmup() and \
                        assert total_steps // (self.config.val_freq * 1000) - val_step == 1
                        val_step += 1
                        log.info(f'Doing intermediate eval at {self.state.epoch}.{val_step}')

                        all_metrics = self.val_step()
                        metrics_dict, preds = self.metrics_list_to_dict(all_metrics)
                        # will have multisave appended if it is multitasking
                        self.save_callback(metrics_dict, preds, intermed_epoch=val_step)
                        # self.model.train()      # restore train
                        self.train()              # restore train

                # preprocess batch
                pbatch: ProcessedBatch = self._batch_to_objects(batch)
                loss, _ = self.model_forward(pbatch.src_ids, pbatch.src_mask, pbatch.tgt_ids)
                loss = loss.mean()      # needed for parallel; no effect if dim == [1]

                # Backward
                self.train_backward(batch_num, loss)
                loss = loss.detach()

                progress_bar.update(pbatch.batch_size)
                progress_bar.set_postfix(epoch=self.state.epoch, loss=loss.item())

                # todo: do we want step-level logging? probably okay - but this logs including multitask
                self.state.increment_steps(pbatch.batch_size)
                total_steps += pbatch.batch_size

        # todo: log steps for train/loss and train/LR that are not all

        # todo: adafactor is not logging the LR
        util.log_wandb_new(
            {'train/loss_all': loss.item(),
             'train/LR_all': self.optimizer.param_groups[0]['lr'],
             },
            step=self.state.step,
            epoch=self.state.epoch,
            use_step_for_logging=k_use_step_for_logging)

    ###
    # The following methods should maybe be in a further subclass; they are used by
    # most of the classes that subclass T5Trainer
    ###

    # other misc functions to be shared
    @overrides
    def _generate_outputs_greedy(self, src_ids, src_mask, skip_special_tokens=True):
        generated_ids_greedy = self.model.generate(src_ids,
                                                   attention_mask=src_mask
                                                   )  # (batch x seq length)
        outputs_decoded_greedy = self.tokenizer.batch_decode(generated_ids_greedy,
                                                             skip_special_tokens=skip_special_tokens)
        return outputs_decoded_greedy, generated_ids_greedy

    # overrides
    @overrides
    def _generate_outputs_sampled(self, src_ids, src_mask, batch_size):
        generated_ids_sampled = self.model.generate(
            # see wandb sweep for details. length_penalty=0.01 might actually be better
            src_ids,
            attention_mask=src_mask,
            num_beams=self.config.generation_beams,
            num_return_sequences=self.config.generation_beams,
            do_sample=False,
            max_length=10,
            length_penalty=0.05
        )
        outputs_decoded_sampled = self.tokenizer.batch_decode(generated_ids_sampled, skip_special_tokens=True)
        outputs_decoded_sampled = np.array_split(outputs_decoded_sampled, batch_size)
        return outputs_decoded_sampled


###
# Additional helper functions
###

def pre_setup(parsed_args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if parsed_args.dev_run:
        os.environ["WANDB_MODE"] = "dryrun"
        # parsed_args.num_val = parsed_args.num_train = parsed_args.multitask_num = 3000
        parsed_args.num_val = parsed_args.num_train = parsed_args.multitask_num = 10000


def setup_wandb_for_run(local_args, do_symlink=True):
    wb_dir = local_args.wandb_dir
    assert os.path.isdir(wb_dir)
    # wandb todo:
    # - job_type
    # - save_code / reinit / id / resume
    # if local_args.resume_wandb_id is not None:
    #     assert local_args.resume_dir is not None
    #     log.info(f'resuming run from dir {local_args.resume}')
    #     wandb.init(id=local_args.resume_wandb_id,
    #                project=local_args.project,
    #                resume="must")
    # else:
    wandb.init(project=local_args.project,
               name=local_args.name,  # run_name
               notes=local_args.comment,
               force=True,
               dir=wb_dir)
    wandb.config.update(local_args)  # directly add all local arguments
    # wandb.tensorboard.patch(save=True, tensorboardX=True)

    if do_symlink:
        util.symlink_dir(wandb.run, f'{local_args.project}_{local_args.name}')

    # Log info about the run
    run_dir = wandb.run.dir
    log_string = "\n"
    log_string += f'Project: {local_args.project}\n' \
                  f'Run: {local_args.name}\n' \
                  f'Comment: {local_args.comment}\n' \
                  f'Directory: {run_dir}\n'
    return run_dir
