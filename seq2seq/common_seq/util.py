"""
Substantially adapted from squad code

"""
from __future__ import annotations

import logging
import os
import random
import socket
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import tqdm
import wandb

log = logging.getLogger(__name__)

@dataclass
class ProcessedBatch:
    src_ids: torch.Tensor
    src_mask: torch.Tensor
    tgt_ids: torch.Tensor
    orig_text_input: List[str]
    orig_text_output: List[str]

    batch_size: int
    idxs: Optional[torch.Tensor] = None


@dataclass
class PerBatchValStep:
    """
	Produced for each batch during a val call; Use to pass outputs around to metrics
	"""
    loss_val: Optional[float] = None
    outputs_greedy: Optional[List[str]] = None
    outputs_greedy_ids: Optional[torch.Tensor] = None
    outputs_sampled: Optional[List[List[str]]] = None


###
# Useful misc utilities
###
def symlink_dir(wb_run_obj, readable_name):
    path_spec = '{wandb_dir}/' + readable_name + '-{timespec}-{run_id}'
    sym_path = wb_run_obj._settings._path_convert(path_spec)

    # link {wandb_dir}/{run_mode}-{timespec}-{run_id}/files ->
    #   {wandb_dir}/run_name-{timespec}-{run_id}
    os.symlink(wb_run_obj.dir, sym_path)
    log.info(f'sym: {sym_path} -> {wb_run_obj.dir}')

def get_available_devices(assert_cuda=False) -> Tuple:
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        if len(gpu_ids) > 1:
            log.warning("more than 1 gpu found")

        assert gpu_ids[0] == 0
        device = torch.device(f'cuda:{gpu_ids[0]}')     # cuda:0

        # torch.cuda.set_device(device)     # removed 4/15/2021
    else:
        if assert_cuda:
            raise ValueError('no cuda found')
        device = torch.device('cpu')

    log.info(f"Device: {device}\t GPU IDs: {gpu_ids}\t machine: {socket.gethostname()}\n")

    return device, gpu_ids


def config_logger(logger, log_dir, log_level="debug", filename="log.txt"):
    """

    :param logger:
    :return:
    """
    if len(logger.handlers):
        log.warning(f'Logger had handlers already set WTF\n'
                    f'..... CLEARING')
        logger.handlers.clear()

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    if log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif log_level == "info":
        logger.setLevel(logging.INFO)
    else:
        raise ValueError(f"Invalid log level {log_level}")

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    # console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
    #                                       datefmt='%m.%d.%y %H:%M:%S')
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)s()]\t %(message)s',
        datefmt='%m.%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger(log_dir, name, log_level="debug", filename="log.txt"):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    logger = logging.getLogger(name)
    config_logger(logger, log_dir, log_level, filename)
    return logger


def set_seed(seed=42):
    log.info("Setting seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # todo: cuda deterministic?


class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update_sum_direct(self, num_succ, num_samples):
        self.count += num_samples
        self.sum += num_succ
        self.avg = self.sum/self.count

    def update(self, val: float, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

###
# Tensorboard and other save functions
###

def log_scalar(name: str, value, step=None,
               log_wandb=True):
    """
    1/13:2021: some calls use actual step; some calls use epoch. verify that wandb can handle this
    # this might fail with tbx since the step will vary considerably across runs
    """
    if log_wandb:
        wandb.log({name: value}, step=step)

def log_wandb_new(log_dict: Dict,
                  use_step_for_logging: bool,
                  step: int,
                  epoch: int):
    """
    1/13:2021: some calls use actual step; some calls use epoch. verify that wandb can handle this
    # this might fail with tbx since the step will vary considerably across runs
    """
    # todo: switch over to this once we are done with this project
    if use_step_for_logging:
        step_for_logging = step
        log_dict['epoch'] = epoch
    else:
        step_for_logging = epoch
        log_dict['all_step'] = step
    wandb.log(log_dict, step=step_for_logging)

    # step_for_logging = epoch
    # log_dict['true_step'] = step
    # wandb.log(log_dict, step=step_for_logging)

def log_wandb(log_dict: Dict,
              step: Optional[int]=None):
    """
    1/13:2021: some calls use actual step; some calls use epoch. verify that wandb can handle this
    # this might fail with tbx since the step will vary considerably across runs
    """
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)

def save_preds(preds: List[Tuple[str,str,str]], save_dir, file_name, epoch):
    """Save predictions `preds` to a CSV file named `file_name` in `save_dir`.

    Args:
        preds (list): List of predictions each of the form (source, target, actual),
        save_dir (str): Directory in which to save the predictions file.
        file_name (str): File name for the CSV file.

    Returns:
        save_path (str): Path where CSV file was saved.
    """
    # Validate format
    # if (not isinstance(preds, list)
    #         or any(not isinstance(p, tuple) or len(p) != 3 for p in preds)):
    #     raise ValueError('preds must be a list of tuples (id, start, end)')

    # Make sure predictions are sorted by ID
    # preds = sorted(preds, key=lambda p: p[0])

    # Save to a CSV file
    save_path = os.path.join(save_dir, f'{file_name}_{epoch}.csv')
    np.savetxt(save_path, np.array(preds), delimiter='\t', fmt='%s')

    return save_path
