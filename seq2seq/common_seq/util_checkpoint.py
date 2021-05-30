import json
import logging
import os
from json import JSONEncoder
from typing import TypedDict, Dict, Optional, List, Tuple, Union

import numpy
import torch
from transformers import PreTrainedModel

log = logging.getLogger(__name__)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# todo: should this be a dataclass
class CheckpointDict(TypedDict):
    model_state: Dict
    optimizer: Dict
    scheduler: Optional[Dict]
    config: Dict
    step: int
    epoch: int


def load_ckpt(path: str, model: PreTrainedModel, map_location=None,
              log_info=True) -> CheckpointDict:
    ckpt_dict: CheckpointDict = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt_dict['model_state'])
    if log_info:
        try:
            with open(f'{path}.txt') as f:
                log.info(f'Loading model from {path}:\n{f.readlines()}')
        except:
            pass
    return ckpt_dict


class CheckpointSaver:
    """Class to save and load model checkpoints.

Save the best checkpoints as measured by a metric value passed into the
`save` method. Overwrite checkpoints with better checkpoints once
`max_checkpoints` have been saved.

Args:
"""

    def __init__(self, save_dir,
                 metrics_to_track: List[Tuple[str, bool]],
                 save_most_recent=True):
        if save_most_recent:
            metrics_to_track.append(('epoch', True))

        self.save_dir = save_dir
        # Maps metric_name => (maximize_metric, best_val, saved_path, symlink)
        self.best_vals: Dict[str, Tuple[bool, Optional[int], Optional[str], Optional[str]]] = \
            self._init_best_vals(metrics_to_track)
        # Maps path to number of pointers (i.e. the number of metrics that still have this as best model
        self.saved_models: Dict[str, int] = dict()

        log.info(f'Saver will track (metric, maximize?)\n {metrics_to_track}')

    def _init_best_vals(self, metrics_to_track: List[Tuple[str, bool]]):
        best_vals = dict()
        for metric, maximize_metric in metrics_to_track:
            best_vals[metric] = (maximize_metric, None, None, None)
        return best_vals

    def _dump_json(self, filename, object):
        with open(filename, 'w') as fp:
            json.dump(object, fp, cls=NumpyArrayEncoder)

    def save_if_best(self,
                     epoch: float,
                     trainer,
                     metric_dict: Dict[str, Union[int, float]],
                     preds: Optional[List[Tuple]] = None,
                     save_model: bool = True):
        """
        Save model and outputs

        When a single epoch / model checkpoint maximizes multiple metrics, we will save only one time,
        keeping pointers and garbage collecting when that epoch no longer maximizes any metric

        :param epoch:
        :param trainer:
        :param metric_dict:
        :param preds:
        :param save_model: whether to save the actual model (otherwise saves only outputs)
        :return:  Noreturn
        """

        if preds is None and not save_model:
            log.warning(f'Nothing to save (no preds and not saving model)')
            return

        # file extensions that will be generated; used for autoremoval
        file_list = [".txt", ".preds.json"]
        if save_model:
            file_list.append("")        # the base file is also written

        metric_dict.update({"epoch": epoch})    # always saves on most recent epoch

        def save_most_recent():
            """
            Actually save the model and write all the files
            """
            checkpoint_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth.tar')
            log.info(f'Saving most recent model at epoch={epoch} to {checkpoint_path}')
            self.saved_models[checkpoint_path] = 0  # record that we are tracking this path

            # do the actual saving
            readme_dict = metric_dict.copy()
            readme_dict.update(dict(name=trainer.config.name))
            self._dump_json(checkpoint_path + ".txt", readme_dict)
            if save_model:
                ckpt_dict = trainer.make_ckpt_dict()
                torch.save(ckpt_dict, checkpoint_path)
            if preds is not None:
                self._dump_json(checkpoint_path + ".preds.json", preds)
            return checkpoint_path

        model_save_path = None      # we will only save once per checkpoint call
                                    # but potentially multiple metrics will point to this

        for metric_name, (maximize_metric, best_val, prev_path, prev_sym_path) in self.best_vals.items():
            # prev_path tracks the reference that we are using for this metric
            new_val = metric_dict.get(metric_name, None)
            if new_val is None:  # nothing to save since metric was not reported
                continue
            if best_val is not None:  # then need to compare
                if maximize_metric and not new_val > best_val:
                    continue
                if not maximize_metric and not new_val < best_val:
                    continue

            # we should save
            log.info(f"Best metric for {metric_name} = {new_val} at epoch={epoch}")
            if prev_path is not None:
                self.saved_models[prev_path] -= 1  # first decrement pointer to previous
                # remove the symlinks
                for ext in file_list:
                    try: os.remove(prev_sym_path + ext)
                    except OSError: pass

            # if this is the first metric maximized by this, then we need to actually save
            # otherwise, we're just adding another pointer
            if model_save_path is None:
                model_save_path = save_most_recent()

            # increment tracking on this path
            self.saved_models[model_save_path] += 1  # increment the counter to new one
            metric_name_safe = metric_name.replace("/", "_")

            # prepare the sym link
            sym_path = os.path.join(self.save_dir, f'ckpt_{metric_name_safe}_{new_val:.2f}_{epoch}.pth.tar')

            # record that we saved this
            self.best_vals[metric_name] = (maximize_metric, new_val, model_save_path, sym_path)

            # actually make the symlinks
            for ext in file_list:
                try:
                    if os.path.isfile(model_save_path + ext):   # e.g. if we didn't save preds
                        os.symlink(model_save_path + ext, sym_path + ext)
                except: pass
            # todo: i think we were saving twice before
            # self._dump_json(sym_path + ".txt", metric_dict)
            # if preds is not None:
            #     self._dump_json(sym_path + ".preds.json", preds)

        # Garbage collect any stale model pointers
        # note that we previously removed the symlinks above, this is just the actual files now
        for k in list(self.saved_models.keys()):
            v = self.saved_models[k]
            if v == 0:
                for ext in file_list:
                    try: os.remove(k + ext)
                    except OSError:
                        log.warning(f'Failed to remove checkpoint {k}')
                self.saved_models.pop(k)
