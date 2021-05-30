from __future__ import annotations

import logging
from collections import Counter
from typing import Tuple, List, Dict, Set, Callable, Optional, Union, Any

import torch

from .util import ProcessedBatch, PerBatchValStep

log = logging.getLogger(__name__)


# todo: metrics should be callable metaclass function
class MetricsDict:
    def __init__(self, avg_metrics=None, no_avg_metrics=None):
        if avg_metrics is None:
            avg_metrics = dict()
        if no_avg_metrics is None:
            no_avg_metrics = dict()
        self.avg_metrics = avg_metrics
        self.no_avg_metrics = no_avg_metrics


MetricFcn = Callable[[PerBatchValStep, ProcessedBatch],
                     Union[MetricsDict,
                           Tuple[MetricsDict, torch.Tensor]]]


class MetricsPredsWrapper:
    """
    If label is given, all metrics will be prefixed with the label

    For individual metrics, label should be tied to the val set and passed with update

    """

    def __init__(self, metrics_dict: Optional[MetricsDict] = None,
                 label: str = "",
                 avg_divisor: Optional[int] = None):
        if metrics_dict is None:
            metrics_dict = MetricsDict()
        self.md = metrics_dict  # should be accessed only via get_all_metrics()

        self.preds: List[Tuple[str, str, str, Any, ...]] = []       # input, target, greedy, sampled
        self.label = ""
        if label != "":
            self.label = label + "/"

        self.avg_divisor = avg_divisor

    def get_all_metrics(self, avg_divisor: Optional[int] = None) -> Tuple[str, float, float]:
        """
        Return the k, value (averaged if necessary) and the original value
        """
        assert avg_divisor is not None or self.avg_divisor is not None
        if avg_divisor is None:
            avg_divisor = self.avg_divisor

        for k, v in self.md.avg_metrics.items():
            yield self.label + k, v / avg_divisor, v
        for k, v in self.md.no_avg_metrics.items():
            yield self.label + k, v, v

    def get_all_metrics_dict(self) -> Dict[str, float]:
        ret_dict = dict()
        for k, _, v in self.get_all_metrics():
            ret_dict[k] = v
        return ret_dict

    def update_for_batch(self,
                         metric_fcns: List[MetricFcn],
                         valstep_batch: PerBatchValStep,
                         pbatch: ProcessedBatch,
                         metric_label: str = ""):
        if metric_label != "":
            metric_label = metric_label + "/"

        # update predictions
        if pbatch.idxs is not None:
            preds = list(zip(pbatch.idxs,       # will be populated for json DL, or for idxs provided file
                             pbatch.orig_text_input,
                             pbatch.orig_text_output,
                             valstep_batch.outputs_greedy,
                             valstep_batch.outputs_sampled))
        else:   # todo(json): deprecate this
            raise NotImplemented
            # preds = list(zip(pbatch.orig_text_input,
            #                 pbatch.orig_text_output,
            #                 valstep_batch.outputs_greedy,
            #                  valstep_batch.outputs_sampled))
        self.preds.extend(preds)

        # update metrics
        for f in metric_fcns:
            result = f(valstep_batch, pbatch)
            if type(result) == tuple:
                raise NotImplemented('no longer support result tuple')
                # new_metrics_dict, correct_indices = result
                # if correct_indices is not None:
                #     preds_correct = [self.preds['greedy'][i] for i in correct_indices.tolist()]
                #     self.preds[f'{f.__name__}_correct'].extend(preds_correct)
            else:
                new_metrics_dict = result

            self.update(new_metrics_dict, metric_label)

    # todo: should be internal only method
    def update(self, new_dict: MetricsDict, label=""):
        # need to iterate since there are floats (otherwise use counter)
        for k, v in new_dict.avg_metrics.items():
            self.md.avg_metrics[label + k] = self.md.avg_metrics.get(label + k, 0) + v
        for k, v in new_dict.no_avg_metrics.items():
            self.md.no_avg_metrics[label + k] = self.md.no_avg_metrics.get(label + k, 0) + v

    def add_val(self, key, val, avg: bool, label=""):
        if label != "":
            label = label + "/"
        if avg:
            self.md.avg_metrics[label + key] = val
        else:
            self.md.no_avg_metrics[label + key] = val


# see test_util for test verification

# from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Dice%27s_coefficient#Python

def _remove_spaces(s: str) -> str:
    return s.replace(' ','').strip()

def compute_metrics_sampled_primary(*args) -> MetricsDict:
    return compute_metrics_sampled(*args,
                                   primary_only=True)

def compute_metrics_sampled(valstep_batch: PerBatchValStep,
                            pbatch: ProcessedBatch,
                            label_sets: Optional[List[str, Set[int]]] = None,
                            primary_only: bool = False) -> MetricsDict:
    sampled_outputs: List[List[str]] = valstep_batch.outputs_sampled
    assert sampled_outputs is not None
    greedy_outputs: List[str] = valstep_batch.outputs_greedy
    tgt_outputs: List[str] = pbatch.orig_text_output
    idxs = pbatch.idxs  # optional tensor

    #########
    # special checks on idxs and label_sets when we are doing label metrisc
    if label_sets is not None:
        assert idxs is not None
        assert label_sets is not None and len(label_sets) > 0
        assert type(idxs[0].item()) is int, f'{idxs[:10]}'
        # label_list = list(label_set)
        # assert type(label_list[0]) is int, f'{label_list[:10]}'
        label_counters = Counter()
        idxs = idxs.tolist()
    # do this so that we can zip it in the for loop, even if we're not doing it
    if idxs is None:
        idxs = [-1] * len(greedy_outputs)
    ###########

    # aggregates
    ct_greedy = 0  # num times tgt matches greedy (after lowercase)
    ct_sampled = 0  # num times tgt in the sample set
    ct_top_sampled = 0  # num where, after length filter, the top answer is correct
    ct_top_5_sampled = 0
    # num_lost = 0  # when greedy gets it but sample doesn't

    cum_num_correct_length = 0.0  # will turn into pct correct by divide by num_seq
    cum_num_words_correct = 0.0

    num_sampled = len(sampled_outputs[0])

    # iterate through batch
    for g, sample_list, t, clue_idx in zip(greedy_outputs, sampled_outputs, tgt_outputs, idxs):
        # lower case, remove spaces, strip
        sample_list: List[str] = list(map(lambda x: x.lower(), sample_list))
        sample_list_no_spaces: List[str] = list(map(_remove_spaces, sample_list))
        g = _remove_spaces(g)
        tgt_no_spaces = _remove_spaces(t)
        tgt_len_no_spaces = len(tgt_no_spaces)

        # filter to correct len
        samples_no_spaces_filtered = list(filter(lambda x: len(x) == tgt_len_no_spaces, sample_list_no_spaces))

        # inclusion / exclusion
        # in_greedy = False
        # in_sample = False
        if g == tgt_no_spaces:
            ct_greedy += 1
            # in_greedy = True

        # idx = 0
        for idx, samp in enumerate(samples_no_spaces_filtered):
            if samp == tgt_no_spaces:
                # in_sample=True
                ct_sampled += 1
                if idx == 0:
                    ct_top_sampled += 1
                if idx < 5:
                    ct_top_5_sampled += 1
                break
        # if in_greedy and idx != 0:
        #     num_lost += 1

        # how close to correct length
        cum_num_correct_length += len(samples_no_spaces_filtered)

        # num words
        tgt_spaces = t.count(' ')
        for samp in sample_list:
            if samp.count(' ') == tgt_spaces:
                cum_num_words_correct += 1

        ########
        # for labels
        if label_sets is not None:
            raise NotImplemented
            # assert type(clue_idx) is int
            # for name, label_set in label_sets:
            #     if clue_idx in label_set:
            #         if in_sample:
            #             label_counters[name + "_label"] += 1
            #         # ct_succ_with_label += 1
        #########

    # scale cumulatives by the number of sequences generated
    pct_correct_length = cum_num_correct_length / num_sampled
    pct_correct_wordct = cum_num_words_correct / num_sampled

    if primary_only:
        ret_dict: MetricsDict = MetricsDict(
            avg_metrics=dict(
                num_match_in_sample=ct_sampled,
                num_match_top_sampled=ct_top_sampled,
            )
        )
    else:
        ret_dict: MetricsDict = MetricsDict(
            avg_metrics=dict(
                num_exact_match_char_2=ct_greedy,

                num_match_in_sample=ct_sampled,
                num_match_top_sampled=ct_top_sampled,
                num_match_top_5_sampled=ct_top_5_sampled,

                # num_lost_sample_vs_greedy=num_lost,  # when greedy correct but top sampled differs

                # will be averaged
                pct_correct_length=pct_correct_length,
                pct_correct_wordct=pct_correct_wordct,
            )
        )
    if label_sets is not None:
        raise NotImplemented
        # # don't average
        # # todo: should have a different divisor
        # ret_dict.no_avg_metrics = dict(
        #     **label_counters)
    return ret_dict