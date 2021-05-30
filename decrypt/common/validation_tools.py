from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from glob import glob
from pprint import pp
from typing import *

from tqdm import tqdm

from .puzzle_clue import GuardianClue

logging.getLogger(__name__)

# In case gsheets isn't set up
class DummyWriter:
    def __init__(self):
        self._did_warn = False

    def write_row(self, Any):
        if not self._did_warn:
            logging.warning(f'No gsheets writer exists. No rows will be written. '
                            f'This warning will be printed only once.')
            self._did_warn = True


try:
    # TODO: this needs to be setup to get automatic output writing
    from common.gsheets import Writer
    _vt_writer = Writer()
except:
    logging.warning('No gsheets writer is configured')
    _vt_writer = DummyWriter()


@dataclass
class ModelPrediction:
    @dataclass
    class LabelsDict:
        num_words: Optional[int] = None

    idx: int                # clue index in the val set
    input: str
    target: str
    greedy: str             # greedy output (currently not used)
    sampled: List[str]
    labels: LabelsDict = None
    model_eval: Optional[ModelEval] = None  # will be populated by eval()

    def __post_init__(self):
        if self.labels is None:
            self.labels = self.LabelsDict()

@dataclass
class ModelEval:
    """
    The result of a single input/output to one of our models
    """

    @dataclass
    class Metrics:
        """
        - this is created by the eval() call
        - these fields can all be aggregaged (sum) in a final aggregate call, so they must be
        either bool or int
        """
        generate_none: bool = False     # whether the sample has len 0
        generate_few: bool = False      # whether sample has less than sample_len outputs
        filtered_few: bool = False      # whether the filtered set has fewer than sample outputs

        # after filtering
        top_match: bool = False                 # if top match from sample (after len filter) is correct
        top_match_none: bool = False            # if we didnt have anything the correct len
        top_match_wordct_correct: bool = False  # whether top match has corr. # words (indicated by spaces)
        top_match_len_correct: bool = False     # whether top match has corr len
        top_10_after_filter: bool = False

        in_sample: bool = False    # whether answer in sample (generally top 10)
        in_filtered: bool = False  # whether any of filter is correct
        top_sample_result_len_correct: bool = False     # whether top sampled output has correct len
        top_sample_result_wordct_correct: bool = False     # whether top sampled output has correct len
        # removed since top_match is always filtered to correct len

        sample_len: int = 0         # min(total number of samples generated, sample_len); used for division in agg
        sample_len_pre_truncate: int = 0
        filter_len_pre_truncate: int = 0

    @dataclass
    class SampleMetrics:
        sample_len_correct: int = 0     # num in sample with correct len
        sample_wordct_correct: int = 0  # number of words in sample corr # words

    # actual fields
    metrics = None
    sample_metrics = None       # will be divided by num_rows * sample len

    def __post_init__(self):
        self.metrics = self.Metrics()
        self.sample_metrics = self.SampleMetrics()

    def all_items(self):
        for d in [self.metrics.__dict__, self.sample_metrics.__dict__]:
            for k,v in d.items():
                yield k, v


def filter_to_len(tgt_orig: str, sampled: List[str],
                  do_filter=True) -> \
    Tuple[List[Tuple[str,str]], List[Tuple[str, str]]]:
    def deduped_filtered_list(input_list: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
        seen = set()
        ret = []
        for spaces, no_spaces in input_list:
            if no_spaces in seen:
                continue
            # not seen yet
            seen.add(no_spaces)
            ret.append((spaces, no_spaces))
        return ret

    tgt_no_spaces = tgt_orig.replace(' ', '').strip()
    tgt_len_no_spaces = len(tgt_no_spaces)
    samples_no_spaces = list(map(lambda x: x.replace(' ','').strip(), sampled))
    samples_tuple: List[Tuple[str, str]] = list(zip(sampled, samples_no_spaces))

    filtered: List[Tuple[str,str]] = list(filter(lambda x: len(x[1]) == tgt_len_no_spaces, samples_tuple))
    filtered = deduped_filtered_list(filtered)

    if do_filter:
        return samples_tuple, filtered
    else:
        return samples_tuple, samples_tuple


# todo: should be part of the ModelEval class
def eval(mp: ModelPrediction,
         sample_size=10,
         filter_sample_size=10,
         pre_truncate: Optional[int] = None,
         do_filter=True) -> ModelEval:
    """
    1) add labels:
        - number of words == number of spaces

    2) produce a model eval
        - remove spaces so that anything that is equal up to spaces is treated the same
        - we also dedupe the filtered list


    # todo: correct up to spaces

    :param mp:
    :param sample_size:
    :param filter_sample_size:
    :return:
    """

    # verify it's reasonable
    assert len(mp.target) > 0

    # add labels
    mp.labels.num_words = mp.target.count(' ')

    ###
    # do eval
    ###
    output_eval = ModelEval()

    # pretruncate
    if pre_truncate is not None:
        sampled = mp.sampled[:pre_truncate]
    else:
        sampled = mp.sampled


    # sample len: record stats and then limit to length of sample_size
    if not len(sampled) > 0:
        output_eval.metrics.generate_none = True
    if len(sampled) < sample_size:
        output_eval.metrics.generate_few = True
    output_eval.metrics.sample_len = len(sampled[:sample_size])
    output_eval.metrics.sample_len_pre_truncate = len(sampled)

    # various setup
    tgt_orig = mp.target
    tgt_no_spaces = tgt_orig.replace(' ', '').strip()
    tgt_len_no_spaces = len(tgt_no_spaces)

    # these are a non-deduped list of tuples (orig, orig_no_spaces) and a deduped list of the same
    samples_tuple, filtered = filter_to_len(tgt_orig, sampled, do_filter=do_filter)

    if len(filtered) == 0:
        output_eval.metrics.top_match_none = True
    top_answer = None if len(filtered) == 0 else filtered[0]

    if top_answer is not None:
        if top_answer[0].count(' ') == mp.labels.num_words:
            output_eval.metrics.top_match_wordct_correct = True
        if len(top_answer[1]) == len(tgt_no_spaces):
            output_eval.metrics.top_match_len_correct = True
        if top_answer[1] == tgt_no_spaces:
            output_eval.metrics.top_match = True

    # filtered sample
    output_eval.metrics.filter_len_pre_truncate = len(filtered)
    if len(filtered) < filter_sample_size:
        output_eval.metrics.filtered_few = True
    if tgt_no_spaces in [x[1] for x in filtered[:filter_sample_size]]:
        output_eval.metrics.top_10_after_filter = True
    if tgt_no_spaces in [x[1] for x in filtered]:
        output_eval.metrics.in_filtered = True

    # sample metrics
    for answer, answer_no_spaces in samples_tuple[:sample_size]:
        if tgt_no_spaces == answer_no_spaces:
            output_eval.metrics.in_sample = True
        if len(answer_no_spaces) == tgt_len_no_spaces:
            output_eval.sample_metrics.sample_len_correct += 1
        if answer.count(' ') == mp.labels.num_words:
            output_eval.sample_metrics.sample_wordct_correct += 1
    # top sample output length
    if len(samples_tuple) > 0:      # might have zero generations
        answer, answer_no_spaces = samples_tuple[0]
        if len(answer_no_spaces) == tgt_len_no_spaces:
            output_eval.metrics.top_sample_result_len_correct = True
        if answer.count(' ') == mp.labels.num_words:
            output_eval.metrics.top_sample_result_wordct_correct = True

    return output_eval


def aggregate(mp_set: List[ModelPrediction],
              info_string: str = "",
              filter_fcn: Optional[Callable] = None,
              length_check: Optional[int] = None):
    """

    :param mp_set:
    :param filter_fcn: Use to filter to subset of clues
    :return:
    """
    if length_check is not None:
        assert len(mp_set) == length_check

    ctr = Counter()
    for mp in mp_set:
        assert mp.model_eval is not None
        if filter_fcn is not None and not filter_fcn(mp):
            continue
        ctr['total'] += 1
        model_eval = mp.model_eval
        for k,v in model_eval.all_items():
            ctr[k] += int(v)

    if filter_fcn:
        print(f'With filter {filter_fcn.__name__}')

    dummy = ModelEval()
    for k in dummy.metrics.__dict__.keys():
        ctr[f'agg_{k}'] = ctr[k] / ctr['total']

    for k in dummy.sample_metrics.__dict__.keys():
        ctr[f'agg_{k}'] = ctr[k] / (ctr['sample_len'])

    pp(sorted(ctr.items(), key=lambda x: x[0]))
    return ctr


# prefix with agg to get the averaged value
# these need to match the columns in the spreadsheet
k_output_list = [
    'agg_top_match',
    'agg_top_10_after_filter',
    # '_',
    'agg_top_match_none',
    'agg_filtered_few',

    'agg_in_sample',
    'agg_sample_len_correct',
    'agg_sample_wordct_correct',
    'agg_top_sample_result_len_correct',
    'agg_top_sample_result_wordct_correct',
    ('agg_sample_len_pre_truncate', 1),     # mark as not a percentage
    ('agg_filter_len_pre_truncate', 1),
    '_',
    'agg_in_filtered',
    'agg_top_match_wordct_correct',
    'agg_top_match_len_correct',
    ('total', 1)
]

def write_row(label: str, ctr: Dict):
    csv_output_list = [label] + [''] * 3

    # latex_str = "label "
    for x in k_output_list:
        if x == '_':
            csv_output_list.append('')
            continue
            # output += '\t'

        if not isinstance(x, tuple):
            val = ctr[x]
            val = round(float(val)*100, 1)
        else:
            val = ctr[x[0]]
            val = round(float(val), 1)

        csv_output_list.append(val)
        # latex_str += f' & {val} '
        # output += f'{ctr[x]}\t'
    # latex_str += '\\\\'
    # _vt_writer.write_row([latex_str])
    _vt_writer.write_row(csv_output_list)
    # output += '\n'
    # return output


def all_aggregate(mp_set: List[ModelPrediction],
                  label="empty",
                  filter_fcn: Optional[Callable[[ModelPrediction],bool]] = None,):
                  # do_multi_split=False):
    def multiword_filter(mp: ModelPrediction):
        if ' ' in mp.target:
            return True
        return False

    c = aggregate(mp_set, filter_fcn=filter_fcn)
    write_row(label, c)
    # if do_multi_split:
    #     aggregate(mp_set, filter_fcn=multiword_filter)

###
# set inclusion filter functions
###

def make_set_filter(labels_set: Dict[str, Set[int]], type: str) -> Callable:
    inclusion_set = labels_set[type]

    def check_inclusion(mp: ModelPrediction):
        if mp.idx in inclusion_set:
            return True
        return False

    return check_inclusion


##
# deits
##
def load_deits(val_set: Union[List[GuardianClue], List[str]],
               file_dir):
    """

    :param val_set: either list of guardian clue or string which is soln with spaces
        note that we index using the index in the val_set list, not the real clue index
    :param file_dir:
    :return:
    """
    def backfill_deits(mp: ModelPrediction):
        assert mp.target == ""
        assert mp.idx != ""

        orig_clue = val_set[mp.idx]
        try:
            mp.target = orig_clue.soln_with_spaces
        except:
            mp.target = orig_clue

        # print(orig_clue.clue, )

    j = []
    print(len(glob(file_dir)))
    for filename in tqdm(glob(file_dir)):
        try:
            with open(filename, 'r') as f:
                new_json = json.load(f)
        except Exception as e:
            new_json = None
            print(e)
            print(filename)
        if new_json is not None:
            j.extend(new_json)


    print(f'Loaded: {len(j)}')
    if len(val_set) != len(j):
        print(f'valset = {len(val_set)} != json = {len(j)}')

    # eval (and backfill)
    model_outputs = []
    idx_set = set()
    c = Counter()
    for d in tqdm(j):
        idx, input, tgt, greedy, sampled, did_timeout, did_error = d
        assert idx not in idx_set
        idx_set.add(idx)
        c['timeout'] += did_timeout
        c['error'] += did_error

        mp = ModelPrediction(idx, input, tgt, greedy, sampled)
        if len(mp.target) == 0:
            backfill_deits(mp)
            c['backfill'] += 1
        else:
            orig_clue = val_set[mp.idx]
            try:
                tgt_orig = orig_clue.soln_with_spaces
            except:
                tgt_orig = orig_clue
            assert mp.target == tgt_orig

        # correct the indices for set inclusion / exclusion
        try:
            mp.idx = val_set[mp.idx].idx
        except:
            mp.idx = -1

        mp.model_eval = eval(mp)
        model_outputs.append(mp)
    print(c)

    for i in range(len(val_set)):
        if i not in idx_set and i % 100 == 0:
            print(i)

    return model_outputs


###
# t5
###

def load_and_run_t5(fname, label=None, filter_fcn=None, pre_truncate=None,
                    do_length_filter=True):
    def load_t5(json_out_file: str, pre_truncate=None):
        with open(json_out_file, 'r') as f:
            json_blob = json.load(f)

        # eval (and backfill)
        model_outputs = []
        idx_set = set()
        for d in json_blob:
            idx, input, tgt, greedy, sampled = d
            assert idx not in idx_set
            idx_set.add(idx)

            mp = ModelPrediction(idx, input, tgt, greedy, sampled)
            mp.model_eval = eval(mp, pre_truncate=pre_truncate,
                                 do_filter=do_length_filter)
            model_outputs.append(mp)

        print(len(model_outputs))
        # can use to verify there is an output for all inputs
        # for i in range(28476):
        #     if i not in idx_set:
        #         print(i)

        return model_outputs
    if label is None:
        label = fname
    data = load_t5(fname + '.json',
                   pre_truncate=pre_truncate)
    all_aggregate(data, label=label, filter_fcn=filter_fcn)
