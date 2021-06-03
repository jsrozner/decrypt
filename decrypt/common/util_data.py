"""
Utils to
- dump obj to file (or retrieve from file)
- train/test split function for cryptics
- cryptic Parsed Examples => Data for training models
"""
import json
import logging
import os.path
from pprint import pformat
from typing import *
import decrypt.config as config

from tqdm import tqdm

from decrypt.common.puzzle_clue import (
    Seq2seqDataEntry,
    BaseClue
)
from .anagrammer import Anagrammer

log = logging.getLogger(__name__)

k_data_names = ["train", "val", "test"]


def _check_overwrite(filename):
    if os.path.isfile(filename):
        raise FileExistsError("Cannot write since file_name already exists and overwrite not specified")
#######
# Dataset generation
#######

def write_json_tuple(json_tuple: List[List],
                     comment: str,
                     export_dir,
                     overwrite:bool = False,
                     mod_fn: Optional[Callable] = None):
    assert 1 <= len(json_tuple) <= 3, len(json_tuple)

    def write_json_to_file(json_dict: List, path):
        if not overwrite:
            _check_overwrite(path)
        with open(path, 'w') as fh:
            json.dump(json_dict, fh)

    os.makedirs(export_dir, exist_ok=True)
    for json_out, filename in zip(json_tuple, k_data_names):
        tgt_path = os.path.join(export_dir, filename + ".json")
        write_json_to_file(json_out, tgt_path)

    # write a description
    file_path = os.path.join(export_dir, "README.txt")
    with open(file_path, "w") as f:
        f.write(comment)
        lengths = list(map(len, json_tuple))
        f.write(f'\nTotal: {sum(lengths)}\n'
                f'splits: {lengths}')
        f.write("\n\n")
        sample_set = ""
        for entry in json_tuple[0][:3]:
            sample_set += f'{pformat(entry)}\n'
        f.write(sample_set + "\n\n")
        print(sample_set)

        if mod_fn is not None:
            f.write(f'\nMod fcn applied: {mod_fn.__name__}')

    log.info('Finished writing all files')


def clue_list_tuple_to_train_split_json(
    clue_list_tuple: Tuple[List[BaseClue], ...],    # train, val, test
    comment: str,
    export_dir,
    mod_fn: Optional[Callable] = None,
    overwrite: bool = False):
    assert 1 <= len(clue_list_tuple) <= 3, len(clue_list_tuple)

    def make_json_list(l: List[BaseClue]):
        out_list = []
        for bc in tqdm(l):
            data_entry = Seq2seqDataEntry.from_base_clue(bc, mod_fn=mod_fn)
            json_entry = Seq2seqDataEntry.to_json_dict(data_entry)
            out_list.append(json_entry)
        return out_list

    json_output_tuple = list(map(make_json_list, clue_list_tuple))

    out_ex = json_output_tuple[0][0]
    log.info(f'Source target mapping:\n'
             f'\t{out_ex["input"]} => {out_ex["target"]}\n')

    write_json_tuple(json_output_tuple, comment, export_dir, overwrite=overwrite, mod_fn=mod_fn)

###
# Anagrams
###

def get_anags(max_num_words=1) -> List[List[str]]:
    """
    Return List where each element is a list of words that map to the same set of letters
    """
    anag = Anagrammer(str(config.DataDirs.Generated.anagram_db))   # system autoappends db
    # First populate the anagrams
    anag._populate_possible_anagrams()

    ret_anags = []
    for anag_set in anag._possible_anagrams:
        one_word_anags, multi_word_anags = anag_set.get_lists()     # get one words only
        all_anags = one_word_anags + multi_word_anags
        if max_num_words > 0:
            # list of lists; num lists is the number of words in the anag set
            all_anags = filter(lambda x: len(x) <= max_num_words, all_anags)

        # for multi-word anags, we join them together
        all_anags = list(map(lambda x: " ".join(x), all_anags))
        if len(all_anags) > 1:      # make sure there are at least two realizations of the letter set
            # flattened = [w for realizations in all_anags for w in realizations]       # flatten
            ret_anags.append(all_anags)

    print(len(ret_anags))          # unique sets of letters that produce more than one realized anagram
    print(len(anag._possible_anagrams)) # unique sets of letters that produce a single or multiword anagram
    print(sum(map(lambda x: len(x), ret_anags)))       # all possible words that have at least one other one word anag
    print(ret_anags[0])            # example

    return ret_anags
