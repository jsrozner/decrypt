"""
for XD clue set (i.e. ACW, american crossword)
"""

import csv
import logging
from collections import Counter
from typing import *

from tqdm import tqdm

from decrypt.common.puzzle_clue import BaseClue, make_stc_map, filter_clues
from decrypt.common.util_spellchecker import SpellChecker

logging.getLogger(__name__)


def xd_load_and_filter_clues(filename,
                             remove_if_not_in_dict=False,
                             strip_trailing_period=True,
                             remove_questions=True,
                             remove_likely_abbreviations=True,
                             remove_fillin=True,
                             # try_word_split=False,
                             ) -> List[BaseClue]:

    with open(filename, "r") as f:
        if remove_if_not_in_dict:
            sc = SpellChecker(init_twl_dict=True,
                              init_enchant_dict=True)
        else:
            sc = None

        rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        _ = next(rd)    # skip first header line
        ctr = Counter()
        clue_list = []

        for row in tqdm(rd):
            try:
                answer, clue = row[2], row[3]
            except:
                continue

            if answer == "" or clue == "" or len(clue) < 3:
                ctr["empty"] += 1
                continue

            if remove_fillin and ("_" in clue or "--" in clue):
                ctr["fillin"] += 1
                continue

            if remove_if_not_in_dict and not sc.check_word(answer):
                ctr["not_in_dict"] += 1
                continue

            # if "-Across" in c.clue or "-Down" in c.clue:
            if "Across" in clue or "Down" in clue:
                ctr["ref"] += 1
                continue

            if remove_likely_abbreviations:
                if clue[-1] == "." and len(answer) < 4:
                    ctr["removed_likely_abbrev"] += 1
                    continue

            if remove_questions and clue[-1] == "?":
                ctr["question word"] += 1
                continue

            if strip_trailing_period and clue[-1] == ".":
                # this was implemented wrong originally - truncated the answer rather than clue
                ctr["removed_trailing_period"] += 1
                clue = clue[:-1]
                # answer = answer[:-1]

            # should have no spaces
            answer = answer.replace(' ', '')

            c = BaseClue.from_clue_and_soln(clue, answer)
            clue_list.append(c)

    logging.info(ctr)
    print(ctr)
    logging.info(f'Filtered to {len(clue_list)} clues')
    return clue_list


# modeled after get_clean_clues (for guardian)
def get_clean_xd_clues(filename,
                       remove_if_not_in_dict=True,
                       do_filter_dupes=True) \
    -> Tuple[Dict[str, List[BaseClue]], List[BaseClue]]:

    logging.info(f'loading xd (ACW) set from {filename}')
    all_clue_list = xd_load_and_filter_clues(filename,
                                             remove_if_not_in_dict=remove_if_not_in_dict,
                                             strip_trailing_period=True,
                                             remove_questions=True,
                                             remove_likely_abbreviations=True,
                                             remove_fillin=True)

    # generate soln to clue map
    # soln:str -> List[gc]
    soln_to_clue_map = make_stc_map(all_clue_list)

    # Remove anything that is exactly the same up to small diffs
    # removes 1610 normalized clues
    if do_filter_dupes:
        soln_to_clue_map, all_clue_list = filter_clues(soln_to_clue_map)

    # add indices and a note about dataset
    for idx, c in enumerate(all_clue_list):
        c.idx = idx
        c.dataset = filename

    # print the distribution
    ctr = Counter()
    for c in all_clue_list:
        ctr[len(c.lengths)] += 1
    logging.info(ctr)

    # Verify same length
    assert sum(map(len, soln_to_clue_map.values())) == len(all_clue_list)

    return soln_to_clue_map, all_clue_list
