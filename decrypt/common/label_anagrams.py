import string
from collections import defaultdict
from typing import *

from tqdm import tqdm

import config
from decrypt.scrape_parse.guardian_load import load_guardian_splits


def make_label_set():
    _, all_clues, (_, _, _) = load_guardian_splits(config.DataDirs.Guardian.json_folder, verify=True)
    labels: Dict[str, Set[int]] = defaultdict(set)       # set of the indices for this type
    any_label = set()
    def add_to_labels(name, idx, verify=True):
        if verify:
            assert idx not in any_label
        any_label.add(idx)
        labels[name].add(idx)

    class PunctStripper:
        """
        use to strip punctuation from clues (since punct is not part of outputs)
        """
        def __init__(self):
            self.table_spaces = str.maketrans('','',string.punctuation + " ")   # map punct and space to ''
            self.punct_to_space_table = str.maketrans(string.punctuation,' '*len(string.punctuation))   # map punct to space
        def strip(self, s: str, strip_spaces=True):
            """
            :param s:
            :param strip_spaces: if true, will remove spaces; otherwise all punct will be substituted
            with a space, which is important for generating anagram outputs
            :return:
            """
            if strip_spaces:
                return s.translate(self.table_spaces)
            else:
                return s.translate(self.punct_to_space_table)
    ps = PunctStripper()

    # will find hiddens / reversals (which are either direct, or direct reverse)
    # the anagrams that result potentially take single letters from the start or end of another word

    for sc in tqdm(all_clues):
        c = ps.strip(sc.clue).lower()
        s = sorted(sc.soln.lower())
        tgt_len = len(s)
        for idx in range(0, len(c) - tgt_len + 1):
            sub_part = c[idx:idx+tgt_len]
            # hidden if directly occurs
            if sub_part == sc.soln.lower():
                add_to_labels('hidden', sc.idx)
                break
            # reverse if occurs backward
            if sub_part == sc.soln.lower()[::-1]:
                add_to_labels('reverse', sc.idx)
                break
            # direct anagram if occurs directly in clue once spaces and punct removed
            if sorted(sub_part) == s:
                add_to_labels('anag_direct', sc.idx)
                break

    return labels
