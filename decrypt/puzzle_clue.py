from __future__ import annotations

import logging
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import *

from tqdm import tqdm

log = logging.getLogger(__name__)


##################
# Puzzle and Clue related classes / datastructures for easy manipulation ###
##################
@dataclass
class BaseClue:
    clue: str
    lengths: List[int]
    soln: str
    soln_with_spaces: str = field(init=False)  # soln string, but has spaces between words for multi-word answers
    idx: int = field(init=False)  # unique index in the set
    dataset: str = field(init=False)  # source dataset

    def __post_init__(self):
        self.soln = self.soln.lower()
        self.__populate_soln_with_spaces()
        self.idx = -1  # initially set to -1; but will be set in get_clean_clues()
        self.dataset = ""

    def __populate_soln_with_spaces(self):
        soln_with_spaces = ""
        idx = 0
        if len(self.lengths) > 1:
            for l in self.lengths:
                soln_with_spaces += self.soln[idx: idx + l] + " "
                idx += l
            soln_with_spaces = soln_with_spaces.strip()
        else:
            soln_with_spaces = self.soln
        self.soln_with_spaces = soln_with_spaces

    def clue_with_lengths(self, punct=","):
        return f'{self.clue} ({punct.join(map(str, self.lengths))})'

    @classmethod
    def from_clue_and_one_word_soln(cls, clue: str, soln: str):
        return cls(clue=clue,
                   lengths=[len(soln)],
                   soln=soln)

    @classmethod
    def from_clue_and_soln(cls, clue: str, soln: str):
        splits = soln.split(' ')
        lengths = list(map(lambda x: len(x.strip()), splits))
        return cls(clue=clue,
                   lengths=lengths,
                   soln=soln)

    @classmethod
    def from_json(cls, json_obj: Dict) -> BaseClue:
        json_obj_no_soln_with_spaces = json_obj.copy()
        json_obj_no_soln_with_spaces.pop('soln_with_spaces')
        return cls(**json_obj_no_soln_with_spaces)


@dataclass
class ClueWithGridInfo(BaseClue):
    across_or_down: str  # "across" or "down"
    pos: Tuple[int, int]  # row, col

    @classmethod
    def from_json(cls, json_list: List):
        return cls(*json_list)


@dataclass(order=True)
class GuardianClue(ClueWithGridInfo):
    # identifiers
    unique_clue_id: str  # puzzleid_clue_id.. of form: 21465_1-across, e.g.
    type: str  # "cryptic" or "quiptic"
    number: int  # should be the end of id; a unique ID
    id: str  # e.g. crosswords/cryptics/21465
    # extra metadata
    creator: Optional[str]  # json -> creator -> name
    orig_lengths: str  # sometimes there is a dash instead of a comma separator
    lengths_punctuation: Set[str]  # the punctuation it contains

    @classmethod
    def to_json_dict(cls, gc: GuardianClue) -> Dict:
        return dict(
            clue=gc.clue,
            soln=gc.soln,
            soln_with_spaces=gc.soln_with_spaces,
            lengths=gc.lengths
        )


####
# Functions to go from guardian clue json to a filtered list for use in datasetes
# todo: roughly copied from guardian_scrape > __main__ ; clean up; i.e. remove from that location
# This code originally in data_util/guardian_gendatasets
# moved here so that could be shared with cryptics_parsing for identifying clue types
####
def make_stc_map(clue_list: List[BaseClue]) -> DefaultDict[str, List[BaseClue]]:
    soln_to_clue_map: defaultdict[str, List[GuardianClue]] = defaultdict(list)
    for c in tqdm(clue_list):
        soln_to_clue_map[c.soln].append(c)
    return soln_to_clue_map


# Use to find duplicate clues
def normalize(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ''.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def filter_clues(soln_to_clue_map: defaultdict[str, List[BaseClue]]) \
                 -> Tuple[Dict[str, List[BaseClue]], List[BaseClue]]:
    # Remove anything that is exactly the same up to small diffs
    # removes 1610 normalized clues
    soln_to_clue_map_clean: Dict[str, List[BaseClue]] = defaultdict(list)
    all_clues_clean = []
    count_removed = 0
    output_count = 0
    for k, v in tqdm(soln_to_clue_map.items()):
        # each v is a list, so we compare the clues that have the same soln
        set_of_clues: Set[str] = set()
        clean_list = []

        for gc in v:
            norm_clue = normalize(gc.clue)
            if norm_clue in set_of_clues:
                count_removed += 1
                continue
            else:
                set_of_clues.add(norm_clue)
                clean_list.append(gc)
        if len(clean_list) > 0:
            soln_to_clue_map_clean[k] = clean_list
            output_count += len(clean_list)
            all_clues_clean.extend(clean_list)

    print(f'removed {count_removed} exact dupes')
    print(output_count)

    return soln_to_clue_map_clean, all_clues_clean
