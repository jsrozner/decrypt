from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List

@dataclass_json
@dataclass
class Substitution:
    new_clue_str: str
    substituted_word: str

@dataclass_json
@dataclass
class ClueWithSubstitutions:
    orig_input: str
    word_to_be_swapped: str     # anagram substrate
    target: str

    substitutions: List[Substitution]
