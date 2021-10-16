import glob
import json
import logging
import random
import re
from collections import Counter
from collections import defaultdict
from pprint import pp
from typing import *
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from decrypt.common.puzzle_clue import (
    BaseClue, GuardianClue, CleanGuardianClue,
    filter_clues, make_stc_map
)
from .util import _gen_filename, str_hash as safe_hash

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# for verification
k_expected_puz_count = 5518
k_expected_clue_ct = 142380
k_5111_clue_text = 'Great issue, relatively speaking'

# handles typing for the class (constructor)
TGuardClue = TypeVar("TGuardClue", bound=GuardianClue)

# todo: some clues seem not to be stripped? (i.e. trailing space)
def clean_and_add_clues_from_guardian_json_puzzle_to_dict(path: str,
                                                          puzzle_dict: Dict[str, List[GuardianClue]],
                                                          ctr: Counter,
                                                          clue_cls: Type[TGuardClue],
                                                          skip_if_in_dict: bool = True):
    """
    Parses puzzle from path (json)

    Will automatically filter. All solutions will be lowercased.
    Args:
        path: a json file with a puzzle
        puzzle_dict: the accumulation dict (in case accumulating results over multiple runs)
        ctr: Counter object
        skip_if_in_dict: skip parsing a puzzle if already in dict; this is unnecessary because the
            computation is so cheap

    """
    # Set up: accumulation list and file handler
    clue_list = []
    with open(path, "r") as f:
        puz_json = json.load(f)

    # Make sure we haven't seen this puzzle before (based on ID)
    puz_id = puz_json['id']
    if puzzle_dict.get(puz_id) is not None and skip_if_in_dict:
        ctr['stat: already_parsed'] += 1
        return
    ctr['stat: parsed_puzzle'] += 1

    # Get the puzzle information
    cw_type = puz_json['crosswordType']
    number = puz_json['number']
    try:
        creator = puz_json['creator']['name']
    except KeyError:
        creator = None

    # Iterate through the puzzle entries (clues);
    # - extract relevant info from json
    # - filter
    for entry in puz_json['entries']:
        if len(entry.get('group')) > 1:
            ctr['invalid: clue group'] += 1
            continue

        clue_id: str = entry['id']  # of form [#]-[across,down]
        base_clue: str = entry['clue']  # invalid base clue will be caught by regexp below

        base_direction: str = entry['direction']
        base_soln: str = entry['solution']
        base_pos: Dict[str, str] = entry['position']  # {'x': 6, 'y': 0}

        assert base_direction in ['across', 'down']
        across_or_down = base_direction

        # there are 4 clues with nonalphabetic solutions that we admit
        solution = base_soln.lower()

        x, y = int(base_pos['x']), int(base_pos['y'])
        pos = (x, y)

        # standardize hyphens / dashes (very small number of clues)
        for c in ["—", "–"]:
            base_clue = base_clue.replace(c, "-")
        if '\xad' in base_clue:  # this is a soft break that can be removed
            base_clue = base_clue.replace('\xad', "")

        # Filtering based on clue portion. Make sure the whole clue (clue (lengths)) matches regexp
        # regexp: (allow an optional space) (parse clue and length portions)
        clue_text_pattern = re.compile("^(.*)(?: )?\((.*)\)$")
        try:
            base_clue = base_clue.replace('\n', "").replace('\r', '').replace('\t', '')  # trailing misc whitespace
            if "  " in base_clue:
                base_clue = re.sub(' +', ' ', base_clue)  # double spaces
            re_match = clue_text_pattern.match(base_clue)
            clue_text = re_match.group(1).strip()
            numbers = re_match.group(2)
            numbers_split = re.split('\W+', numbers)  # split on all punctuation
            numbers_split = list(filter(lambda x: len(x) > 0, numbers_split))
        except Exception:
            # sometimes a clue relies on another clue (e.g. "See #"); ignore them
            ctr['invalid: regexp'] += 1
            continue

        # Now filter based on the clue portion
        # make sure starts with an alphabetic character
        if len(clue_text) < 2:
            ctr['invalid: zero-len clue text after regexp'] += 1
            continue

        # filter clues that have invalid characters
        allowed_chars = [",", "'", "?", "!", ".", ":", "-", " ", ";",
                         "\"", '”', '“',
                         "(", ")", "=", "&", "/"]
        allowed_chars += [
            '+',  # 1 clue
            '@',  # 3 clues
            '#',  # 4 clues
            '_',  # fillin, 107 clues
            '£',  # 1 clue
            '*',  # allowed in clue, but not at start (clue grouping)
            '…',  # allowed in clue, but not at start (continuation clue)
            '¿',  # unclear, 3 clues
            '[', ']',  # utterance or similar (23)
            '’', '‘'  # omission; # todo: should be 1) merged, 2) specially handled in the knn baseline vectorizer
        ]
        # not_allowed
        # '<',    # invalid html
        # '†'     # marks a puzzle annotation (i.e. provides additional info)

        # make sure clue contains only alpha or allowed chars
        if not clue_text.isalpha():
            # First check for numbers
            # many clues with numbers in them have reference to another clue
            # todo: we could try to substitute in
            has_number = False
            for c in clue_text:
                if c.isnumeric():
                    has_number = True
                    break
            if has_number:
                ctr['invalid: number in clue (commonly references another clue)'] += 1
                continue

            # Then check for other bad characters
            bad_clue = False
            for c in clue_text:
                if not c.isalpha() and c not in allowed_chars:
                    # ctr['invalid: disallowed char in clue; sub_stat ' + c] += 1
                    bad_clue = True
                    break  # todo: could also get the set of bad characters
            if bad_clue:
                ctr['invalid: unrecognized char in clue (e.g. html)'] += 1
                continue

        # extra filtering for the start of the clue
        allowed_start_chars = ["\"", "'", '[', '#', '_', '“', '’', '-', '(',
                               '?'  # this is the ? (Hamlet) => tobeornottobe clue
                               ]
        # not allowed
        # * -> indicates a clue grouping within puzzle
        if not clue_text[0].isalpha() and not clue_text[0] in allowed_start_chars:
            ctr['invalid: invalid start char (most are continuation clues)'] += 1
            # ctr['invalid: non-alpha start; sub_stat: ' + clue_text[0]] += 1

            if clue_text[0] not in ['*', '…', '.']:  # these are not surprising
                print(f'nonalpha start ({clue_text[0]}): {clue_text} => {solution}')
            continue

        # Now filter based on the lengths portion
        clue_lenghts_punct = set()  # we store the punctuation that appears. we can filter later
        for c in numbers:  # iterate through the characters in numbers string: i.e. #,#-#,...
            if c.isnumeric() or c == ' ':  # we will check alphabetic below. for now we admit alpha and num
                continue
            clue_lenghts_punct.add(c)
            ctr[f'length punct: {c}'] += 1  # record any non-space seperator

        # store the lengths of each word in the answer
        clue_lengths: List[int] = []
        for x in numbers_split:  # numbers_split is a list of strings
            if x.isnumeric():
                clue_lengths.append(int(x))
            else:
                raise ValueError('invalid character in numbers_split')

        # this can happen when answers span multiple solution boxes
        total_len = sum(clue_lengths)
        if len(solution) != total_len:
            ctr['invalid: soln length does not match specified lens (multi box soln)'] += 1
            continue

        unique_clue_id = f'{cw_type}_{number}_{clue_id}'
        new_clue = clue_cls(unique_clue_id=unique_clue_id,
                                type=cw_type,
                                number=number,
                                id=puz_id,
                                creator=creator,
                                orig_lengths=numbers,
                                lengths_punctuation=clue_lenghts_punct,
                                across_or_down=across_or_down,
                                pos=pos,
                                clue=clue_text,
                                lengths=clue_lengths,
                                soln=solution,
                                )
        clue_list.append(new_clue)
        ctr[len(clue_lengths)] += 1  # record number of clues with num_words in solns

    ctr['stat: total_clues'] += len(clue_list)  # one clue_list per puzzle; this is shared across calls to this function
    puzzle_dict[puz_id] = clue_list  # augment the database
    return clue_list


####
# Methods to load in the json
####
def all_json_files_to_json_list(json_files_dir, subsite, puzzle_dict: Dict,
                                clue_cls: Type[TGuardClue],
                                skip_if_in_dict: bool = True,
                                verify=True) -> List[GuardianClue]:
    clue_list: List[GuardianClue] = []
    ctr = Counter()

    file_glob_path = _gen_filename(json_files_dir, subsite=subsite, ext=".json", return_glob=True)
    log.info(f'Using file glob at {file_glob_path}')
    file_glob = glob.glob(file_glob_path)
    log.info(f'Glob has size {len(file_glob)}')
    if len(file_glob) != k_expected_puz_count:
        log.warning('File glob is a different size from expected. Your dataset will be different from the one'
                    ' in the paper')
        if verify:
            raise NotImplemented('Your glob is different from expected. Manually remove this line '
                                 'or set verify=False to continue')
    else:
        log.info(f'Glob size matches the expected one from Decrypting paper')

    for f in tqdm(sorted(file_glob)):
        new_clues = clean_and_add_clues_from_guardian_json_puzzle_to_dict(f, puzzle_dict, ctr,
                                                                          skip_if_in_dict=skip_if_in_dict,
                                                                          clue_cls=clue_cls)
        clue_list.extend(new_clues)
    pp(sorted(ctr.items(), key=lambda x: str(x)))
    print(f'Total clues: len(puzz_list)')
    return clue_list


# runs over actual json files; don't use if using the distributed clue set
def orig_get_clean_clues(json_output_dir,
                    do_filter_dupes=True,
                    verify=True,
                    strip_identifying_info=False,
                    ) -> Tuple[Dict[str, List[BaseClue]], List[BaseClue]]:
    log.info(f'loading from {json_output_dir}')
    parsed_puzzles: Dict[str, List[GuardianClue]] = defaultdict(None)  # map from puz_id => List[GuardianClue]

    # load full glob
    if strip_identifying_info:
        clue_cls = CleanGuardianClue
    else:
        clue_cls = GuardianClue
    all_clue_list = all_json_files_to_json_list(json_output_dir,
                                                subsite="cryptic",
                                                puzzle_dict=parsed_puzzles,
                                                skip_if_in_dict=True,
                                                verify=verify,
                                                clue_cls=clue_cls)


    soln_to_clue_map = make_stc_map(all_clue_list)

    # Remove anything that is exactly the same up to small diffs
    # removes 1610 normalized clues
    if do_filter_dupes:
        soln_to_clue_map, all_clue_list = filter_clues(soln_to_clue_map)

    return soln_to_clue_map, all_clue_list


def get_clean_clues(json_file_or_json_dir,
                    load_from_json_files: bool = False,
                    verify=True,
                    ) -> Tuple[Dict[str, List[BaseClue]], List[BaseClue]]:
    if load_from_json_files:
        soln_to_clue_map, all_clue_list = orig_get_clean_clues(
            json_file_or_json_dir)
    else:
        with open(json_file_or_json_dir, 'r') as f:
            all_clue_list = json.load(f)
        all_clue_list = list(map(CleanGuardianClue.from_json, all_clue_list))
        soln_to_clue_map = make_stc_map(all_clue_list)

    # add indices and a note about dataset
    for idx, c in enumerate(all_clue_list):
        c.idx = idx
        # if not strip_identifying_info:
        #     c.dataset = json_output_dir

    # print the distribution
    ctr = Counter()
    for c in all_clue_list:
        ctr[len(c.lengths)] += 1
    log.info(ctr)

    # Verify same length
    assert sum(map(len, soln_to_clue_map.values())) == len(all_clue_list)

    if verify:
        assert len(all_clue_list) == 142380, f'Your clues do not match the ones in Decrypting paper'
        log.info(f'Clue list length matches Decrypting paper expected length')

    return soln_to_clue_map, all_clue_list





def check_splits(all_clues, input_tuple):
    train, val, test = input_tuple

    log.info(f'Got splits of lenghts {list(map(len, (train, val, test)))}')
    log.info(f'First three clues of train set:\n\t{train[:3]}')

    assert sum(map(len, input_tuple)) == len(all_clues)


# each of these returns a tuple of
# - soln to clue map (string to List of clues mapping to that soln): Dict[str, List[GuardianClue]
# - list of all clues (List[GuardianClue])
# - Tuple of three lists (the train, val, test splits), each is List[GuardianClue]
SplitReturn = Tuple[Dict[str, List[BaseClue]], List[BaseClue], Tuple[List[BaseClue], ...]]


def load_guardian_splits(json_dir, seed=42, verify=True, load_from_files=False) -> SplitReturn:
    soln_to_clue_map, all_clues = get_clean_clues(json_dir, verify=verify, load_from_json_files=load_from_files)
    train, test = train_test_split(all_clues, test_size=0.2, random_state=seed)
    train, val = train_test_split(train, test_size=0.25, random_state=seed)

    check_splits(all_clues, (train, val, test))

    if verify:
        assert train[5111].clue == k_5111_clue_text, f'Your splits do not match the ones in Decrypting paper'
        log.info('Verifying splits match Decrypting paper: Spot test clue 5111 has correct text')

    return soln_to_clue_map, all_clues, (train, val, test)


def load_guardian_splits_disjoint(json_dir, seed=42, verify=True, load_from_files=False) -> SplitReturn:
    soln_to_clue_map, all_clues = get_clean_clues(json_dir, verify=verify, load_from_json_files=load_from_files)

    splits = [0.2, 0.25]

    # split based on keys
    train_keys, test_keys = train_test_split(list(soln_to_clue_map.keys()), test_size=splits[0], random_state=seed)
    train_keys, val_keys = train_test_split(train_keys, test_size=splits[1], random_state=seed)

    rng = random.Random(seed)

    def get_all_values(key_set) -> List[BaseClue]:
        ret_list: List[BaseClue] = []
        for k in key_set:
            ret_list.extend(soln_to_clue_map[k])

        rng.shuffle(ret_list)
        return ret_list

    all_keys_tuple = [train_keys, val_keys, test_keys]
    all_clues_tuple = tuple(map(get_all_values, all_keys_tuple))

    check_splits(all_clues, all_clues_tuple)

    return soln_to_clue_map, all_clues, all_clues_tuple


def make_disjoint_split(all_clues: List[BaseClue],
                        seed=42) -> Tuple[List[BaseClue], ...]:
    soln_to_clue_map = make_stc_map(all_clues)
    train, val, test = [], [], []
    for k, v in soln_to_clue_map.items():
        h = safe_hash(k[:2]) % 5  # normal hash function is not deterministic across python runs
        if h < 3:
            train.extend(v)
        elif h < 4:
            val.extend(v)
        else:
            test.extend(v)

    out_tuple = train, val, test
    rng = random.Random(seed)
    for l in out_tuple:
        rng.shuffle(l)
    check_splits(all_clues, out_tuple)

    return out_tuple


def load_guardian_splits_disjoint_hash(json_dir: str, seed=42, verify=True, load_from_files=False) -> SplitReturn:
    """
    Produce a disjoint split based on hashing the first two letters
    :return: SplitReturn (see this file)
    """
    soln_to_clue_map, all_clues = get_clean_clues(json_dir, verify=verify, load_from_json_files=load_from_files)
    out_tuple = make_disjoint_split(all_clues, seed)

    return soln_to_clue_map, all_clues, out_tuple
