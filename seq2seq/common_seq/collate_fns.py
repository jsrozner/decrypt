import logging
import random

from common_seq.types import *
from common_seq.util_dataloader_batch import default_collate_fn_json

log = logging.getLogger(__name__)


# collate function factory
def collate_fn_from_pretokenize(pretokenize_fn: Callable) -> collate_fn_type:
    def coll_fn(tokenizer: PreTrainedTokenizerFast, batch_list: List[Dict]) -> Dict:
        return default_collate_fn_json(tokenizer, batch_list, pre_tokenize_fn=pretokenize_fn)
    return coll_fn


def _add_label(orig_input: str, label: str):
    return f'{label}: {orig_input}'


def make_pretokenize_prepend_label(label: str) -> pretokenize_fn:
    def pre_tokenize_prepend_label(batch_list: List[Dict]) -> Tuple[List, ...]:
        src_text, tgt_text, idxs = [], [], []
        for e in batch_list:
            input = e['input']
            tgt = e['target']
            idx = e['idx']

            # input = f'{label}: {input}'
            input = _add_label(input, label)

            src_text.append(input)
            tgt_text.append(tgt)
            idxs.append(idx)

        return src_text, tgt_text, idxs

    return pre_tokenize_prepend_label



# note that casing will be off slightly on this - it will always be lower case
def make_pretokenize_descramble(label: Optional[str], word_only: bool = False):
    rng = random.Random(42)

    def randomize_letters(s: str) -> str:
        x = list(s)
        rng.shuffle(x)
        return "".join(x)

    def pre_tokenize_descramble(batch_list: List[Dict]):
        src_text, tgt_text, idxs = [], [], []
        for e in batch_list:
            input = e['input']
            tgt = e['target']
            idx = e['idx']

            tgt_scrambled = randomize_letters(tgt)

            # parse length string (and move to end)
            splits = input.split(' ')
            assert splits[-1][0] == '('
            input_no_len = ' '.join(splits[:-1])
            input_no_len_lower = input_no_len[0].lower() + input_no_len[1:]
            len_str = splits[-1]

            if word_only:
                input = f'{tgt_scrambled} {len_str}'
            else:
                if rng.randint(0, 1) == 0:
                    input = f'{tgt_scrambled} {input_no_len_lower} {len_str}'
                else:
                    input = f'{input_no_len_lower} {tgt_scrambled} {len_str}'

            # finalize
            if label is not None:
                input = _add_label(input, label)
            src_text.append(input)
            tgt_text.append(tgt)
            idxs.append(idx)

        return src_text, tgt_text, idxs
    return pre_tokenize_descramble

## for anagramming
# note that casing will be off slightly on this - it will always be lower case
import json
def make_pretokenize_anagram(label: Optional[str],
                             anag_indic_file: str):

    logging.info(f'Opening {anag_indic_file} for anag indicators')
    with open(anag_indic_file, 'r') as f:
        anag_indics = json.load(f)

    rng = random.Random(42)


    def pre_tokenize_descramble(batch_list: List[Dict]):
        src_text, tgt_text, idxs = [], [], []
        anag_indic_sampled = random.choices(anag_indics, k=len(batch_list))         # with replacement

        for e, anag_indic in zip(batch_list, anag_indic_sampled):
            anag_list = e['anag_list']      # list of words mapping to common set of letters
            idx = e['idx']

            choices = random.sample(anag_list, 2)       # no replacement
            lhs, rhs = tuple(choices)

            # add lengths (target is the rhs, so lengths uses the rhs)
            lengths = list(map(lambda x: str(len(x)),
                               rhs.split(' ')))
            len_str = f'({",".join(lengths)})'

            # input and target
            if rng.randint(0, 1) == 0:
                input = f'{lhs} {anag_indic} {len_str}'
            else:
                input = f'{anag_indic} {lhs} {len_str}'
            tgt = rhs

            # finalize
            if label is not None:
                input = _add_label(input, label)
            src_text.append(input)
            tgt_text.append(tgt)
            idxs.append(idx)

        return src_text, tgt_text, idxs
    return pre_tokenize_descramble

