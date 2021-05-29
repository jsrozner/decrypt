"""
Used for cryptic baselines evaluation.
"""
from lemminflect import getAllInflections
from nltk.corpus import wordnet as wn
from typing import Set
from pprint import pprint as pp

def all_inflect(w, word_len):
    out = set()
    for k, v in getAllInflections(w).items():
        if word_len is not None:
            out.update(filter(lambda x: len(x) == word_len, v))
        else:
            out.update(v)
    return out

def all_lemmas_for_word(w: str,
                        min_word_len,
                        remove_word=False) -> Set[str]:
    """
    Args:
        w: word
        min_word_len: min_word_len of a synonym to be included in the set

    Returns: Set of all synonyms (lemmas) for the word, in all of its senses (synsets)
    """
    # For every synset, for every synonym (lemma), if the lemma is > min_word_len
    synsets = wn.synsets(w)
    ret = set([lem for ss in synsets for lem in ss.lemma_names() if len(lem) >= min_word_len])
    if remove_word:
        ret = [x for x in ret if w[:-1] not in x]

    return ret



# todo: we should only do matching parts of speech, and we should only consider central nouns in the
# noun phrase, e.g. via the parse tree
def in_defn(clue: str, answer: str, min_word_len=3):
    """
    Check whether any of the lemma names for the clue synsets are in any of the
    lemma sets of the definition of the answer

    The answer often entails the clue word. For example, for a clue of 'bird' with answer
    'hoopoes' we have defn of hoopoes: any of several crested Old World birds
            with a slender downward-curved bill
    which contains bird.
    """
    clue_lemmas = all_lemmas_for_word(clue, min_word_len)

    overlap_ct = 0   # number of times we find a clue lemma in the definition set
    for answer_ss in wn.synsets(answer):   # for synset in answer
        defn_words = answer_ss.definition().split(" ")
        defn_lemma_set = set()     # set of all lemmas for the definition words
        for w in defn_words:
            if len(w) >= min_word_len:
                defn_lemma_set |= all_lemmas_for_word(w, min_word_len)

        # check whether any clue lemma is in any definition lemma (i.e. is a synonym
        # of any of the definition words) for this synset
        for cw in clue_lemmas:
            if cw in defn_lemma_set:
                #print(f"found {cw} in defn_word_set")
                overlap_ct += 1

    return overlap_ct > 0

# def in_closure_set(w1: str, w2: str, max_depth:int=3, print=False,
#                     closure_fn=lambda x: x.hypernyms()) -> bool:
#     """
#     Check whether w1 is in closure set (default hypernym) of w2
#     by computing they hypernym set of w2 (for each of its possible synsets)
#     and then checking whether the w1 (or any of its synonyms) matches one of those hypernyms
#     Args:
#         clue:
#         answer:
#         max_depth: how deep to go in hypernym tree
#
#     Returns:
#
#     """
#     lookup_syns = set(map(lambda x: x.name(),
#                          wn.synsets(w1)))
#     if print: pp(lookup_syns)
#
#     all_closures = set([ss.closure(closure_fn, depth=max_depth) for ss in wn.synsets(w2)])
#     all_closure_names
#         closure_hyp_set = set(map(lambda hyp: hyp.name(),
#                                  ss.closure(closure_fn, depth=max_depth)))
#         closure_hyp_set |= {ss.name()}  # also include "same-level" synonyms
#         if print: pp(closure_hyp_set)
#         if not lookup_syns.isdisjoint(closure_hyp_set):
#             return True
#
#     return False


def in_closure_set(w1: str, w2: str, max_depth:int=3, print=False,
                    closure_fn=lambda x: x.hypernyms()) -> bool:
    """
    Check whether w1 is in closure set (default hypernym) of w2
    by computing they hypernym set of w2 (for each of its possible synsets)
    and then checking whether the w1 (or any of its synonyms) matches one of those hypernyms
    Args:
        clue:
        answer:
        max_depth: how deep to go in hypernym tree

    Returns:

    """
    # Get all synonyms of the lookup word
    lookup_syns = set(wn.synsets(w1))
    if print: pp(lookup_syns)

    # compute the closure sets of the closure word
    synsets = wn.synsets(w2)
    all_closures = set(c for ss in synsets for c in ss.closure(closure_fn, depth=max_depth))
    all_closures |= set(ss for ss in synsets)      # add depth 0 names
    if print: pp(all_closures)

    if not lookup_syns.isdisjoint(all_closures):
        return True

    return False