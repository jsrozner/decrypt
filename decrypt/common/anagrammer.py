"""
Adapted from origcrypt/anagrammer

An Anagrammer has a dictionary mapping (sorted letters) => AnagramSet

AnagramSets are used for reading in the various dictionaries to generate lists of potential anagrams.
An AnagramSet corresponds to a set of sorted letters. It has two dictionaries:
For each of these, a sort of the unsorted letters gives the sorted letters
    - map (unsorted letters) => [One word anagram]
    - map (unsorted letters) => [multi-word-anagram]
In both cases we represent the anagrammable as a list of words. For a single word anagram the list has len == 1
"""
import logging
import random
import shelve
import string
from collections import Counter
from collections import defaultdict
from os import path
from typing import *

from tqdm import tqdm

import config
from .util_spellchecker import (
    line_parser_US_dic,
    line_parser_chenwiki,
    SpellChecker,
    get_shelve_dbhandler_open_flag
)

logging.getLogger(__name__)

k_default_chenwiki_input_file_name = str(config.DataDirs.OriginalData.k_chen_wiki)
k_default_base_input_file_name = str(config.DataDirs.OriginalData.k_US_dic)
k_default_output_file_name = str(config.DataDirs.Generated.anagram_db)

class AnagramSet:
    """
    Assumes that a given ordering of letters has only one valid parse into words

    """
    def __init__(self, new_item: List[str], new_item_ltrs: str):
        # For one-word-anagrams, each list will have only one word
        # But this is done so that both dicts look the same
        self.one_word_anagrams: Dict[str, List[str]] = defaultdict()
        self.multi_word_anagrams: Dict[str, List[str]] = defaultdict()
        self._num_anagrams = 0
        self._num_one_word_anagrams = 0

        self.add_to_anag_set(new_item, new_item_ltrs)

    def add_to_anag_set(self, item: List[str], item_ltrs: str,
                        log_errors = True) -> bool:
        if len(item) == 1:
            existing = self.one_word_anagrams.get(item_ltrs)
            if existing is not None:
                if log_errors:
                    logging.error(f"Double inserting {existing}, {item}")
                return False
            self.one_word_anagrams[item_ltrs] = item
            self._num_one_word_anagrams += 1
        else:
            existing = self.multi_word_anagrams.get(item_ltrs)
            if existing is not None:
                if log_errors:
                    logging.error(f"Double inserting {existing}, {item}")
                return False
            self.multi_word_anagrams[item_ltrs] = item

        self._num_anagrams += 1
        return True

    # todo: is this necessary? need to make sure we don't modify the lookup
    def get_lists(self) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Returns: Tuple: (one word anagrams, multi-word-anagrams)
            where each is a List of Lists

        """
        # list and tuple are equivalent
        return list(self.one_word_anagrams.values()), list(self.multi_word_anagrams.values())



class Anagrammer():
    """ Anagram database

    Attributes:
        db: opened anagram database. See genanagrams for details of the structure.
        db is a map from str => AnagramSet

    """
    def __init__(self, anagram_database):
        # logging.info(f"Initializing Singleton Anagrammer from {anagram_database}")
        logging.info(f"Initializing (non-singleton) Anagrammer from {anagram_database}")
        self._translation_table = str.maketrans('','',string.punctuation)
        self.db = self.__init_db(anagram_database)

        self._possible_anagrams: Optional[List[AnagramSet]] = None
        logging.info(f"DONE: Initialized Anagrammer from {anagram_database}")

    def __init_db(self, anagram_database):
        if not path.exists(anagram_database + ".db"):
            logging.exception(f'Given anagram database file {anagram_database + ".db"} does not exist')
            raise Exception(f'Given anagram database file {anagram_database} does not exist')
        try:
            db = shelve.open(anagram_database, flag='r')
            logging.debug("Opened anagram database successfully")
            return db
        except Exception as e:
            logging.exception(f'While trying to open {anagram_database}, exception:')
            print(f"path is: {path.curdir}")

            raise e

    def __look_up(self, char_string: str) -> Optional[AnagramSet]:
        """ Perform a lookup on a set of characters. Internal method.

        :param str char_string: letters to use in lookup
        :return: Valid anagrams
        :rtype: list[str]
        """

        chars_no_punct = char_string.translate(self._translation_table)
        lookup = "".join(sorted(chars_no_punct))  # Sort for hashing, essentially
        if lookup in self.db:
            lookup_result: AnagramSet = self.db[lookup]
            return lookup_result
        else:
            return None

    def get_anagrams(self, letters: str,
                     remove_letters=False,
                     include_multi_word_anagrams=False) -> List[List[str]]:
        """
        """
        logging.debug("Looking up in anagram db: " + letters)
        result_anag_set = self.__look_up(letters)

        if result_anag_set is None:
            return []

        one_word_anagrams, multi_word_anagrams = result_anag_set.get_lists()
        # todo: should assert something in case we have one word

        # don't return the letters themselves
        if remove_letters and [letters] in one_word_anagrams:
            one_word_anagrams.remove([letters])

        results = one_word_anagrams
        if include_multi_word_anagrams:
            results.extend(multi_word_anagrams)
        return results

    def get_anagrams_flat(self, letters: str, **kwargs) -> List[str]:
        """
        Return flat list of anagram outputs.

        Signature same as get_anagrams
        """
        list_of_lists = self.get_anagrams(letters, **kwargs)
        return ["".join(x) for x in list_of_lists]

    def is_word(self, word):
        """ Check if word is present in our anagram dictionary

        :param str word:
        :return: True if present in anagram dictionary
        :rtype: bool
        """
        result = self.__look_up(word)
        if result is not None:
            return word in result.one_word_anagrams

    def get_random_anag_sample(self, sample_count: int = 20,
                               return_set = "both") -> List[List[str]]:
        def rand_samp() -> Optional[List[List[str]]]:
            poss_anags : List[AnagramSet] = random.sample(self._possible_anagrams, sample_count)
            for anag_set in poss_anags:
                # multiword only
                # if return_set == "require_mu":
                #     if anag_set._num_anagrams - anag_set._num_one_word_anagrams < 2:
                #         continue
                #     else:
                #         return anag_set.get_lists()[1]
                #
                # both
                if return_set == "both":      # we already know that this is a valid set, since we filtered
                    one_word, multi_word = anag_set.get_lists()
                    one_word.extend(multi_word)
                    return one_word      # we know there are sufficient num anags

                else:   # return_set == "single"
                # otherwise one-word only
                    if anag_set._num_one_word_anagrams < 2:
                        continue
                    else:
                        return anag_set.get_lists()[0]
            return None

        if self._possible_anagrams is None:
            self._populate_possible_anagrams()
        res = None
        while res is None:
            res = rand_samp()
        return res


    def _populate_possible_anagrams(self):
        self._possible_anagrams = []
        for anag_set in tqdm(self.db.values()):
            if anag_set._num_anagrams >=2:
                self._possible_anagrams.append(anag_set)
        logging.info(f"Total anagramable: {len(self._possible_anagrams)}")




def gen_db_with_both_inputs(output_filename=k_default_output_file_name,
                            update_flag: str=""):
    def add_word_to_anagram_db(x: List[str], db: shelve.DbfilenameShelf) -> bool:
        x_ltrs_unsorted = "".join(x)
        x_ltrs_sorted = "".join(sorted(x_ltrs_unsorted))
        # previously x was just a string

        is_new_word = True
        if x_ltrs_sorted in db:
            temp: AnagramSet = db[x_ltrs_sorted]
            if not temp.add_to_anag_set(x, x_ltrs_unsorted, log_errors=False):
                is_new_word = False

            # we re-add back whether or not it's different. this seems to speed up the processing
            db[x_ltrs_sorted] = temp
        else:
            db[x_ltrs_sorted] = AnagramSet(x, x_ltrs_unsorted)  # ow. insert as List

        return is_new_word


    # todo: write into shelf which files have been added as a config variable
    def add_file_to_database(fh,
                             db: shelve.DbfilenameShelf,
                             line_parser_fcn: Union[Callable[[str], List[str]],
                                                    Callable[[bytes], List[str]]]) -> NoReturn:
        """
        Generates a database mapping:
            <ordered string of letters> => [List[one word anagrams]
                                            List[multi word anagrams represented as lists]]
        """
        ctr = Counter()
        for l in tqdm(fh):
            x = line_parser_fcn(l)  # x is a List of strs
            if not x:               # if empty list or returns None
                continue
            ctr[len(x)] += 1

            if not add_word_to_anagram_db(x, db):   # returns true if it's a new word
                ctr["dupes"] += 1

        print(ctr)
        print(f'Done.')


    # set up line parser function
    spell_chkr = SpellChecker()
    def line_parser_fcn_with_spellchkr(input_line: str):
        return line_parser_chenwiki(input_line, spell_chkr)

    # verify the db flags
    dbhandler_flag = get_shelve_dbhandler_open_flag(output_filename, update_flag=update_flag)
    if not dbhandler_flag:
        return

    logging.info(f"Adding to db {output_filename} with updateflag {update_flag}")
    with shelve.open(output_filename, flag=dbhandler_flag) as db:
        with open(k_default_chenwiki_input_file_name, "r") as fh:
            add_file_to_database(fh, db, line_parser_fcn=line_parser_fcn_with_spellchkr)
        with open(k_default_base_input_file_name, "rb") as fh:
            add_file_to_database(fh, db, line_parser_fcn=line_parser_US_dic)


