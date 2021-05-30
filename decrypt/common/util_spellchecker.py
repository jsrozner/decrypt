from __future__ import annotations

import logging
import os
from typing import *

import enchant
from tqdm import tqdm

import config

logging.getLogger(__name__)

def get_shelve_dbhandler_open_flag(output_filename: str, update_flag: str = "") -> Optional[str]:
    flag = ""
    if update_flag == "new":    # generate new, don't overwrite
        if os.path.isfile(output_filename + ".db"):
            logging.warning(f"File already exists. Use other update_type flag")
            return None
        flag = "n"
    elif update_flag == "update":  # update:
        if not os.path.isfile(output_filename + ".db"):
            logging.warning(f"Attempting to update a database that does not exist. Failed")
            return None
        logging.info(f"Updating database at {output_filename}")
        flag = "w"
    elif update_flag == "overwrite":  # overwrite
        logging.info(f"Overwriting database at {output_filename}")
        flag = "n"
    else:
        logging.warning(f"Invalid flag. Failed")
        return None

    return flag

def line_parser_US_dic(input_line: bytes, log_errors=False) -> Optional[List[str]]:
    try:
        x = input_line.decode("utf-8")
        x = x.strip()
        return [x]
    except UnicodeDecodeError:
        if log_errors:
            print(f"unicode decode fail: {repr(input_line)}")
        return None

def line_parser_chenwiki(input_line: str,
                         spell_chkr: SpellChecker,
                         spell_check_single_words: bool = True) -> Optional[List[str]]:
    """
    For use with data/chenwiki.txt
    """
    try:
        input_word = input_line.split(";")[0].lower()
        split_word_list = spell_chkr.split_mixed_word(input_word)
        if split_word_list:
            return split_word_list

        # if we do spellchecking, then verify that it is a valid word before returning
        if spell_check_single_words and not spell_chkr.check_word(input_word, special_handle_short_words=True):
            return None

        # otherwise we can always return the input word by itself
        return [input_word]

    except IndexError:
        return None

# todo: enchant is no longer maintained and double checking is inefficient
class SpellChecker:
    def __init__(self,
                 dict_files: List[Tuple[str, bool]] = None,
                 init_enchant_dict=True,
                 init_twl_dict=True,
                 log_init_errors=False):
        """

        Args:
            dict_files: List of tuples of <filename, is_bytes>
        """
        print("Initialized a spellchecker")
        self.dict = set()
        self.enchant_dict = None
        self.twl_short_word_dict = set()
        if init_enchant_dict:
            self.enchant_dict = enchant.Dict("en_US")
        if init_twl_dict:
            self.__add_twl_contents_to_dict(config.DataDirs.Generated.twl_tex_dict)

        if dict_files is None:
            dict_files = [(config.DataDirs.OriginalData.k_US_dic, True)]
        for df in dict_files:
            self.__add_file_contents_to_dict(df, log_init_errors)
        logging.info("Done setting up spellchecker")

    def __del__(self):
        print("DEL called for spellchecker")

    def __add_twl_contents_to_dict(self, file: str):
        logging.info(f'Reading file into dict: {file}')
        print(f"This will fail if you have not downloaded or generated twl_dict.txt")
        with open(file, 'r') as f:
            for input_line in tqdm(f):
                word = input_line.strip()
                if word != "":
                    if len(word) < 3:
                        self.twl_short_word_dict.add(word.lower())
                    else:
                        self.dict.add(word.lower())

        logging.info(f'Done reading file: {file}')

    def __add_file_contents_to_dict(self, file: Tuple[str, bool], log_errors):
        logging.info(f'Reading file into dict: {file[0]}')
        if file[1]:         # bytes
            with open(file[0], 'rb') as f:
                for input_line in tqdm(f):
                    word_list = line_parser_US_dic(input_line, log_errors=log_errors)
                    if word_list is not None and len(word_list) > 0 and word_list[0] != "":
                        self.dict.add(word_list[0].lower())
        else:               # not bytes
            with open(file[0], 'r') as f:
                for input_line in tqdm(f):
                    word = input_line.strip()
                    if word != "":
                        self.dict.add(word.lower())

        logging.info(f'Done reading file: {file[0]}')

    def check_word(self, w: str,
                   lower_case: bool = True,
                   special_handle_short_words: bool = False,
                   check_twl_short_dict: bool = True,
                   check_enchant_dict: bool = True,
                   print_info: bool = False,
                   use_base_dict=True) -> bool:
        if lower_case:
            w = w.lower()

        one_letter_words = ["a", "i"]
        two_letter_words = ["ad", "am", "an", "as", "at",
                            "do", "go", "he", "hi", "if", "in",
                            "is", "it", "me", "my", "no", "of", "on", "or",
                            "so", "to", "up", "us"]
        in_dict = w in self.dict
        in_short_words = w in one_letter_words or w in two_letter_words
        in_twl_short = w in self.twl_short_word_dict
        in_enchant_lower = self.enchant_dict is not None and self.enchant_dict.check(w)
        in_enchant_upper = self.enchant_dict is not None and self.enchant_dict.check(w.capitalize())

        if print_info:
            print(f'dict: {in_dict}\t twl_short: {in_twl_short}\t short_word: {in_short_words}\n'
                  f'enchant_lower: {in_enchant_lower}\t enchant_upper: {in_enchant_upper}')

        # Some heuristics to fix problems with short words pre-empting the backtracking alg
        if special_handle_short_words and len(w) <= 3:
            if len(w) < 3:
                return in_short_words
            else:   # len == 3
                return in_dict and (in_enchant_lower or in_enchant_upper)


        # Otherwise, successively check dicts
        if use_base_dict and in_dict:
            return True
        elif check_twl_short_dict and in_twl_short:
            return True
        elif check_enchant_dict and (in_enchant_lower or in_enchant_upper):
            return True
        else:
            return False

    def split_mixed_word(self, input_word: str) -> Optional[List[str]]:
        """
        Recursive backtracking, (greedy) algorithm for determining the set of words
        in a word without spaces.

        # todo: some three letter words will (still?) cause a problem

        Returns: List of words (str) that compose the input string
            None: if no valid split found
        """
        # Don't pass around the spell_chkr
        wlen = len(input_word)
        for end_idx in range(wlen, 0, -1):
            w = input_word[0:end_idx]
            if self.check_word(w, special_handle_short_words=True):
                if end_idx == wlen:     # base case, we are done
                    return [w]
                # otherwise, need to compute possibly terminating words
                next = input_word[end_idx:]
                next_result = self.split_mixed_word(next)
                if next_result is None:
                    continue
                else:
                    ret = [w]
                    ret.extend(next_result)
                    return ret
        return None
