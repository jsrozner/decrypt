import os
from pathlib import Path

#####
# File locations
#####

# parse root directory
k_dir = Path(os.path.abspath(__file__)).parent


# data dirs
class DataDirs:

    class Deits:
        k_deits_main = k_dir / 'deits'
        k_deits_clues = k_deits_main / 'clues'
        k_deits_outputs = k_deits_main / 'outputs'

    class Guardian:
        json_folder = k_dir / "data/puzzles/"

    class OriginalData:
        k_xd_cw = k_dir / "data/original/xd/clues.tsv"
        k_US_dic = k_dir / "data/original/us/US.dic"
        k_chen_wiki = k_dir / "data/original/chenwiki/chen_wiki.txt"

        # copied from deits directory
        k_deits_anagram_list = k_dir / "data/original/deits_anag_indic/ana_"

        k_names = k_dir / "data/original/names/"

        # cryptonite
        _cryptonite_original_data_dir = k_dir / "data/original/cryptonite"
        k_cryptonite_offical = _cryptonite_original_data_dir / "cryptonite-official-split"
        k_cryptonite_naive = _cryptonite_original_data_dir / "cryptonite-naive-split"


    # our generated files that are not model inputs
    class Generated:
        xd_cw_clean_json = k_dir / "data/generated/xd_clean.json"

        # generated from TWL06
        # https://github.com/fogleman/TWL06 (no license)
        twl_tex_dict = k_dir / "data/generated/twl_dict.txt"

        # anagrams
        anagram_db = k_dir / "data/generated/anag_db"

    class DataExport:
        _base = k_dir / "data/clue_json/"

        # guardian
        _guardian_base_dir = _base / "guardian"
        guardian_naive_random_split = _guardian_base_dir / "naive_random"
        guardian_naive_disjoint_split = _guardian_base_dir / "naive_disjoint"
        guardian_word_init_disjoint_split = _guardian_base_dir / "word_init_disjoint"

        ####
        # curricular
        _curricular = _base / "curricular"

        # ACW
        _ACW_sub_dir = "ACW_data"
        xd_cw_json = _curricular / _ACW_sub_dir

        # anagramming
        _anag_sub_dir = "anagram"
        anag_dir = _curricular / _anag_sub_dir  # anagrams (train.json) and anag_indics
        anag_indics = anag_dir / "anag_indics.json"
        ####

        # descrambling
        _descramble_dir = _base / "descramble"
        descramble_random = _descramble_dir / "random_split"
        descramble_word_init_disjoint = _descramble_dir / "word_initial"

        # 6.4 wordplay
        wordplay_dir = _base / "wordplay"

        # 6.5 cryptonite
        _crypto_dir = _base / "cryptonite"
        crypto_naive = _crypto_dir / "naive"
        crypto_naive_disjoint = _crypto_dir / "official_theirs"
        crypto_word_init_disjoint = _crypto_dir / "word_init_disjoint"




