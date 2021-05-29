import os
from pathlib import Path

#####
# File locations
#####

# parse root directory
k_dir = Path(os.path.abspath(__file__)).parent

# class FileLocs:
    # k_wiki_abbrev = k_dir + "cryptic-code/" + "data/generated/wiki_abbrev.txt"
    # k_anag_dict = path.curdir + "/data/anagrams"         # system will auto-append db

    # class Indicators:
    #     k_homophone_indics = k_dir + "cryptic-code/data/indicators/homophone.txt"
    #     k_hidden_indics = k_dir + "cryptic-code/data/indicators/hidden.txt"


# data dirs
class DataDirs:
    # class Cryptonite:
    #     main = k_dir + "cryptic-supp/cryptonite-main/data/"
    #     official_split = main + "cryptonite-official-split/"        # cryponite-[test|train|val].jsonl
    #     naive_split = main + "cryptonite-naive-split/"

    class Deits:
        k_deits_main = k_dir / 'deits'
        k_deits_clues = k_deits_main / 'clues'
        k_deits_outputs = k_deits_main / 'outputs'

    class Guardian:
        json_folder = k_dir / "data/puzzles/"


    # class Export:
    #     main = k_dir + "cryptic-data/data_export (sym)/"
    #
    # class ClusterRuns:
    #     main = k_dir + "cryptic-data/cluster_save/"
    class OriginalData:
        k_xd_cw = k_dir / "data/original/xd/clues.tsv"
        k_US_dic = k_dir / "data/original/us/US.dic"
        # k_chen_wiki = k_dir / "data/original/chen_wiki.txt"
        #
        # k_names = k_dir / "cryptic-code/data/original/names/"

    class Generated:
        xd_cw_clean_json = k_dir / "data/generated/xd_clean.json"

        # generated from TWL06
        # https://github.com/fogleman/TWL06 (no license)
        twl_tex_dict = k_dir / "data/generated/twl_dict.txt"

    class DataExport:
        _base = k_dir / "data/clue_json/"
        xd_cw_json = _base / "ACW_data"