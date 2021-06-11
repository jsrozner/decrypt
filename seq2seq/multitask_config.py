import common_seq.collate_fns as cfns
from common_seq import util_metrics
from common_seq.util_multiloader import MultitaskConfig, TaskConfig
# import sys
# sys.path.append('../../decrypt_root')       # decrypt_root to enable config import
# import decrypt_root.config as config

# relative import issues with config
k_curr_dir = "../data/clue_json/curricular"
# k_curr_dir = config.DataDirs.DataExport._curricular
# subdirectories within k_curr_dir with the content
# k_task_dir = config.DataDirs.DataExport._ACW_sub_dir
k_task_dir = "ACW_data"
# k_anag_task_dir = config.DataDirs.DataExport.anag_dir   # has train.json
k_anag_task_dir = "anagram"
k_anag_indic_file = f'{k_curr_dir}/{k_anag_task_dir}/anag_indics.json'

k_default_args = dict(
    multitask_dir=k_curr_dir,
    reset=True,
    val_split_pct=0.99
)

# american crossword (with label)
task_ACW = TaskConfig(
    dir=k_task_dir,        # 2.4m
    name="acw",
    val_fcn_list=[util_metrics.compute_metrics_sampled_primary],
    # adds a label, like 'phrase: <input>'
    collate_fn=cfns.collate_fn_from_pretokenize(cfns.make_pretokenize_prepend_label('phrase'))
)
task_ACW_descramble = TaskConfig(
    dir=k_task_dir,        # 2.4m
    name="acw_descramble",
    val_fcn_list=[util_metrics.compute_metrics_sampled_primary],
    collate_fn = cfns.collate_fn_from_pretokenize(cfns.make_pretokenize_descramble(label='descramble'))
)
task_ACW_descramble_word = TaskConfig(
    dir=k_task_dir,        # 2.4m
    name="acw_descramble_word",
    val_fcn_list=[util_metrics.compute_metrics_sampled],
    # add a label and descramble (will lowercase the first letter of the clue)
    collate_fn = cfns.collate_fn_from_pretokenize(
        cfns.make_pretokenize_descramble(label='descramble word', word_only=True))
)
task_anagram = TaskConfig(
    dir= k_anag_task_dir,
    name="anag_with_indic",
    val_fcn_list=[util_metrics.compute_metrics_sampled],
    # add a label and descramble (will lowercase the first letter of the clue)
    collate_fn = cfns.collate_fn_from_pretokenize(
        cfns.make_pretokenize_anagram(label='anagram',
                                      anag_indic_file=k_anag_indic_file))
)


multi_config = dict(
    # ACW only
    ACW = MultitaskConfig(
        freq_list=[20, 6],      # same as 20, 3,3 w.r.t. total pretraining examples
        num_warmup=4,
        tasks=[task_ACW],
        **k_default_args
    ),

    # ACW-descramble only
    ACW_descramble = MultitaskConfig(
        freq_list=[20, 6],      # same as 20, 3,3 w.r.t. total pretraining examples
        num_warmup=4,
        tasks=[task_ACW_descramble],
        **k_default_args
    ),

    # ACW + ACW-descramble
    # top performing
    ACW__ACW_descramble = MultitaskConfig(
        freq_list=[20, 3, 3],
        num_warmup=2,       # use 2 -> roughly translates to 4 total epochs
        tasks=[task_ACW, task_ACW_descramble],
        **k_default_args
    ),

    # ACW + ACW-descramble-word
    # crosswords + descramble bare
    ACW__ACW_descramble_word = MultitaskConfig(
        freq_list=[20, 3, 3],
        num_warmup=2,       # use 2 -> roughly translates to 4 total epochs
        tasks=[task_ACW, task_ACW_descramble_word],
        **k_default_args
    ),

    # ACW + anagram
    ACW__anagram = MultitaskConfig(
        freq_list=[20, 3, 3],
        num_warmup=2,
        tasks=[task_ACW, task_anagram],
        **k_default_args
    ),

    # ACW + ACW-descramble + anagram
    # has 7:6 ratio of pretraining batches (i.e. more)
    ACW__ACW_descramble__anagram = MultitaskConfig(
        freq_list=[20, 3, 3, 1],
        num_warmup=2,
        tasks=[task_ACW, task_ACW_descramble, task_anagram],
        **k_default_args
    ),

    final_top_result_scaled_up = MultitaskConfig(
        freq_list=[20, 3, 3],
        num_warmup=4,           # scale up for the top performing
        tasks=[task_ACW, task_ACW_descramble],
        **k_default_args
    ),

    # Cryptonite - best multitask approach (ACW + ACW-descramble)
    # has an extra epoch of warmup
    cfg_crypto_acw_acwdesc = MultitaskConfig(
        freq_list=[20, 3, 3],
        num_warmup=3,           # one more than above
        tasks=[task_ACW, task_ACW_descramble],
        **k_default_args
    ),
)
