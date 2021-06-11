import logging
import random
import sys
from typing import List, Dict

import wandb
from overrides import overrides
from transformers import PreTrainedTokenizerFast

import args_cryptic as args
from common_seq import util as util
from common_seq import util_metrics
from common_seq.util_dataloader_batch import default_collate_fn_json
from train_abc import (
    T5Trainer,
    setup_wandb_for_run, pre_setup,
    RunHelper
)

log = logging.getLogger(__name__)

class AnagTrainer(T5Trainer):
    @overrides
    def init_val_fcns(self):
        self.val_fcn_list: List[util_metrics.MetricFcn] = [
            util_metrics.compute_metrics_sampled
        ]

    @overrides
    def setup_dataloaders(self):
        # mods to support new dataloader
        if not self.config.randomize_train_scramble:
            raise NotImplemented

        self.setup_dataloaders_xd_descramble()

    def setup_dataloaders_xd_descramble(self):
        assert self.config.randomize_train_scramble, f'For descramble, the flag --random_train_scramble must be set'
        assert self.config.add_defn is not None, f'For descramble, must specify either --add_defn or --no_defn'

        # otherwise randomize
        # uses function closures
        log.info("Randomizing train set")
        rng = random.Random()
        rng.seed(42)

        def randomize_letters(s: str) -> str:
            x = list(s)
            rng.shuffle(x)
            return "".join(x)

        def pre_tokenize_fn_xd_cw(batch_list: List[Dict]):
            src_text, tgt_text, idxs = [], [], []
            for e in batch_list:
                defn = e['defn']
                tgt = e['target']
                tgt_scrambled = randomize_letters(tgt)

                if not self.config.copy:
                    if self.config.add_defn:
                        src_text.append(f'{tgt_scrambled} | {defn}')
                    else:
                        src_text.append(f'{tgt_scrambled}')
                else:   # copy
                    if self.config.add_defn:
                        src_text.append(f'{tgt} | {defn}')
                    else:
                        src_text.append(f'{tgt}')

                tgt_text.append(tgt)
                idxs.append(-1)         # dummy indices

            return src_text, tgt_text, idxs

        def coll_fn(tokenizer: PreTrainedTokenizerFast, batch_list: List[Dict]) -> Dict:
            return default_collate_fn_json(tokenizer, batch_list, pre_tokenize_fn=pre_tokenize_fn_xd_cw)

        self.setup_dataloaders_no_multi(batched_collate_fcns=[coll_fn, coll_fn])


def add_extra_args(parser):
    parser.add_argument('--randomize_train_scramble',
                        action='store_true',
                        help='Whether to randomize the scrambling in train examples. I.e.'
                             'The collate function will scramble the letters each time, '
                             'so each time a word is shown in train it will (likely) be ordered differently')
    # whether to append definition
    parser.add_argument('--add_defn',
                        action='store_true',
                        dest='add_defn',
                        help='whether to append phrasal definition')
    parser.add_argument('--no_defn',
                        action='store_false',
                        dest='add_defn')
    parser.set_defaults(add_defn=None)       # will be false for eval
    parser.add_argument('--copy',
                        action='store_true',
                        help='Whether to do a copy task (i.e no scrambling)')


if __name__ == '__main__':
    # repeated logic
    parsed_args = args.get_args(add_extra_args)     # potentially pass extra args
    pre_setup(parsed_args)
    global_record_dir = setup_wandb_for_run(parsed_args)
    log = logging.getLogger()
    util.config_logger(log, global_record_dir)
    log.info(" ".join(sys.argv[:]))
    util.set_seed(wandb.config.seed)

    ########
    ### run specific config
    ########
    metrics_to_track = [('dev/NLL', False),
                        ('dev/num_match_top_sampled',True)]

    local_rh = RunHelper(global_record_dir, metrics_to_track, wandb.config)
    ####
    # final setup
    local_trainer = AnagTrainer(wandb.config, local_rh)
    wandb.watch(local_trainer.model, log="all")

    local_trainer.run()
