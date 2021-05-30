import logging
import sys
from typing import *

import wandb
from overrides import overrides
from transformers import PreTrainedTokenizerFast

import args_cryptic as args
from common_seq import (
    util,
    util_metrics
)
from common_seq.util_dataloader_batch import default_collate_fn_json
from train_abc import (
    RunHelper,
    pre_setup,
    setup_wandb_for_run, T5Trainer
)
from multitask_config import multi_config



class ClueTrainer(T5Trainer):
    def __init__(self, config, rh, **kwargs):
        super().__init__(config, rh, **kwargs)

        if self.config.special:
            assert self.config.special in ['no_lens', 'no_len_multi']

    @overrides
    def init_val_fcns(self):
        assert self.config.do_sample
        def compute_metrics_sampled_curry(*fcn_args):
            return util_metrics.compute_metrics_sampled(*fcn_args)

        self.val_fcn_list: List[util_metrics.MetricFcn] = [
            compute_metrics_sampled_curry
        ]

    @overrides
    def setup_dataloaders(self):
        if not self.config.special:
            super().setup_dataloaders()
            return

        if self.config.special == 'no_lens':
            self.setup_dataloaders_no_len()
        elif self.config.special == 'no_len_multi':
            assert self.config.multitask
            self.setup_dataloaders_no_len_multi()
        else:
            raise NotImplemented

    def setup_dataloaders_no_len(self):
        log.info('training with no length specification')
        # remove the length specification
        def pre_tokenize(batch_list: List[Dict]):
            src_text, tgt_text, idxs = [], [], []
            for e in batch_list:
                input = e['input']
                splits = input.split(' ')
                assert splits[-1][0] == '('
                input = ' '.join(splits[:-1])

                tgt = e['target']
                idx = e['idx']

                src_text.append(input)
                tgt_text.append(tgt)
                idxs.append(idx)

            return src_text, tgt_text, idxs

        def coll_fn(tokenizer: PreTrainedTokenizerFast, batch_list: List[Dict]) -> Dict:
            return default_collate_fn_json(tokenizer, batch_list, pre_tokenize_fn=pre_tokenize)

        self.setup_dataloaders_no_multi(batched_collate_fcns=[coll_fn, coll_fn])

    def setup_dataloaders_no_len_multi(self):
        # remove the length specification on multitask dataset
        def pre_tokenize(batch_list: List[Dict]):
            src_text, tgt_text, idxs = [], [], []
            for e in batch_list:
                input = e['input']
                splits = input.split(' ')
                assert splits[-1][0] == '('
                input = ' '.join(splits[:-1])

                tgt = e['target']
                idx = e['idx']

                src_text.append(input)
                tgt_text.append(tgt)
                idxs.append(idx)

            return src_text, tgt_text, idxs

        def coll_fn_multi(tokenizer: PreTrainedTokenizerFast, batch_list: List[Dict]) -> Dict:
            return default_collate_fn_json(tokenizer, batch_list, pre_tokenize_fn=pre_tokenize)

        self.setup_dataloaders_multi(batched_collate_fcns=None,
                                     multitask_collate_fn=coll_fn_multi)

    @overrides
    def post_run(self):
        # get the checkpoint
        best_val = self.rh.ckpt_saver.best_vals['dev/num_match_top_sampled']
        ckpt_path, sym_link = best_val[2], best_val[3]
        log.info(f'loading from\n\t{ckpt_path}\n\t{sym_link}')

        preds_path = best_val[2] + "preds.json"
        log.info(f'for final validation:\n'
                 f'\t{preds_path}')


def add_extra_args(parser):
    parser.add_argument('--special',
                        default=None,
                        help='Enables special flags, for example'
                             'no_lens')
    parser.add_argument('--hacky',
                        action='store_true',
                        help='Enables train resume on multitask. See train_abc.py for details of what needs to be done.')

if __name__ == '__main__':
    parsed_args = args.get_args(add_extra_args)
    pre_setup(parsed_args)
    global_record_dir = setup_wandb_for_run(parsed_args)
    log = logging.getLogger()
    util.config_logger(log, global_record_dir)
    log.info(" ".join(sys.argv[:]))
    util.set_seed(wandb.config.seed)

    ########
    ### run specific config
    ########
    metrics_to_track: List[Tuple[str, bool]] = [
        ('dev/num_match_top_sampled', True),
        # ('dev/num_match_top_5_sampled', True),
        # ('dev/NLL', False),
    ]
    # todo: this isn't working
    wandb.run.summary["dev/num_match_top_sampled"] = "best_top_sampled"
    wandb.run.summary["dev/num_match_in_sample"] = "best_in_sample"

    # multitask
    aux_config = dict()
    if wandb.config.multitask:
        aux_config['multitask_config'] = multi_config[wandb.config.multitask]
        # 'acw' must match the name of one of the tasks in multitask_config
        metrics_to_track.extend([('multisave', True),
                                ('multi/acw/num_match_in_sample', True)])   # forces save at the end of multitask training

    local_rh = RunHelper(global_record_dir, metrics_to_track, wandb.config)
    ################
    # final setup (consistent for all
    local_trainer = ClueTrainer(wandb.config, local_rh, aux_config=aux_config)
    wandb.watch(local_trainer.model, log="all")

    local_trainer.run()
