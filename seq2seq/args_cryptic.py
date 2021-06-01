"""
Shortcut args for cryptics
"""
import argparse

def add_args(parser: argparse.ArgumentParser):
    """
    Always need to specify
        --name (run name)
        --project (for wandb)
        --data_dir (see config.py for where outputs are stored)
        --wandb_dir

    for curricular training, need to specify
        --multitask

    for eval (--default_eval)
        --ckpt_path - will be loaded
        --test - whether to run on the test set
    """

    ## shortcuts
    parser.add_argument('--dev_run',
                        action='store_true',
                        help='sets wandb env mode to dry run, reduces number of train/val to 1000')

    parser.add_argument('--default_train',
                        type=str,
                        default=None,
                        help='Default configurations:'
                             'Set to either base or cryptonite')
    parser.add_argument('--default_val',
                        type=str,
                        default=None,
                        help='Various default configurations:'
                             'Set to either base or cryptonite')
    ####
    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.'
                             'Will be used by wandb to identify run')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='Directory containing the train, val, test files')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default=None,
                        help='If given then will load from given model path. '
                             'You should either set no_train (val only), '
                             'or explicitly specify whether resuming train with --resume_train or '
                             '--no_resume_train')

    parser.add_argument('--resume_train',
                        action='store_true',
                        dest='resume_train',
                        help='whether to resume train with optimizer and scheduler state.'
                             'Need to pass a ckpt_path')
    parser.add_argument('--no_resume_train',
                        action='store_false',
                        dest='resume_train')
    parser.set_defaults(resume_train=None)       # will be false for eval

    parser.add_argument('--no_train',
                        action='store_true',
                        help='Set to no_train when we want to do eval only.')

    parser.add_argument('--test',
                        action='store_true',
                        help='Eval will be done on test rather than val set')

    ## new
    parser.add_argument('--ada_constant',
                        action='store_true',
                        help='Whether to use constant LR with adafactor. Used for t5-large training')
    parser.add_argument('--multi_gpu',
                        type=int,
                        default=None,
                        help='Whether to use dataparallel with multiple gpus.'
                             'Also need to set k_data_parallel=True in train_abc')

    # defaults that are auto set but that will vary / depend on --default_eval and
    # these will have defaults set (varies for train / eval)
    parser.add_argument('--project',
                        type=str,
                        required=True)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=15,     # set to 1 by default_eval
                        help='Number of epochs for which to train. Negative means forever.'
                             ' i.e. all training data this many times through')
    parser.add_argument('--generation_beams',
                        type=int,
                        default=5,      # default eval 100
                        help='Number of beams (and return sequences) to use in generation')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')

    # defaults for both
    parser.add_argument('--model_name',
                        type=str,
                        default='t5-base',
                        help='which t5 model to load, e.g. t5-small, t5-base')
    parser.add_argument('--wandb_dir',
                        type=str,
                        required=True,
                        help='Directory in which to add folder wandb for storing all files')
    parser.set_defaults(do_sample=True)
    parser.add_argument('--no-save', dest='do_save', action='store_false')
    parser.set_defaults(do_save=True)       # will be false for eval
    parser.set_defaults(batched_dl=True)
    parser.set_defaults(fast_tokenizer=True)
    parser.set_defaults(ada=True)
    parser.set_defaults(add_special_tokens=False)
    parser.set_defaults(use_json=True)

    # other multitask arguments are in multitaskconfig
    parser.add_argument('--multitask',
                        type=str,
                        help='Whether to do multitask training. To do multitask, provide a '
                             'update cfg/multi_cfg with a new object. Specify that config by str')

    # don't modify these
    parser.add_argument('--num_train',
                        type=int,
                        default=-1,
                        help='Number of train examples to consider. Will reduce dataset. Neg ignores')
    parser.add_argument('--num_val',
                        type=int,
                        default=-1,
                        help='Number of val examples to consider. Will reduce dataset. Neg ignores')
    parser.add_argument('--multitask_num',
                        type=int,
                        default=-1,
                        help='number of multitask examples to use in the dataloader; -1 is all.'
                             'Generally used only by dev_run to speed up')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--val_freq',
                        type=int,
                        default=None,
                        help='How often to do validation in thousands of steps (must be fewer than # examples)')
    parser.add_argument('--early_stopping',
                        type=str,
                        default=None,
                        help='Metric to track for early stopping, assumes lower is better')
    parser.add_argument('--grad_accum_steps',
                        type=int,
                        default=1,
                        help='Number of batches to accumulate')
    # default to don't use
    parser.add_argument('--comment',
                        type=str,
                        default="",
                        help='A comment to store with the run')


def get_args(extra_args_fn=None):
    """Get arguments needed in train.py."""

    # parse the args
    parser = argparse.ArgumentParser('Train T5 for decrypting cryptic crosswords')
    add_args(parser)
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()

    ###
    # misc validation
    if args.dev_run or args.no_train:
        args.do_save = False
        args.val_freq = None
        # note that example counts for --dev_run will be changed (reduced) in pre_setup function
        # see train_abc

    ###

    ###
    # default training / val
    assert args.default_train is None or args.default_val is None, \
        f'Cannot do both default_train and default_val simultaneously'
    if args.default_train is not None:
        if args.default_train == 'base':
            # args.project = "cryptics_train"
            # ada_constant is False (i.e. we use relative step)
            args.generation_beams = 5
            args.batch_size = 256       # alternatively can do 128 and accum_steps=2
            # grad_accum_steps = 1
            # args.num_epochs = 15
            # default model is t5-base

        elif args.default_train == 'cryptonite':
            # args.project = "cryptonite"
            args.ada_constant = True
            args.generation_beams = 5
            args.batch_size = 64
            args.grad_accum_steps = 12
            # args.num_epochs = 15            # for the naive split, can train to 20 epochs
            args.model_name = 't5-large'
            # args.val_freq = 100       # set to 100 for the disjoint set
        else:
            raise NotImplemented

    if args.default_val is not None:
        assert args.ckpt_path is not None, \
            f'To run default eval, need to pass a model checkpoint with --ckpt_path'
        args.no_train = True
        args.val_freq = None
        args.resume_train = False
        args.num_epochs = 1     # just needed for args validation
        args.do_save = False    # no checkpointing

        if args.default_val == 'base':
            # args.project = "cryptics_val"
            args.generation_beams = 100
            args.batch_size = 16            # can change this depending on your GPU; doesn't affect results
            # default model is t5-base
        elif args.default_val == 'cryptonite':
            # default for cryptonite eval
            # args.project = "cryptonite"
            args.generation_beams = 5       # cryptonite originally used only 5 beams, copy their implementation
            args.batch_size = 64            # can change depending on GPU; doesn't affect val resulst
            args.model_name = 't5-large'
        else:
            raise NotImplemented(f'Invalid option {args.default_val} for --default_val ')
    ######

    return args
