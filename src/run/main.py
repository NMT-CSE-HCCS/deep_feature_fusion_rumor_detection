import os
import sys
sys.path.append('.')
import time
import traceback

from src.utils.random_seed import get_timebaesed_random_seed
from src.logger.get_configured_logger import get_logger_seperate_config


def getlogger(args):
    midpath = f'{args.model}_{args.dataset}/'
    fn = f"fold={args.fold}_t={time.strftime('%Y-%m-%d-%H:%M:%S')}_pid={os.getpid()}"
    root_logger = get_logger_seperate_config(args.debug, args.expname, midpath, fn)

    return root_logger

def set_random_seed(args):
    if args.determ:
        args.seed = 42
    else:
        args.seed = get_timebaesed_random_seed()

def main():
    from src.trainer.main_trainer_test import train
    from src.trainer.options import setup_arg
    args = setup_arg()
    root_logger = getlogger(args)
    set_random_seed(args)

    try:
        train(args)
    except Exception:
        root_logger.error(f"Error pid {os.getpid()}: {traceback.format_exc()}")


if __name__ == '__main__':
    main()
    