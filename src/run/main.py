#!/fs1/epscor/home/zluo_epscor/.conda/envs/py38c11/bin/python3.8 

#SBATCH --job-name py   ## name that will show up in the queue
#SBATCH --output ./slurm/slurm_out/%x-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --error ./slurm/slurm_out/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=epscor  ## the partitions to run in (comma seperated)
##SBATCH --exclude=discovery-g[1,12,13]
##SBATCH --nodelist=discovery-g[12,13]
#SBATCH --time=3-24:00:00  ## time for analysis (day-hour:min:sec)



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
        # print(f"Error pid {os.getpid()}: {traceback.format_exc()}")


if __name__ == '__main__':
    main()
    