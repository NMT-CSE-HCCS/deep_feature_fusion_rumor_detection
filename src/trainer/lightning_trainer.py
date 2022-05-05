import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.utils.cuda_status import get_num_gpus
from pytorch_lightning.callbacks import TQDMProgressBar
import time

logger = logging.getLogger(__name__)

def setup_logger(exp_name,version=None):
    pl_logger = TensorBoardLogger(
        save_dir='tensorboard_logs',
        name=exp_name,
        version=version,
        )
    return pl_logger

def get_trainer(args, version=None, precision=32, fast_dev_run=False):
    if args.no_cuda:
        args.gpus = 0
    else:
        args.gpus = get_num_gpus()
    pb_cb = TQDMProgressBar(refresh_rate=0.2)
    
    if args.profiler:
        profiler='pytorch'
        trainer = pl.Trainer(
            gpus=args.gpus,
            fast_dev_run=fast_dev_run,
            precision=precision,
            max_epochs=args.epochs,
            logger=setup_logger(f'{args.model}_{args.dataset}_fold={args.fold}_{time.strftime("%Y-%m-%d-%H:%M:%S")}', version),
            callbacks=[pb_cb],
            profiler=profiler
        )
    else:
        trainer = pl.Trainer(
            gpus=args.gpus,
            fast_dev_run=fast_dev_run,
            precision=precision,
            max_epochs=args.epochs,
            logger=setup_logger(f'{args.model}_{args.dataset}_fold={args.fold}_{time.strftime("%Y-%m-%d-%H:%M:%S")}', version),
            callbacks=[pb_cb],
        )
    return trainer