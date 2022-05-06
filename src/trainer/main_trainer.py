import logging
import os

from src.datasets.dataset_select import DatasetSelection
from src.models.model_select import ModelSelection

from .lightning_trainer import get_trainer

logger = logging.getLogger(__name__)


def lightning_training(args, dataset, model):
    logger.info(f'[Trianer] setup training environment')
    
    # setup pytorch lightning trainer
    trainer = get_trainer(args)

    # setup fold 
    dataset.setup_fold(args.fold)

    # fit model with given dataset
    logger.info(f'[Trainer] start fitting model')
    trainer.fit(model, datamodule=dataset)
    # retrieve results
    fit_results = trainer.logged_metrics

    # load the checkpoint of the best model considering the best valiation accuracy
    fit_ckp_cb = trainer.checkpoint_callback
    best_model_path = fit_ckp_cb.best_model_path
    logger.info(f'[Trainer]')

    # print early stop info
    earlystop_cb = trainer.early_stopping_callback
    if earlystop_cb.stopped_epoch != 0:
        logger.info(f'[Trainer] earlystopped at epoch {earlystop_cb.stopped_epoch}')

    # test model on the best checkpoint
    if os.path.isfile(best_model_path):
        logger.info(f'[Trainer] start testing model')
        test_results = trainer.test(ckpt_path=best_model_path, datamodule=dataset)[0]

    # delete checkpoint if delete checkpoint is set
    if os.path.isfile(best_model_path) and args.delete_checkpoint:
        logger.info(f'[Trainer] checkpoint deleted {best_model_path}')
        os.remove(best_model_path)

    results = {**fit_results, **test_results}
    logger.info('[Trainer] test_results {}'.format(results))

def setup_parameters(args):
    if args.model == 'RoBERTa_CNN':
        vars(args).update({'tokenizer': 'roberta-base'})
    else:
        vars(args).update({'tokenizer':'bert-base-cased'})

    vars(args).update({
            'feature_dim': 7,
            'nclass': 4,
        })

    if args.fold > args.nfold:
        raise ValueError(f'fold should be between 1 and nfold={args.nfold}, but got fold={args.fold}')

def train(args):
    setup_parameters(args)
    logger.info(f'[ScriptInfo] {args.dataset} {args.model}')
    logger.info(f'[Arguments] \n{args}')

    # setup model and dataset
    model = ModelSelection.getModel(args.model, args=args)
    dataset = DatasetSelection.getDataset(args.dataset, args=args)

    # training with pytorch lightning
    lightning_training(args,dataset,model)
    
    