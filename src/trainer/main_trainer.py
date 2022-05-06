import logging
import os

from src.datasets.dataset_select import DatasetSelection
from src.models.model_select import ModelSelection

from .lightning_trainer import get_trainer

logger = logging.getLogger(__name__)


def lightning_training(args, dataset, model):
    logger.info(f'setup training environment')
    trainer = get_trainer(args)

    dataset.setup_fold(args.fold)
    # model training
    trainer.fit(model, datamodule=dataset)
    fit_results = trainer.logged_metrics
    fit_ckp_cb = trainer.checkpoint_callback
    earlystop_cb = trainer.early_stopping_callback

    # model testing
    best_model_path = fit_ckp_cb.best_model_path
    if os.path.isfile(best_model_path):
        test_results = trainer.test(ckpt_path=best_model_path, datamodule=dataset)[0]

    if os.path.isfile(best_model_path):
        os.remove(best_model_path)

    results = {**fit_results, **test_results}
    logger.info('test_results {}'.format(results))

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
    logger.info(f'[StartTraining] {args.dataset} {args.model}')
    logger.info(args)
    model = ModelSelection.getModel(args.model, args=args)
    dataset = DatasetSelection.getDataset(args.dataset, args=args)
    # setup lighting model 
    lightning_training(args,dataset,model)
    
    