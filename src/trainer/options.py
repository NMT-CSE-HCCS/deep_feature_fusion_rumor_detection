import argparse
import logging

from src.datasets.dataset_select import DatasetSelection
from src.models.model_select import ModelSelection

logger = logging.getLogger(__name__)


def add_model_hyperparameter(parser):
    parser.add_argument('--model',type=str, default=ModelSelection.default(),
                        choices=ModelSelection.choices(), help='Choose the model in ["RoBERTa_CNN","BERT_CNN"]')
    
def add_dataset_parameter(parser):
    parser.add_argument('--dataset', type=str, default=DatasetSelection.default(), 
                        choices=DatasetSelection.choices(), help='Choose the dataset in ["twitter15","twitter16"]')
    parser.add_argument('--max_token', type=int, default=128, 
                        help='max token when tokenizing tweet text')
    parser.add_argument('--max_tree', type=int, default=200, 
                        help='the max tree node in propagation tree')
    parser.add_argument('--nfold', type=int, default=5,
                        help='total number of splits for k fold cross-validation')
    parser.add_argument('--fold', type=int, default=1, 
                        help='fold number between 1 and nfold.')

def add_trainer_parameter(parser):
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping. Set it the same as epochs to avoid early stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for training all modules except pre-trained module')
    parser.add_argument('--lr_trans', type=float, default=2e-5, 
                        help='learning rate for finetuning on pre-trained transformer-based module')
    parser.add_argument('--weight_decay', type=float, default=5e-2,
                        help='Weight decay (L2 loss on parameters) for AdamW.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size used in dataloader')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of workers used in torch dataloader.')

def add_system_parameter(parser):
    parser.add_argument('--expname', type=str, default='rumor_detection', 
                        help='It affects logging subfolder, etc.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disables CUDA training. When not using no_cuda, the code is running on cuda when cuda exists.')
    parser.add_argument('--debug', action='store_true',
                        help='Debug flag for logging debug level')
    parser.add_argument('--determ', action='store_true',
                        help='Deterministic flag for achieving the most deterministic training')
    parser.add_argument('--seed', type=int, default=42, help='Specify random seed when determ is used')
    parser.add_argument('--delete_checkpoint', action='store_true', 
                        help='Delete the trained best model checkpoint when used, otherwise the checkpoint will be kept.')


def build_arg():
    parser = argparse.ArgumentParser()
    add_system_parameter(parser)
    add_dataset_parameter(parser)
    add_model_hyperparameter(parser)
    add_trainer_parameter(parser)
    return parser

def setup_arg():
    parser = build_arg()
    return parser.parse_args()

