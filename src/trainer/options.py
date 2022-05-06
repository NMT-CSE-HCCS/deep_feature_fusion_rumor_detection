import argparse
import logging

from src.datasets.dataset_select import DatasetSelection
from src.models.model_select import ModelSelection

logger = logging.getLogger(__name__)


def add_model_hyperparameter(parser):
    parser.add_argument('--model',type=str, default=ModelSelection.default(),
                        choices=ModelSelection.choices())
    
def add_dataset_parameter(parser):
    parser.add_argument('--dataset', type=str, default=DatasetSelection.default(), 
                        choices=DatasetSelection.choices())
    parser.add_argument('--max_token', type=int, default=128, help='max token when tokenizing tweet text')
    parser.add_argument('--max_tree', type=int, default=200, help='the max tree node in propagation tree')
    parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size')
    parser.add_argument('--nfold', type=int, default=5, help='total number of splits for k fold cross-validation')
    parser.add_argument('--fold', type=int, default=1, help='fold in (1,nfold)')

def add_trainer_parameter(parser):
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=20,
                            help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for training all modules except pre-trained module')
    parser.add_argument('--lr_trans', type=float, default=2e-5, 
                        help='learning rate for finetuning on pre-trained transformer-based module')
    parser.add_argument('--weight_decay', type=float, default=5e-2,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--determ', action='store_true',
                        help='Deterministic flag')
    parser.add_argument('--num_workers', type=int, default=8)

def add_system_parameter(parser):
    parser.add_argument('--expname', type=str, default='rumor_detection')
    parser.add_argument('--no_cuda', action='store_true',
                    help='Disables CUDA training.')
    parser.add_argument('--debug', action='store_true',
                            help='Debug flag for logging debug level')
    parser.add_argument('--seed', type=int, default=42)


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

