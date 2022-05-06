import logging
from typing import Any, List

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MeanMetric
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

def pretty_print_confmx_pandas(confmx):
    pd.set_option('display.max_columns', None)
    df_confmx = pd.DataFrame(confmx.numpy())
    df_confmx['sum'] = df_confmx.sum(axis=1)
    str_confmx = str(df_confmx)
    pd.reset_option('display.max_columns')
    return str_confmx


"""
BaseModule is implemented in Pytorch-Lightning
See more details in Pytorch-Lightning
https://github.com/PyTorchLightning/pytorch-lightning 
"""
class BaseModule(pl.LightningModule):
    def __init__(self, args, nclass, model):
        super(BaseModule,self).__init__()
        self.args = args
        self.model = model
        self.nclass = nclass
        self.all_metrics = nn.ModuleDict()
        for phase in ['train','val', 'test']:
            self.all_metrics[phase+'_metrics'] = nn.ModuleDict({
                    "acc": Accuracy(),
                    "accmacro": Accuracy(num_classes=nclass,average='macro'),
                    "loss": MeanMetric(),
                    "f1macro": F1Score(num_classes=nclass,average='macro'),
                    "f1micro": F1Score(num_classes=nclass,average='micro'),
                    "f1none": F1Score(num_classes=nclass,average='none'),
                    "confmx": ConfusionMatrix(nclass)
                })

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def loss(self, pred, label):
        loss = F.cross_entropy(pred, label)
        return loss
    
    def metrics(self, phase, pred, label, loss):
        phase_metrics = self.all_metrics[phase+'_metrics']
        for mk, metric in phase_metrics.items():
            if mk == 'loss':
                result = metric(loss)
            elif mk == 'acc':
                result = metric(pred,label)
                self.log(f'{phase}_acc_step', result, sync_dist=True, prog_bar=True, batch_size=self.args.batch_size)
            else:
                result = metric(pred,label)

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase+'_metrics']
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute()
            metric.reset()

        self.log_epoch_end(phase, metrics)
        if phase == 'test':
            self.stored_test_confmx = metrics['confmx']

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx.cpu().numpy().tolist()
        return []

    def log_epoch_end(self, phase, metrics):
        self.log(f'{phase}_loss', metrics['loss'])
        self.log(f'{phase}_acc_epoch', metrics['acc'])
        self.log(f'{phase}_f1macro_epoch', metrics['f1macro'])
        self.log(f'{phase}_accmacro', metrics['accmacro'])
        self.log(f'{phase}_f1micro', metrics['f1micro'])
        self.log(f'{phase}_f1macro', metrics['f1macro'])
        self.log(f'{phase}_acc', metrics['acc'])
        # self.log(f'{phase}_confmx', metrics['confmx'])
        
        logger.info(f'[{phase}_acc_epoch] {metrics["acc"]} at {self.current_epoch}')
        logger.info(f'[{phase}_accmacro] {metrics["accmacro"]}')
        logger.info(f'[{phase}_loss] {metrics["loss"].item()}')
        logger.info(f'[{phase}_f1_score] {metrics["f1micro"]}')
        logger.info(f'[{phase}_f1_score_macro] {metrics["f1macro"]}')
        logger.info(f'[{phase}_confmx] \n{pretty_print_confmx_pandas(metrics["confmx"].detach().cpu().type(torch.long))}')

    def configure_optimizers(self):
        optimizer = AdamW([
            {'params': self.model.transformer.parameters(),'lr': self.args.lr_trans,'weight_decay':self.args.weight_decay},
            {'params': self.model.cnn.parameters(),'lr': self.args.lr,'weight_decay':self.args.weight_decay},
            {'params': self.model.classifier.parameters(),'lr':self.args.lr,'weight_decay':self.args.weight_decay},
        ], eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer
                ,num_warmup_steps=self.args.epochs//10, num_training_steps=self.args.epochs)
        return {
            "optimizer":optimizer,
            "lr_scheduler":scheduler
        }

    def configure_callbacks(self):
        monitor = 'val_acc_epoch'
        earlystop = EarlyStopping(
            monitor=monitor,
            patience=self.args.patience,
            mode='max'
        )
        exp_name = f'{self.args.model}_{self.args.dataset}_fold={self.args.fold}_nfold={self.args.nfold}'
        ckp_cb = ModelCheckpoint(
            dirpath=f'model_checkpoint/{exp_name}/',
            filename= 'model-{epoch:02d}-{val_acc_epoch:.3f}',
            monitor=monitor,
            save_top_k=1,
            mode='max'
            )
        
        return [earlystop, ckp_cb]

    def get_predict(self, y):
        a, y_hat = torch.max(y, dim=1)
        return y_hat

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        x = batch[:-1]
        y = batch[-1]

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        
        # predict
        y_hat = self.get_predict(y_hat)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True, batch_size=self.args.batch_size)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)

        self.metrics(phase, y_hat, y, loss)

        return loss

    def training_step(self, batch, batch_nb):
        phase = 'train'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def training_epoch_end(self, outputs) -> None:
        phase = 'train'
        self.metrics_end(phase)

    def validation_step(self, batch, batch_nb):
        phase = 'val'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'val'
        self.metrics_end(phase)

    def test_step(self, batch, batch_nb):
        phase = 'test'
        # fwd
        x = batch[:-1]
        y = batch[-1]
        
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        # acc
        y_hat = self.get_predict(y_hat)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True, batch_size=self.args.batch_size)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        self.metrics(phase, y_hat, y, loss)
        
        return 

    def test_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'test'
        self.metrics_end(phase)
    