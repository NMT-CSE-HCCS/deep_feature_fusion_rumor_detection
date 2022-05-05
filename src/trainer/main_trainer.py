from datetime import time
import os
import sys
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
import pytorch_lightning as pl
from lib.models.bert import BertMNLIFinetuner
from lib.models.roberta import RoBERTaFinetuner
from lib.settings.config import settings
from lib.transfer_learn.param import Param
from lib.utils.twitter_data import TwitterData
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#from lib.utils.status_sqlite_bert import Status
from lib.utils.status_sqlite_classes import Status
from lib.utils.log import setup_custom_logger
import torch

#from pytorch_memlab import MemReporter

jobid = os.environ.get('SLURM_JOB_ID')
logger = None
if not jobid:
    logger = setup_custom_logger('pytorch_lightning','console_paralle',timestamp=True)
else:
    logger = setup_custom_logger('pytorch_lightning',f'slurm_{jobid}_paralle')

pid = os.environ.get('SLURM_TASK_PID')
nodelist = os.environ.get('SLURM_NODELIST')
logger.info(f'[starting] pid {pid} {nodelist}')

def my_except_hook(exctype, value, traceback):
    if torch.cuda.is_available():
        logger.info(torch.cuda.memory_summary())
        pid = os.environ.get('SLURM_TASK_PID')
        nodelist = os.environ.get('SLURM_NODELIST')
        logger.error(f'Exception Happened {traceback} {value} {pid} {nodelist}')
        sys.__excepthook__(exctype, value, traceback)

sys.excepthook = my_except_hook

__all__ = ['TransferFactory']


class TransferFactory():
    def __init__(self):
        exp = int(list(settings.transfer.param.exp)[0])
        self.status = Status()
        self.max_epoch = 200
        
    def set_config(self, p: Param, fold):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pl.seed_everything(1234)
        monitor_key = 'acc'
        if monitor_key == 'loss':
            monitor_mode = 'min'
        elif monitor_key == 'acc':
            monitor_mode = 'max'
        else:
            raise ValueError(f'monitor_key "{monitor_key}" is incorrect')
        
        es_cb = EarlyStopping(
            monitor=f'val_{monitor_key}_epoch',
            patience=20,
            mode=monitor_mode,
        )
        self.es_cb = es_cb
        model_name = p.pretrain_model.split('-')[0]

        ckp_cb = ModelCheckpoint(
            monitor=f'val_{monitor_key}_epoch',
            dirpath=settings.checkpoint,
            filename= model_name + '-exp=' + str(p.exp) + '-d=' + p.dnn + '-sp=' + p.split_type + '-maxl=' + str(p.max_tree_len) + '-fold='+ str(fold) + '-{epoch:02d}-{val_acc_epoch:.3f}',
            save_top_k=1,
            mode=monitor_mode
        )
        self.ckp_cb = ckp_cb

        logger.info('[Device Count] {}'.format(torch.cuda.device_count()))
        gpu_counts = torch.cuda.device_count()
        if gpu_counts > 0:
            logger.info('[Device Name] {}'.format(torch.cuda.get_device_name()))
        
        logger_flag = True
        if logger_flag:
            # comet_logger = CometLogger(
            #     api_key='RHywDLGIc61n40dBpkSqcmqp7',
            #     project_name='fakenews',  # Optional
            #     workspace='lzrpotato',
            #     experiment_name=p.experiment_name + '-fd=' + str(fold),
            #     offline=False,
            # )
            

            tb_logger = TensorBoardLogger(
                save_dir = './logger/',
                name = p.experiment_name,
                
            )
            lr_monitor = LearningRateMonitor(
                logging_interval='epoch',
            )
            if gpu_counts == 0:
                precision = 32
            else:
                precision = 16
            self.trainer = pl.Trainer(gpus=gpu_counts,
                                #auto_select_gpus = True,
                                precision=precision,
                                profiler=True,
                                max_epochs=self.max_epoch,
                                progress_bar_refresh_rate=0,
                                flush_logs_every_n_steps=100,
                                callbacks=[es_cb,ckp_cb,lr_monitor],
                                logger=tb_logger
                                )
        else:
            self.trainer = pl.Trainer(gpus=gpu_counts,
                #auto_select_gpus = True,
                max_epochs=self.max_epoch,
                progress_bar_refresh_rate=0,
                flush_logs_every_n_steps=100,
                callbacks=[es_cb,ckp_cb],
            )

    def run(self, p: Param):
        subclass = False
        if p.nclass == 4:
            subclass = False
        else:
            subclass = True
        
        twdata = TwitterData(settings.data, p.pretrain_model,tree=p.tree,split_type=p.split_type,
            max_tree_length=p.max_tree_len,limit=p.limit,cv=True,n_splits=5,subclass=subclass,kfold_deterministic=True)
        twdata.setup_kfold()
        results_kfold = []
        for fold in range(5):
            if self.check_state(p,fold) == 1:
                continue

            if torch.cuda.is_available():
                logger.info(f'[GPU][B] {torch.cuda.memory_reserved()/1024/1024/1024} GB')
                torch.cuda.empty_cache()
                logger.info(f'[GPU][B] {torch.cuda.memory_reserved()/1024/1024/1024} GB')
            
            twdata.kfold_get_by_fold(fold)

            self.set_config(p,fold)
            model = None

            if p.pretrain_model.split('-')[0] == 'roberta':
                model = RoBERTaFinetuner(ep=p,fold=fold,feature_dim=twdata.feature_dim,nclass=p.nclass,max_epoch=self.max_epoch)
            elif p.pretrain_model.split('-')[0] == 'bert':
                model = BertMNLIFinetuner(ep=p,fold=fold,feature_dim=twdata.feature_dim,nclass=p.nclass,max_epoch=self.max_epoch)
            else:
                raise ValueError(f'Incorrect pretrain_model "{p.pretrain_model}"')

            test_result = None
            if p.classifier.split('_')[0] == 'dense':
                self.trainer.fit(model,train_dataloader=twdata.train_dataloader,val_dataloaders=twdata.val_dataloader)
                #model.load_from_checkpoint(self.ckp_cb.best_model_path)
                #model = RoBERTaFinetuner.load_from_checkpoint(self.ckp_cb.best_model_path)
                #reporter = MemReporter(model)
                #reporter.report()
                test_result = self.trainer.test(test_dataloaders=twdata.test_dataloader)
                
                del self.trainer
                del model
                if torch.cuda.is_available():
                    logger.info(f'[GPU] {torch.cuda.memory_reserved()/1024/1024/1024} GB')
                    torch.cuda.empty_cache()
                    logger.info(f'[GPU] {torch.cuda.memory_reserved()/1024/1024/1024} GB')
                #exit()
            results_kfold.append(test_result[0])
            logger.info(f'[test_result] {test_result[0]}')
            self.save_state(p, test_result[0], fold)
            if os.path.isfile(self.ckp_cb.best_model_path):
                logger.info(f'rm file {self.ckp_cb.best_model_path}')
                os.remove(self.ckp_cb.best_model_path)
            else:
                logger.warning(f'[Warning] best model path {self.ckp_cb.best_model_path} is incorrect')

        #self.save_state_kfold(p, results_kfold)
        #self.save_state(p, results_kfold[0])

    def save_state(self, p: Param, result: dict, fold):
        if p.nclass == 4:
            res = {'acc':result['test_acc_epoch'],'c1':result['fscore'][1],'c2':result['fscore'][0],'c3':result['fscore'][2],'c4':result['fscore'][3]}
        else:
            res = {'acc':result['test_acc_epoch'],'c1':result['fscore'][0],'c2':result['fscore'][1],'c3':0,'c4':0}
        
        par = {'exp':p.exp,'nclass':p.nclass,'dnn':p.dnn,'fold':fold,'splittype':p.split_type,'maxlen':p.max_tree_len,'pretrain':p.pretrain_model,
                'aux':p.auxiliary,'stopepoch':self.es_cb.stopped_epoch,'bestepoch':int(self.es_cb.stopped_epoch-self.es_cb.wait_count)}
        
        par.update(res)
        logger.info(par)
        self.status.save_status(par)

    def check_state(self, p: Param, fold):
        par = {'exp':p.exp,'nclass':p.nclass,'dnn':p.dnn,'fold':fold,'splittype':p.split_type,
                'aux':p.auxiliary,'maxlen':p.max_tree_len,'pretrain':p.pretrain_model,'fold':fold}
        
        return self.status.read_status(par)