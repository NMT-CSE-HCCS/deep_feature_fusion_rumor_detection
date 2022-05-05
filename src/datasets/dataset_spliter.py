import logging
import pandas as pd
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class BaseSpliter():
    def __init__(self, unique_str: str, split_strategy: str,
                nrepeat: int, labels, deterministic=True, index_path='./split/',
                ):
        self.unique_str = unique_str
        self.split_strategy = split_strategy
        self.nrepeat = nrepeat
        self.labels = labels
        self.deterministic = deterministic
        self.index_path = index_path

    def get_split_repeat(self, repeat):
        self.repeat = repeat
        train_index, val_index, test_index = self._setup_split_index(repeat)
        logger.debug(f'[spliter summary] (splits {self.splits}) train {len(train_index)} val {len(val_index)} test {len(test_index)}')
        return train_index, val_index, test_index

    def get_split_generator(self):
        for repeat in range(1, self.nrepeat+1):
            train_index, val_index, test_index = self._setup_split_index(repeat)
            yield train_index, val_index, test_index

    def _splits_(self, labels):
        train_index, val_index, test_index = None,None,None
        yield train_index, val_index, test_index

    def _split_and_save_index(self, path):
        train_idx, val_idx, test_idx = self._splits_(self.labels)
        train_idx = [(i, 0) for i in train_idx]
        test_idx = [(i, 2) for i in test_idx]

        df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
        df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
        if val_idx is not None:
            val_idx = [(i,1) for i in val_idx]
            df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
            df_tuple = (df_train, df_val, df_test)
        else:
            df_tuple = (df_train, df_test)
        
        df_split_index = pd.concat(df_tuple, axis=0, ignore_index=True)
        if self.deterministic:
            df_split_index.to_csv(path, index=False)
            logger.debug(f'[spliter] index created and saved!')
        else:
            logger.debug(f'[spliter] index created without saving to file!')
        
        return df_split_index

    def _setup_split_index(self, cur_split):
        folder = f'split_{self.split_strategy}_{self.unique_str}_nrep={self.nrepeat}/'
        path = Path(self.index_path).joinpath(folder)
        path.mkdir(parents=True,exist_ok=True)
        if self.nrepeat <= 1:
            path = path.joinpath(f'index.csv')
        else:
            path = path.joinpath(f'index_{cur_split}.csv')
        if not os.path.isfile(path) or not self.deterministic:
            df_split_index = self._split_and_save_index(path)
        else:
            df_split_index = pd.read_csv(path)
            if len(df_split_index) != len(self.labels):
                logger.debug(f'[spliter] df_split_index and labels length not match {len(df_split_index)} != {len(self.labels)}')
                df_split_index = self._split_and_save_index(path)
            else:
                logger.debug(f'[spliter] Read index from file, length of index {len(df_split_index)}')

        return self._df_to_index(df_split_index)

    def _df_to_index(self, df_split_index):
        train_index = df_split_index[df_split_index['train_type']==0]['index'].to_numpy()
        val_index = df_split_index[df_split_index['train_type']==1]['index'].to_numpy()
        test_index = df_split_index[df_split_index['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index
        
        return train_index, val_index, test_index


class BaseKFold():
    def __init__(self, unique_str: str, split_strategy: str,
                nfold: int, labels, deterministic=True, index_path='./split/', seed=None):
        self.unique_str = unique_str
        self.split_strategy = split_strategy
        self.nfold = nfold
        self.labels = labels
        self.deterministic = deterministic
        self.index_path = index_path
        self.seed = seed
        self.kfolds = self._setup_kfolds()
        
    def get_fold(self, fold):
        if fold > self.nfold:
            raise ValueError(f'fold should be between 1 and nfold={self.nfold}, but got fold={fold}')
        train_index, val_index, test_index = self._df_to_index(self.kfolds[fold])
        logger.debug(f'[spliter summary] (splits {self.split_strategy}) train {len(train_index)} val {len(val_index)} test {len(test_index)}')
        return train_index, val_index, test_index

    def get_kfold_generator(self):
        for fold in range(1, self.nfold+1):
            train_index, val_index, test_index = self._df_to_index(self.kfolds[fold])
            yield train_index, val_index, test_index

    def _split_strategy_(self, labels):
        train_index, val_index, test_index = None,None,None
        yield train_index, val_index, test_index

    def _setup_kfolds(self):
        folder = f'kfold_{self.split_strategy}_{self.unique_str}_nfold={self.nfold}/'
        path = Path(self.index_path).joinpath(folder)
        
        if self.deterministic:
            # read splited index from disk
            path.mkdir(parents=True,exist_ok=True)

            fold_generated = True
            kfolds = {}
            for fold in range(self.nfold):
                path = path.joinpath(f'index_{fold}.csv')
                if not os.path.isfile(path):
                    fold_generated = False
                    break
                df_fold_index = pd.read_csv(path)
                if len(df_fold_index) != len(self.labels):
                    logger.debug(f'[spliter] df_fold_index and labels length not match {len(df_fold_index)} != {len(self.labels)}')
                    fold_generated = False
                    break
                kfolds[fold] = df_fold_index
            
            if fold_generated:
                return kfolds

        kfolds = {}
        for fold, idx in enumerate(self._split_strategy_(self.labels), 1):
            df_fold_index = self._to_df_and_save(path, *idx)
            kfolds[fold] = df_fold_index

        return kfolds

    def _to_df_and_save(self, path, train_idx, val_idx, test_idx):
        train_idx = [(i, 0) for i in train_idx]
        test_idx = [(i, 2) for i in test_idx]

        df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
        df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
        if val_idx is not None:
            val_idx = [(i,1) for i in val_idx]
            df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
            df_tuple = (df_train, df_val, df_test)
        else:
            df_tuple = (df_train, df_test)
        
        df_split_index = pd.concat(df_tuple, axis=0, ignore_index=True)
        if self.deterministic:
            df_split_index.to_csv(path, index=False)
            logger.debug(f'[spliter] index created and saved!')
        else:
            logger.debug(f'[spliter] index created without saving to file!')
        
        return df_split_index

    def _df_to_index(self, df_split_index):
        train_index = df_split_index[df_split_index['train_type']==0]['index'].to_numpy()
        val_index = df_split_index[df_split_index['train_type']==1]['index'].to_numpy()
        test_index = df_split_index[df_split_index['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index
        
        return train_index, val_index, test_index
    