from sklearn.model_selection import StratifiedKFold, train_test_split
from ..dataset_spliter import BaseKFold
import numpy as np
from src.utils.random_seed import get_timebaesed_random_seed

class TwitterKFold(BaseKFold):
    def __init__(self, unique_str: str, split_strategy: str, nfold: int, labels, deterministic=True, index_path='./split/', seed=None):
        super().__init__(unique_str, split_strategy, nfold, labels, deterministic, index_path, seed)

    def _split_strategy_(self, labels):
        if self.split_strategy == '64:16:20':
            if self.seed is None or not self.deterministic:
                seed = get_timebaesed_random_seed()
            
            skf = StratifiedKFold(n_splits=self.nfold, shuffle=True, random_state=seed)
            for train_index, test_index in skf.split(np.arange(len(labels)), labels):
                labels_train = labels[train_index]
                idx_train, idx_val = train_test_split(
                    train_index, train_size=0.8, random_state=seed, shuffle=True, stratify=labels_train)

                train_index = idx_train
                val_index = idx_val
                yield train_index, val_index, test_index
        return super()._split_strategy_(labels)