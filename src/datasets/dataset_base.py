from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from typing import Union, List
import logging

logger = logging.getLogger(__name__)

def find_class(labels: Union[np.ndarray, List]):
    classes = sorted(np.unique(labels))
    class_to_index = {classname: i for i,
                            classname in enumerate(classes)}
    logger.info(f'class_to_index {class_to_index}')
    nclass = len(classes)
    index = np.vectorize(class_to_index.__getitem__)(labels)
    if len(index.shape) == 2:
        index = index.reshape(-1)
    logger.info(f'Label counts: {list(enumerate(np.bincount(index)))}')
    return index, nclass, class_to_index

class DatasetHooks():
    def __init__(self) -> None:
        pass

    def on_load_data(self):
        """"""
        
    def on_data_augmentation(self, data):
        """"""

    def normalize_fit_transform(self, data):
        """"""

    def normalize_transform(self, data):
        """"""

    def setup_fold(self, fold):
        """"""

    def on_split(self):
        """"""

    

class DatasetBase(LightningDataModule, DatasetHooks):
    def __init__(self, batch_size,num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # {'fea1': fea1, 'fea2': fea1, 'label': label}
        data_dict = self.on_load_data()
        train_index, val_index, test_index = self.on_split()
        self.train_class_weight = np.bincount(self.labels[train_index])
        self.train_data = []
        self.val_data = []
        self.test_data = []
        for name, data in data_dict.items():
            self.train_data.append((name, data[train_index]))
            self.val_data.append((name, data[val_index]))
            self.test_data.append((name, data[test_index]))

    def setup(self, stage):
        if stage in (None, 'fit'):
            norm_train_data, norm_val_data = self._on_normalize(stage, (self.train_data, self.val_data))
            aug_train_data = self._data_augmentation(norm_train_data)
            self.train_set = self._to_dataset(aug_train_data)
            self.val_set = self._to_dataset(norm_val_data)
        if stage in (None, 'test', 'predict'):
            norm_test_data = self._on_normalize(stage, self.test_data)
            self.test_set = self._to_dataset(norm_test_data)

    def _to_dataset(self, data):
        tensor_list = []
        # features
        for name, d in data:
            if name == 'label':
                tensor_list += [torch.tensor(d, dtype=torch.long)]
            else:
                tensor_list += [torch.tensor(d)]
        
        return TensorDataset(*tensor_list)

    def _data_augmentation(self, data):
        ret = self.on_data_augmentation(data)
        if ret is None:
            return data
        return ret

    def _on_normalize(self, stage, data):
        if stage in (None, 'fit'):
            train_data, val_data = data
            norm_train_data = self.normalize_fit_transform(train_data)
            norm_val_data = self.normalize_transform(val_data)
            if norm_train_data is None or norm_val_data is None:
                return train_data, val_data
            return norm_train_data, norm_train_data

        if stage in (None, 'test', 'predict'):
            test_data = data
            nrom_test_data = self.normalize_transform(test_data)
            if nrom_test_data is None:
                return test_data
            return nrom_test_data

    def _to_dataloader(self, dataset, shuffle, batch_size, drop_last=False, sampler=None):
        if sampler: shuffle = False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers
        )

    def train_dataloader(self):
        return self._to_dataloader(self.train_set, True, self.batch_size)

    def val_dataloader(self):
        return self._to_dataloader(self.val_set, False, self.batch_size)

    def test_dataloader(self):
        return self._to_dataloader(self.test_set, False, self.batch_size)
    