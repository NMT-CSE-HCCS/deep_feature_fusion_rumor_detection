import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging

import numpy as np
import torch
from transformers import AutoTokenizer

from ..dataset_base import DatasetBase, find_class
from .load_twitter_dataset import load_data
from .Twitter_spliter import TwitterKFold
from pathlib import Path

logger = logging.getLogger(__name__)


class TwitterDataset(DatasetBase):
    def __init__(self, root, dataset, tokenizer, max_token, max_tree,
            batch_size, nfold, deterministic, num_workers, seed
    ) -> None:
        super().__init__(batch_size, num_workers)
        self.root = Path(root)
        if not self.root.exists():
            msg = f"dataset_root '{self.root}' doesn't exist.\n" + \
                "Please modify 'dataset_root' in options.py file"
            raise ValueError(msg)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_token = max_token
        self.max_tree = max_tree
        self.nfold = nfold
        self.deterministic = deterministic
        self.seed = seed

    @property
    def key_name(self):
        _key_name = f'dataset={self.dataset}'
        return _key_name

    def on_load_data(self):
        self.tweet_text, self.x_tree, self.y = load_data(
            self.root, self.dataset, self.max_tree)
        self.labels, self.nclass, self.class_to_index = find_class(self.y)
        self.x_text = self._text_tokenizing(self.tweet_text, self.tokenizer, self.max_token)
        return {**self.x_text, "tree": self.x_tree, 'label': self.labels}

    def _text_tokenizing(self, text, tokenizer, max_token):
        _tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=True)
        tokenzied_txt = _tokenizer.batch_encode_plus(
            text.reshape(-1).tolist(),
            max_length=max_token,
            padding='max_length',
            truncation=True,
        )
        text_feature_dict = {}
        for k,v in tokenzied_txt.items():
            text_feature_dict[k] = np.array(v)
        
        return text_feature_dict

    def setup_fold(self, fold):
        """setup current fold 
        fold: (1, nfold)
        """
        self.fold = fold

    def on_split(self):
        if not hasattr(self, 'fold'):
            raise ValueError('Call setup_fold in advance')
        spliter = TwitterKFold(self.key_name, split_strategy='64:16:20', nfold=5,
                    labels=self.labels, deterministic=self.deterministic, index_path='./split/', seed=self.seed)
        return spliter.get_fold(self.fold)
    
    def _to_list(self):
        features_list = []
        if self.tokenizer.split('-')[0] == 'bert':
            for i in range(len(self.y)):
                features_list.append((
                    torch.tensor(self.txt_fea['input_ids'][i]),
                    torch.tensor(self.txt_fea['token_type_ids'][i]),
                    torch.tensor(self.txt_fea['attention_mask'][i]),
                    torch.tensor(self.x_tree[i],dtype=torch.float32),
                    torch.tensor(self.y[i])
                ))
