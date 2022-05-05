from copy import Error
from typing import List, Tuple, Union, Mapping
import os

from networkx.algorithms.dag import descendants
from transformers.models.mbart.tokenization_mbart_fast import FAIRSEQ_LANGUAGE_CODES
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch._C import ErrorReport
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter
import ast
from lib.settings.config import settings
from tqdm import tqdm
import pickle
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from nodevectors import Node2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd

import logging
logger = logging.getLogger('utils.twitterdata')

__all__ = ['TwitterData']


@dataclass
class MyNode():
    id: int
    sid: int
    t: float
    def __init__(self, id, sid, t):
        if id == 'ROOT': 
            self.id = 0
        else:
            self.id = int(id)
        if sid == 'ROOT':
            self.sid = 0
        else:
            self.sid = int(sid)
        self.t = float(t)

    def __repr__(self):
        return str(self.t)


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        logger.info("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        logger.info("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class TwitterData():
    def __init__(
        self,
        rootpath=settings.data,
        pretrain_tokenizer_model='bert-base-cased',
        tree='none',
        max_seq_length=128,
        max_tree_length=100,
        train_batch_size=32,
        val_batch_size=32,
        test_batch_size=32,
        limit=100,
        split_type='tvt',
        cv=False,
        n_splits=5,
        datatype='dataloader',
        subclass=False,
        textformat='token',
        kfold_deterministic=False,
        verbose=True,
        **kwargs
    ):
        super().__init__()
        self.rootpath = rootpath
        self.pretrain_tokenizer_model = pretrain_tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrain_tokenizer_model, use_fast=True)
        self.max_seq_length = max_seq_length
        self.tree = tree
        self.max_tree_length = max_tree_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.limit = limit
        self.n_class = 4
        self.split_type = split_type
        self.cv = cv
        self.n_splits = n_splits
        self.feature_dim = 1
        self.setup_flag = True
        self.subclass=subclass
        self.textformat = textformat
        self.kfold_deterministic = kfold_deterministic
        if datatype not in ['dataloader','numpy','all']:
            raise Error('datatype is either "dataloader" or "all"')
        self.datatype = datatype
        self.skip_id = ['523123779124600833']

        ##############
        self._train_data = None
        self._val_data = None
        self._test_data = None
        
    def prepare_data(self):
        AutoTokenizer.from_pretrained(
            self.pretrain_tokenizer_model, use_fast=True)

    def summary(self):
        self.setup(self)

    def setup(self):
        if self.cv:
            self.setup_kfold()
        else:
            if not self.setup_flag:
                return

            self.setup_flag = False
            logger.info('***** setup dataset *****')
            tw15_X, tw15_y, tw16_X, tw16_y = self._load_data()
            train, val, test = self._data_split(self.split_type, tw15_X, tw15_y, tw16_X, tw16_y)
            self._set_data(train, val, test)
            logger.info('***** finish *****')
            logger.info('class to index {}'.format(self.class_to_index))

    def setup_kfold(self):
        if not self.cv:
            logger.error('TwitterData, please set parameter cv == True to use kfold')
            raise Error
        logger.info('***** setup kfold dataset *****')
        self.train, self.val, self.test = None, None, None
        tw15_X, tw15_y, tw16_X, tw16_y = self._load_data()
        X, y = self._build_kfold_data(self.split_type, tw15_X, tw15_y, tw16_X, tw16_y)
        self._X, self._y = X, y

        if self.kfold_deterministic:
            logger.info('setup kfold deterministic')
            self.kfolds = self.kfold_index_build(X, y)
        else:
            logger.info('setup kfold dynamic')
            self.kf = StratifiedKFold(n_splits=self.n_splits,shuffle=True)
        logger.info('***** finish *****')
        logger.info('class to index {}'.format(self.class_to_index))

    def kfold_get_by_fold(self, fold):
        if self.kfolds is None:
            raise UnboundLocalError('Please set kfold deterministic to True')

        df_fold = self.kfolds[fold]
        
        train_index = df_fold[df_fold['train_type']==0]['index'].to_numpy()
        val_index = df_fold[df_fold['train_type']==1]['index'].to_numpy()
        test_index = df_fold[df_fold['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index
        
        X_train, X_val, X_test = self.X[train_index], self.X[val_index], self.X[test_index]
        y_train, y_val, y_test = self.y[train_index], self.y[val_index], self.y[test_index]
        train = [[data, label] for data, label in zip(X_train, y_train)]
        val = [[data, label] for data, label in zip(X_val, y_val)]
        test = [[data, label] for data, label in zip(X_test, y_test)]
        
        logger.info(f'[kfold summary] train {len(train)} val {len(val)} test {len(test)}')
        self._set_data(train, val, test)


    def kfold_gen(self):
        if self.kf is None:
            raise UnboundLocalError('When kfold deterministic is true, please call kfold_get_by_fold')

        if self._X is None:
            logger.error('Please Call setup_kfold() first')
            raise UnboundLocalError

        for i, data in enumerate(self._next_fold(self._X, self._y)):
            train, val, test = data
            self._set_data(train, val, test)

            logger.info(f'kfold {i+1}/{self.n_splits}')
            yield i

    def kfold_index_build(self, X, y):
        if not os.path.isdir('./kfold/'):
            os.mkdir('./kfold/')
        
        logger.debug('kfold_index_build')
        self.X = X
        self.y = y

        fold_generated = True
        kfolds = {}
        for fold in range(5):
            if not os.path.isfile(f'./kfold/{self.split_type}_fold={fold}_nclass={self.n_class}.csv'):
                fold_generated = False
                break
        
            df_fold = pd.read_csv(f'./kfold/{self.split_type}_fold={fold}_nclass={self.n_class}.csv')
            kfolds[fold] = df_fold
        
        if fold_generated:
            return kfolds
        
        # label train : 0, val: 1, test: 2
        fold = 0
        kfolds = {}
        kf = StratifiedKFold(n_splits=self.n_splits)
        for train_index, test_index in kf.split(X, y):
            val_index = None
            if self.split_type.split('_')[1] == 'tvt':
                y_train = y[train_index]
                idx_train, idx_val = train_test_split(
                    train_index, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)

                train_index = idx_train
                val_index = idx_val
            elif self.split_type.split('_')[1] == 'ttv':
                y_test = y[test_index]
                idx_val, idx_test = train_test_split(
                    test_index, train_size=0.5, random_state=1, shuffle=True, stratify=y_test)

                test_index = idx_test
                val_index = idx_val
            
            train_idx = [(i, 0) for i in train_index]
            test_idx = [(i, 2) for i in test_index]

            df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
            df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
            if val_index is not None:
                val_idx = [(i,1) for i in val_index]
                df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
                df_fold = pd.concat((df_train, df_val, df_test),axis=0, ignore_index=True)
            else:
                df_fold = pd.concat((df_train, df_test),axis=0, ignore_index=True)
            df_fold.to_csv(f'./kfold/{self.split_type}_fold={fold}_nclass={self.n_class}.csv',index=False)
            
            kfolds[fold] = df_fold

            fold += 1
        
        return kfolds

    def _next_fold(self, X, y):
        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if self.split_type.split('_')[1] == 'tvt':
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)
                
                train = [[data, label] for data, label in zip(X_train, y_train)]
                val = [[data, label] for data, label in zip(X_val, y_val)]
                test = [[data, label] for data, label in zip(X_test, y_test)]
                
                split_sum = len(train) + len(val) + len(test)
                tr_ = int(len(train)/split_sum*100)
                va_ = int(len(val)/split_sum*100)
                te_ = int(len(test)/split_sum*100)
                logger.info(f'train_val_test split ratio {tr_}:{va_}:{te_}')

                yield train,val,test

            elif self.split_type.split('_')[1] == 'tv':
                train = [[data, label] for data, label in zip(X_train, y_train)]
                val = [[data, label] for data, label in zip(X_test, y_test)]
                test = [[data, label] for data, label in zip(X_test, y_test)]

                split_sum = len(train) + len(test)
                tr_ = int(len(train)/split_sum*100)
                te_ = int(len(test)/split_sum*100)
                logger.info(f'train_test split ratio {tr_}:{te_}')

                yield train,val,test

    def _convert_to_features_all(self, train, val, test, datatype):
        dataset = {'train': train, 'val': val, 'test': test}
        for split in dataset.keys():
            d = dataset[split]
            if d is None:
                continue
            
            dataset[split] = self._convert_to_features(d, datatype)
        return dataset

    def _convert_to_features(self, example, datatype, indices=None):
        source = []
        tree = []
        label = []
        for x, y in example:
            source.append(x[0])
            tree.append(x[1])
            label.append(y)

        features = self.tokenizer.batch_encode_plus(
            list(source),
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
        )

        features_ = []
        if datatype == 'dataloader':
            logger.debug('self.pretrain_tokenizer_model {}'.format(self.pretrain_tokenizer_model))
            if self.pretrain_tokenizer_model.split('-')[0] == 'bert':
                if self.tree == 'node2vec' or self.tree == 'tree':
                    for i in range(len(label)):
                        features_.append((torch.tensor(features['input_ids'][i]), torch.tensor(features['token_type_ids'][i]),
                                        torch.tensor(features['attention_mask'][i]), torch.tensor(tree[i],dtype=torch.float32), torch.tensor(label[i])))
                elif self.tree == 'none':
                    for i in range(len(label)):
                        features_.append((torch.tensor(features['input_ids'][i]),  torch.tensor(features['token_type_ids'][i]),
                                        torch.tensor(features['attention_mask'][i]), torch.tensor(label[i])))
            elif self.pretrain_tokenizer_model.split('-')[0] == 'roberta':
                if self.tree == 'node2vec' or self.tree == 'tree':
                    for i in range(len(label)):
                        features_.append((torch.tensor(features['input_ids'][i]),torch.tensor(features['attention_mask'][i]),
                                        torch.tensor(tree[i],dtype=torch.float32), torch.tensor(label[i])))
                elif self.tree == 'none':
                    for i in range(len(label)):
                        features_.append((torch.tensor(features['input_ids'][i]), torch.tensor(features['attention_mask'][i]), 
                                        torch.tensor(label[i])))
            else:
                raise ValueError(f'pretrain_tokenizer_model {self.pretrain_tokenizer_model} is incorrect')
            
            return features_
        elif datatype == 'numpy':
            citt = self.tokenizer.convert_ids_to_tokens
            
            if self.textformat == 'raw':
                if self.tree == 'node2vec' or self.tree == 'tree':
                    for i in range(len(label)):
                        features_.append((np.array(source[i]), np.array(tree[i]), np.array(label[i])))
                elif self.tree == 'none':
                    for i in range(len(label)):
                        features_.append((np.array(source[i]), np.array(label[i])))
            elif self.textformat == 'token':
                if self.tree == 'node2vec' or self.tree == 'tree':
                    for i in range(len(label)):
                        features_.append((np.array(' '.join(citt(features['input_ids'][i]))), np.array(tree[i]), np.array(label[i])))
                elif self.tree == 'none':
                    for i in range(len(label)):
                        features_.append((np.array(' '.join(citt(features['input_ids'][i]))), np.array(label[i])))
            else:
                raise Error('textformat should be "raw" or "token"')
            return features_
    
    def _load_data(self):
        tw = ['twitter15','twitter16']
        data, trees = {}, {}
        if self.tree == 'node2vec':
            wv = self._create_node2vec(tw)

        for t in tw:
            source_p = os.path.join(self.rootpath,t,'source_tweets.txt')
            label_p = os.path.join(self.rootpath,t,'label.txt')
            tree_p = os.path.join(self.rootpath,t,'tree')
            #data[t] = self._combine_text_label(self._read_text(source_p), self._read_label(label_p))

            if self.tree == 'node2vec':
                trees = self._encode_graph(wv,tree_p)
            elif self.tree == 'gcn':
                #trees = self
                pass
            else:
                tree_map = self._read_tree(t, tree_p)
                trees, mean, std = self._encode_tree(t,tree_map,self.max_tree_length,padding=True)
            
            data[t] = self._combine_data(self._read_text(source_p), trees, self._read_label(label_p))

        
        tw15_X, tw15_y = data[tw[0]]
        tw16_X, tw16_y = data[tw[1]]

        if self.subclass:
            tw15_X,tw15_y = self._class_filter(tw15_X,tw15_y)
            tw16_X,tw16_y = self._class_filter(tw16_X,tw16_y)
        
        self._find_class(tw15_y, tw16_y)

        tw15_y = self._class_to_index(tw15_y)
        tw16_y = self._class_to_index(tw16_y)
        return tw15_X, tw15_y, tw16_X, tw16_y

    def _find_class(self, label1, label2):
        label = np.concatenate((label1, label2))
        classes = sorted(np.unique(label))
        self.class_to_index = {classname: i for i,
                               classname in enumerate(classes)}
        logger.info('class_to_index {}'.format(self.class_to_index))
        self.class_names = classes
        self.n_class = len(classes)
        self.classes = [i for i in range(len(self.class_names))]

    def _class_to_index(self, label):
        index = np.vectorize(self.class_to_index.__getitem__)(label)
        return index

    def _class_filter(self, X, y):
        index = np.where((y == 'false') | (y=='true'),True,False)
        X_ = X[index]
        y_ = y[index]
        return X_, y_

    def _build_kfold_data(self, split_type, tw15_X, tw15_y, tw16_X, tw16_y):
        X, y = None, None
        if len(split_type.split('_')) != 2:
            raise Error('split_type should be in "15_tv" or "16_tvt"')

        if split_type.split('_')[0] == 'all':
            X = np.concatenate((tw15_X, tw16_X))
            y = np.concatenate((tw15_y, tw16_y))
            
        elif split_type.split('_')[0] == '15':
            X, y = tw15_X, tw15_y

        elif split_type.split('_')[0] == '16':
            X, y = tw16_X, tw16_y
        
        else:
            raise Error(f'split_type is {split_type}, split_type should be in "15_tv" or "16_tvt"')
        
        return X, y

    def _data_split(self, split_type, tw15_X, tw15_y, tw16_X, tw16_y):
        X, y = None, None
        if split_type.split('_')[0] == 'all':
            X = np.concatenate((tw15_X, tw16_X))
            y = np.concatenate((tw15_y, tw16_y))

        if split_type.split('_')[0] == '15':
            X = tw15_X
            y = tw15_y

        if split_type.split('_')[0] == '16':
            X = tw16_X
            y = tw16_y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=1, shuffle=True, stratify=y)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)

        train = [[data, label] for data, label in zip(X_train, y_train)]
        val = [[data, label] for data, label in zip(X_val, y_val)]
        test = [[data, label] for data, label in zip(X_test, y_test)]

        return train, val, test

    def _set_data(self, train, val, test):
        if self.datatype == 'dataloader':
            self._set_dataloader(train, val, test)
        elif self.datatype == 'numpy':
            self._set_numpy_data(train, val, test)
        elif self.datatype == 'all':
            self._set_dataloader(train, val, test)
            self._set_numpy_data(train, val, test)
        else:
            raise ValueError(f'DataType is incorrect {self.datatype}')

    def _set_numpy_data(self, train, val, test):
        dataset = self._convert_to_features_all(train, val, test, 'numpy')
        #dataset = {'train':train,'val':val,'test':test}
        self.np_dataset = {}
        for k, data in dataset.items():
            x, t, y = [], [], []
            if self.tree == 'none':
                for xi, yi in data:
                    x.append(xi)
                    y.append(yi)
                x = np.array(x)
                y = np.array(y)
                self.np_dataset[k] = (x,None,y)
            elif self.tree == 'tree':
                for xi, ti, yi in data:
                    x.append(xi)
                    t.append(ti)
                    y.append(yi)
                x = np.array(x)
                t = np.array(t)
                y = np.array(y)
                self.np_dataset[k] = (x,t,y)

    @property
    def train_data(self):
        return self.np_dataset['train']

    @property
    def val_data(self):
        return self.np_dataset['val']

    @property
    def test_data(self):
        return self.np_dataset['test']

    def _set_dataloader(self, train, val, test, shuffle=True):
        dataset = self._convert_to_features_all(train, val, test, 'dataloader')
        if torch.cuda.is_available():
            logger.info(f'[GPU][TB] {torch.cuda.memory_reserved()/1024/1024/1024} GB')
        
        if self._train_data is not None:
            del self._train_data
        if self._test_data is not None:
            del self._test_data
        if self._val_data is not None:
            del self._val_data
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f'[GPU][TA] {torch.cuda.memory_reserved()/1024/1024/1024} GB')

        self._train_data = DataLoader(dataset['train'],
                                      batch_size=self.train_batch_size,
                                      pin_memory=True,
                                      drop_last=True,
                                      shuffle=shuffle,
                                      num_workers=8)
        self._test_data = DataLoader(dataset['test'],
                                     batch_size=self.test_batch_size,
                                     pin_memory=True,
                                     shuffle=False,
                                     num_workers=8)
        if dataset['val'] is not None:
            self._val_data = DataLoader(dataset['val'],
                                        batch_size=self.val_batch_size,
                                        pin_memory=True,
                                        shuffle=False,
                                        num_workers=8)

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_data

    @property
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._val_data

    @property
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._test_data

    def _combine_text_label(self, texts, labels):
        text_label = []
        for id, text in texts.items():
            label = labels[id]
            text_label.append([text, label])

        text_label = np.array(text_label)

        return text_label[:, 0], text_label[:, 1]

    def _combine_data(self, texts, trees, labels):
        data = []

        for id, text in texts.items():
            if str(id) in self.skip_id:
                continue
            label = labels[id]
            tree = trees[id]
            data.append([text, tree, label])

        data = np.array(data,dtype=object)
        data = self._normalize_data(data)
        
        return data[:, 0:2], data[:,2]

    def _normalize_data(self, data):
        return data

    def _read_text(self, path):
        pairs = {}
        with open(path, mode='r') as f:
            for line in f:
                id, text = line.split('\t')
                if id not in pairs.keys():

                    pairs[int(id)] = text
                else:
                    logger.error('error')
        return pairs

    def _read_label(self, path):
        pairs = {}
        with open(path, mode='r') as f:
            for line in f:
                label, id = line.split(':')
                if id not in pairs.keys():

                    pairs[int(id)] = label
                else:
                    logger.error('error')
        return pairs

    def _read_tree(self, t, path):
        pickle_fn = f"tree_maps_{t}.p"
        if os.path.isfile(os.path.join(settings.checkpoint,pickle_fn)):
            tree_map = pickle.load(open(os.path.join(settings.checkpoint,pickle_fn), "rb" ))
            logger.info(f'load {pickle_fn}')
            return tree_map
        
        tree_map = {}
        for fn in tqdm(os.listdir(path)):
            index = fn.split('.')[0]
            tree_map[int(index)] = self._build_tree(os.path.join(path,fn))
        
        pickle.dump(tree_map, open(os.path.join(settings.checkpoint,pickle_fn), "wb"))
        logger.info(f'saved {pickle_fn}')
        return tree_map

    def _build_tree(self, fn):
        root = None
        nodemap = {}
        with open(fn, mode='r') as f:
            for line in f:
                splited = line.split('->')
                p = ast.literal_eval(splited[0])
                c = ast.literal_eval(splited[1])
                np = Node(MyNode(*p))
                
                if root is None and np.name.id == 0:
                    root = Node(MyNode(*c))
                    nodemap[root.name.id] = root
                    continue
                    
                if np.name.id not in nodemap:
                    nodemap[np.name.id] = np
                myp = nodemap[np.name.id]

                nc = Node(MyNode(*c), parent=myp)
                nodemap[nc.name.id] = nc
        
        return root

    def TimeOrderIter(self, root: Node):
        def sortkey(node):
            return node.name.t
        allnodes = [root]
        allnodes.extend(list(root.descendants))
        allnodes.sort(key=sortkey)

        for node in allnodes:
            yield node

    def _padding_trees(self, encoded_trees: Mapping[str, np.array], max_length=500, random_choice=False):
        padded_encodings = {}
        for index, encoding  in encoded_trees.items():
            encoding = encoding.T
            len_e = len(encoding)
            diff =  max_length - len_e

            if diff > 0:
                if not random_choice:
                    encoding = np.pad(encoding, [(0, diff),(0, 0)])
                    #encoding = np.pad(encoding, (0, diff))
                else:
                    rows = range(encoding.shape[0])
                    indexs = np.random.choice(rows, diff, replace=True)
                    rds = encoding[indexs,:]
                    encoding = np.concatenate((encoding, rds))
            #print(f'__padding_trees {len_e} {len(padded_encoding)}')
            padded_encodings[index] = encoding.T
        
        return padded_encodings

    def _encode_tree(self, split_type, tree_map: Mapping[str, Node], max_length=500, padding=False, log_transform=False, deduct_first=False, random_choice=False, standarization=False, time_of_reply=False):
        encoded_trees = {}
        self.feature_dim = 7
        for index in sorted(tree_map.keys()):
            if max_length == 0:
                encoded_trees[index] = []
                continue
            root = tree_map[index]
            total_e = len(root.descendants)
            total_h = root.height
            #print(len(root.descendants), root.is_root, len(root.leaves), total_h)
            root_t = root.name.t
            encoding = []

            for i, node in enumerate(self.TimeOrderIter(root)):
            #for i, node in enumerate(LevelOrderIter(root)):
                if max_length != -1 and i >= max_length:
                    break
                
                if node.name.t-root_t < 0:
                   continue

                if not time_of_reply:
                    time_elapse = node.name.t-root_t
                else:
                    if node.parent is None:
                        parent_t = root_t
                    else:
                        parent_t = node.parent.name.t
                    time_elapse = node.name.t - parent_t
                if time_elapse < 0:
                    logger.debug(index, time_elapse)
                    continue
                element_node = (time_elapse,
                                len(node.children)/total_e, 
                                node.depth/total_h,
                                len(node.siblings)/total_e,
                                len(node.descendants)/total_e,
                                float(node.is_leaf), 
                                float(node.is_root))
                
                encoding.append(element_node)

            encoding = np.array(encoding)
            if deduct_first:
                encoding = encoding - encoding[1]
                encoding[0] = 0.0
            
            '''
            df_elapsed_time = pd.DataFrame(encoding[:,0])
            df_elapsed_time.to_csv('elapsed_time_1.csv',index=False)
            print(df_elapsed_time)
            exit()
            ''' 

            if log_transform:
                en_log = np.log10(encoding[:,0]+1)
                encoding[:,0] = en_log
            
            if False and padding:
                len_e = len(encoding)
                
                diff =  max_length - len_e
                if diff > 0:
                    if not random_choice:
                        encoding = np.pad(encoding, [(0, diff),(0, 0)])
                        #encoding = np.pad(encoding, (0, diff))
                    else:
                        rows = range(encoding.shape[0])
                        indexs = np.random.choice(rows, diff, replace=True)
                        rds = encoding[indexs,:]
                        encoding = np.concatenate((encoding, rds))

            #print(encoding.shape)
            encoding = encoding.T
            
            #print(encoding.shape)
            encoded_trees[index] = encoding
        
        if max_length == 0:
            return encoded_trees, 0, 0
        
        power_transform = True

        if power_transform:
            my_mean = 0
            my_std = 0
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='box-cox')
            all_times = []
            to_avoid_zero_value = 0.00000001
            for k in encoded_trees:
                all_times.append(encoded_trees[k][0,:]+to_avoid_zero_value)
            
            all_times = np.concatenate(all_times)
            pt.fit(all_times.reshape(-1,1))
            for k in encoded_trees:
                rest = pt.transform((encoded_trees[k][0,:]+to_avoid_zero_value).reshape(-1,1))
                encoded_trees[k][0,:] = rest.reshape(-1)

            #df_all_time = pd.DataFrame(all_times)
            #df_all_time.to_csv(f'all_time_{split_type}_{max_length}.csv',index=False)
        else:
            avg = []
            
            for k, v in encoded_trees.items():
                avg.append(np.average(v[0,:]))
            
            my_mean = np.average(avg)
            my_std = np.std(avg)

            avg = []
            for k in encoded_trees.keys():
                if not standarization:
                    encoded_trees[k][0,:] = (encoded_trees[k][0,:] - my_mean)
                else:
                    encoded_trees[k][0,:] = (encoded_trees[k][0,:] - my_mean)/my_std

                avg.append(np.average(encoded_trees[k][0,:]))
        
        if padding:
            logger.debug('padding true')
            encoded_trees = self._padding_trees(encoded_trees,max_length,random_choice)
        
        logger.info(f'[Standarization] mean {my_mean} std {my_std}')
        '''
        print('my_mean ', my_mean, ' my_std', my_std)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5))
        ax.hist(avg,bins=list(np.arange(-5,5,0.1)))
        plt.savefig('twitter_avg.png')
        plt.close()
        '''

        return encoded_trees, my_mean, my_std

    def _create_gcn(self, tw):
        
        path = os.path.join(self.rootpath,t,'tree')
        for fn in tqdm(os.listdir(path)):
            DG = self._build_graph(DG, os.path.join(path,fn), self.limit)

    def _create_node2vec(self, tw):
        dimensions = int(self.max_tree_length)
        limit = self.limit
        window = self.limit
        # window = 10
        fn_word2vec_emb = f'keyedvec_d={dimensions}_l={limit}_w={window}.emb'
        path_word2vec_emb = os.path.join('./checkpoint',fn_word2vec_emb)
        graph_fn = f'weighted_l={limit}.edgelist'
        graph_path = os.path.join('./checkpoint',graph_fn)
        DG = nx.DiGraph()
        if not os.path.isfile(path_word2vec_emb) or not os.path.isfile(graph_path):
            if os.path.isfile(graph_path):
                logger.debug('read_weighted_edgelist')
                DG = nx.read_weighted_edgelist(graph_path,create_using=nx.DiGraph)
            else:
                for t in tw:
                    path = os.path.join(self.rootpath,t,'tree')
                    for fn in tqdm(os.listdir(path)):
                        DG = self._build_graph(DG,os.path.join(path,fn),limit)
                nx.write_weighted_edgelist(DG,graph_path)
            self._learn_node2vec_nodevectors(DG, dimensions,path_word2vec_emb, window)
        return KeyedVectors.load_word2vec_format(path_word2vec_emb)

    def _learn_node2vec_nodevectors(self, G, dimension, path_word2vec_emb, window):
        logger.debug('_learn_node2vec_nodevectors')
        epoch_logger = EpochLogger()
        g2v = Node2Vec(n_components=dimension,verbose=True,w2vparams={'window':window,'negative':5,'iter':1,'batch_words':128,'workers':56,'callbacks':[epoch_logger]})
        g2v.fit(G)
        g2v.save(f'./checkpoint/node2vec_d={dimension}_l={self.limit}_w={window}')
        g2v.save_vectors(path_word2vec_emb)
        return

    def _learn_node2vec(self, DG, word2vec_emb):
        num_walk = 10
        walk_len = 80
        dimensions = self.max_tree_length
        window_size = 10
        workers = 56
        itern = 10
        import time
        G = node2vec.Graph(DG,False,1,1)
        logger.debug('preprocess_transition_probs')
        ts = time.time()
        G.preprocess_transition_probs()
        te = time.time()
        logger.debug(f'preprocess_transition_probs {te-ts:.02f}s ')
        logger.debug('simulate_walks')
        ts = time.time()
        walks = G.simulate_walks(num_walk,walk_len)
        te = time.time()
        logger.debug(f'simulate_walks {te-ts:02f}s')
        logger.debug(f'learn_embeddings')
        ts = time.time()
        self.learn_embeddings(word2vec_emb,walks,dimensions,window_size,workers,itern)
        te = time.time()
        logger.debug(f'learn_embeddings {te-ts:02f}s')
        wv = KeyedVectors.load_word2vec_format(os.path.join('./checkpoint',word2vec_emb))
        return wv

    def _build_graph(self, DG, fn, limit=20):
        count = 0
        limit_count = 0
        root_find = False
        with open(fn, mode='r') as f:
            for line in f:
                splited = line.split('->')
                p = ast.literal_eval(splited[0])
                c = ast.literal_eval(splited[1])
                if limit_count >= limit:
                    if root_find:
                        break
                    else:
                        logger.debug('root find ',line)
                        if p[1] == 'ROOT':
                            root_find = True
                            break
                        continue
                
                if p[1] == 'ROOT':
                    root_find = True
                    continue
                # if p[1] == c[1]:
                #     continue
                weight = float(c[2]) - float(p[2])
                if weight == 0.0:
                    weight = 0.001
                if weight < 0 or float(c[2]) < 0 or float(p[2]) < 0:
                    count += 1
                    continue
                edge = (p[0]+'_'+p[1],c[0]+'_'+c[1],weight)
                limit_count += 1
                DG.add_weighted_edges_from([edge])
        
        return DG

    def learn_embeddings(self, name, walks, dimensions, window_size,workers,itern):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=itern)
        model.wv.save_word2vec_format(os.path.join('./checkpoint',name))
        return

    def _encode_graph(self, wv, path):
        graph_map = {}
        for fn in os.listdir(path):
            index = fn.split('.')[0]
            root_index = None
            
            with open(os.path.join(path,fn), mode='r') as f:
                for line in f:
                    splited = line.split('->')
                    p = ast.literal_eval(splited[0])
                    c = ast.literal_eval(splited[1])
                    if p[1] == 'ROOT':
                        root_index =  c[0]+'_'+c[1]
                        if c[1] != index:
                            logger.debug(fn)
                        break
            if index in self.skip_id:
                continue
            try:
                vector = wv[root_index]
                graph_map[int(index)] = vector
            except:
                vector = None
                self.skip_id.append(index)
                graph_map[int(index)] = vector
        return graph_map