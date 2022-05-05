import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from anytree import Node
from tqdm import tqdm
from src.logger.get_configured_logger import TqdmToLogger
from src.utils.decorator import buffer_value


logger = logging.getLogger(__name__)
tqdm_out = TqdmToLogger(logger, level=logging.INFO)

@dataclass
class MyNode():
    id: int
    sid: int  # session_id
    t: float  # timestamp
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

def load_data(data_root: str, data_name: str, max_tree_length):
    data_root = Path(data_root).joinpath(data_name)
    source_path = data_root.joinpath('source_tweets.txt')
    label_path = data_root.joinpath('label.txt')
    tree_path = data_root.joinpath('tree')
    source_tweets = load_text(source_path)
    labels = load_label(label_path)

    trees = load_all_trees(f'tree_{data_name}',tree_path)
    encoded_trees = tree_sequential_encoding(f'encoded_tree_{data_name}',trees, max_tree_length)
    
    x_text, x_tree, y = link_all_data(source_tweets, encoded_trees, labels)
    return x_text, x_tree, y

@buffer_value('pickle', '.temp')
def load_all_trees(tree_path: Path):
    logger.debug('[load_all_trees] start')
    tree_map = {}
    files = os.listdir(tree_path)
    for fn in tqdm(files,file=tqdm_out):
        # fn = files[i]
        index = fn.split('.')[0]
        tree_map[int(index)] = load_tree(tree_path.joinpath(fn))
    logger.debug('[load_all_trees] finished')
    return tree_map


def load_tree(fn):
    # logger.debug('[load_tree] start')
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
    # logger.debug('[load_tree] finished')
    return root

def load_text(path):
    logger.debug('[load_text] start')
    source_tweets = {}
    with open(path, mode='r') as f:
        for line in f:
            id, text = line.split('\t')
            if int(id) not in source_tweets:
                source_tweets[int(id)] = text
            else:
                logger.error('error')
    logger.debug('[load_text] finished')
    return source_tweets

def load_label(path):
    logger.debug('[load_label] start')
    pairs = {}
    with open(path, mode='r') as f:
        for line in f:
            label, id = line.split(':')
            if int(id) not in pairs.keys():
                pairs[int(id)] = label
            else:
                logger.error('error')
    logger.debug('[load_label] finished')
    return pairs

def link_all_data(source_tweets, trees, labels):
    # data = []
    skip_id = ['523123779124600833']
    text_list = []
    tree_list = []
    label_list = []
    for id, text in source_tweets.items():
        if str(id) in skip_id:
            continue
        text_list.append(text)
        tree_list.append(trees[id])
        label_list.append(labels[id])
        # label = labels[id]
        # tree = trees[id]
        # data.append([text, tree, label])

    # data = np.array(data,dtype=object)
    # return data[:, 0:1], data[:,1:2], data[:,2:]
    return np.array(text_list), np.array(tree_list, dtype=np.float32), np.array(label_list)

def perform_power_transform_on_timestamp(encoded_trees):
    # power transform
    transformed_trees = {}
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='box-cox')
    all_times = []
    to_avoid_zero_value = 0.00000001
    for k in encoded_trees:
        all_times.append(encoded_trees[k][0,:]+to_avoid_zero_value)
    
    all_times = np.concatenate(all_times)
    pt.fit(all_times.reshape(-1,1))
    for k, tree_encoding in encoded_trees.items():
        timestamps = tree_encoding[0,:] + to_avoid_zero_value
        transformed_timestamps = pt.transform(timestamps.reshape(-1,1))
        # rest = pt.transform((encoded_trees[k][0,:]+to_avoid_zero_value).reshape(-1,1))
        # encoded_trees[k][0,:] = rest.reshape(-1)
        transformed_trees[k] = np.concatenate((transformed_timestamps.reshape(1,-1),tree_encoding[1:,:]))

    return transformed_trees

def padding_trees(encoded_trees: Dict[str, np.ndarray], max_tree_length=500):
    padded_encodings = {}
    for index, encoding  in encoded_trees.items():
        len_e = len(encoding)
        
        diff =  max_tree_length - len_e

        if diff > 0:
            rows = range(encoding.shape[0])
            indexs = np.random.choice(rows, diff, replace=True)
            rds = encoding[indexs,:]
            encoding = np.concatenate((encoding, rds))
        padded_encodings[index] = encoding
    return padded_encodings

def TimeOrderIter(root: Node):
    def sortkey(node):
        return node.name.t
    allnodes = [root]
    allnodes.extend(list(root.descendants))
    allnodes.sort(key=sortkey)

    for node in allnodes:
        yield node

def feature_extraction(root_node, max_tree_length):
    # tree_nodes: total number of nodes in the tree
    tree_nodes = len(root_node.descendants)
    # tree_height: the height of the tree
    tree_height = root_node.height
    # root_t: the timestamp of root node
    root_t = root_node.name.t

    encoding = []

    for i , node in enumerate(TimeOrderIter(root_node)):
        if max_tree_length != -1 and i >= max_tree_length:
            break
        if node.name.t - root_t < 0:
            continue

        time_elapse = node.name.t-root_t
        # sevent features extracted with considering structure information 
        # and time information
        node_features = (
            time_elapse,                        # elapsed time since source tweet
            len(node.children)/tree_nodes,      # number of children
            node.depth/tree_height,             # depth of propagation
            len(node.siblings)/tree_nodes,      # number of siblings
            len(node.descendants)/tree_nodes,   # number of all descendants
            float(node.is_leaf),                # whether it's a leaf node
            float(node.is_root)                 # whether it's a root node
        )
        encoding.append(node_features)

    return encoding

def feature_extraction_for_all_trees(tree_map, max_tree_length):
    encoded_trees: Dict[str, np.ndarray] = {}
    
    for index in tqdm(sorted(tree_map.keys()), file=tqdm_out):
        if max_tree_length == 0:
            encoded_trees[index] = []
            continue

        root_node = tree_map[index]
        encoding = feature_extraction(root_node, max_tree_length)
        encoded_trees[index] = np.array(encoding)
    return encoded_trees

@buffer_value('pickle','.temp')
def tree_sequential_encoding(
        tree_map: Dict[str, Node], 
        max_tree_length=500,
    ):
    encoded_trees = feature_extraction_for_all_trees(tree_map, max_tree_length)

    if max_tree_length == 0:
        return encoded_trees

    encoded_trees = perform_power_transform_on_timestamp(encoded_trees)
    encoded_trees  = padding_trees(encoded_trees, max_tree_length)
    return encoded_trees


