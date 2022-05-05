import pickle
import os
import pandas as pd
import marshal
import inspect
import joblib
from typing import Any
from time import time
import logging
import traceback

logger = logging.getLogger(__name__)

__all__ = ['buffer_value']

def pandas_to_file(df: pd.DataFrame, path: str):
    df.to_csv(path)

def pandas_from_file(path: str):
    return pd.read_csv(path)

def pickle_to_file(object: Any, path: str):
    pickle.dump(object,open(path,'wb'))

def pickle_from_file(path: str):
    return pickle.load(open(path,'rb'))

def joblib_to_file(object: Any, path: str, compress: int = 0):
    joblib.dump(object, path, compress=compress)

def joblib_from_file(path: str):
    return joblib.load(path)

def protocol_writer(protocol):
    if protocol == 'pandas':
        return pandas_to_file
    elif protocol == 'pickle':
        return pickle_to_file
    elif protocol == 'joblib':
        return joblib_to_file
    else:
        raise ValueError(f'buffer_value protocol_writer Error: get {protocol}!')

def protocol_reader(protocol):
    if protocol == 'pandas':
        return pandas_from_file
    elif protocol == 'pickle':
        return pickle_from_file
    elif protocol == 'joblib':
        return joblib_from_file
    else:
        raise ValueError(f'buffer_value protocol_reader Error: get {protocol}!')
    
def protocol_postfix(protocol):
    if protocol == 'pandas':
        return '.csv'
    elif protocol == 'pickle':
        return '.pickle'
    elif protocol == 'joblib':
        return '.pkl'
    else:
        raise ValueError(f'buffer_value protocol_postfix Error: get {protocol}!')

def buffer_value(protocol, folder, disable=False):
    '''decorator for buffering temporary values in files
    protocol: [ 'pandas' | 'pickle' | 'joblib' ]
    folder: user defined path
    disable: disable reading from and writing to buffered files
    '''
    def decorator(func):
        def BufferWrapper(buffered_file, *args, **kwargs):
            logger.debug(f'[buffer_value] {folder} {buffered_file}')
            if not os.path.isdir(folder):
                os.mkdir(folder)
            
            fpath = os.path.join(os.path.join(folder,buffered_file+protocol_postfix(protocol)))
            
            def run_and_write():
                func_code = inspect.getsource(func)
                out = func(*args,**kwargs)
                if not disable:
                    writer = protocol_writer(protocol)
                    writer((func_code,out), fpath)
                    logger.debug(f'Writer object to {fpath}, @{protocol}, spent time {time()-t1:2.4f} sec')
                return out

            t1 = time()
            rerun_flag = False
            if not os.path.isfile(fpath) or disable:
                rerun_flag = True
            else:
                reader = protocol_reader(protocol)
                try:
                    prev_func_code, out = reader(fpath)
                    func_code = inspect.getsource(func)
                    if prev_func_code != func_code:
                        logger.debug(f'function_code change detected! {len(prev_func_code)} {len(func_code)} {prev_func_code != func_code}')
                        rerun_flag = True
                    else:
                        logger.debug(f'Read object from {fpath} in FUNC {func.__name__} {inspect.getfile(func)} #{inspect.currentframe().f_back.f_lineno}, @{protocol}, spent time {time()-t1:2.4f} sec')
                except Exception as e:
                    logger.debug(f'[buffer_value] {traceback.format_exc()}. Rerun!')
                    rerun_flag = True                
            if rerun_flag:
                out= run_and_write()
            return out
        return BufferWrapper
    return decorator


def timeit(func):
    def timer_wrapper(*args, **kwargs):
        t1 = time()
        out = func(*args, **kwargs)
        logger.info(f'Function {func.__name__} args: [{args}, {kwargs}] spent time {time()-t1:2.4f} sec')
        return out
    return timer_wrapper