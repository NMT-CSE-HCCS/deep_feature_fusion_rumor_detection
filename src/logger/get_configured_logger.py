
import sys
import io
import logging
import logging.config
import os
import time
from pathlib import Path

import yaml


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

class myFileHandler(logging.FileHandler):
    def __init__(self, root, exp_name='', subdirectory='', mode='a', fn='', encoding='utf-8', delay=False):
        path = os.path.join(root, exp_name, subdirectory)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if fn == '':
            fn = f"{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}.log"
        
        filename = os.path.join(path, fn)
        super().__init__(filename, mode, encoding, delay)

class errorFileHandler(logging.FileHandler):
    def __init__(self, root, exp_name='', subdirectory='', mode='a', fn='', encoding='utf-8', delay=False):
        path = os.path.join(root, exp_name, subdirectory)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if fn == '':
            fn = f"{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}.err"
        
        filename = os.path.join(path, fn)
        super().__init__(filename, mode, encoding, delay)

def get_yaml_config(logger_config_path):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),logger_config_path)
    with open(config_path, 'r', encoding='utf-8') as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)
    return config_dict

def get_logger_seperate_config(debug, exp_name = '', subdirectory='', filename = ''):
    logger_config_path = 'logger_config_seperate.yaml'
    config_dict = get_yaml_config(logger_config_path)

    for fh, suffix in zip(['file', 'errorlog'], ['.log','.err']):
        config_dict['handlers'][fh]['fn'] = filename + suffix
        config_dict['handlers'][fh]['exp_name'] = exp_name
        config_dict['handlers'][fh]['subdirectory'] = subdirectory
    
    if debug:
        config_dict['loggers']['src']['level'] = 'DEBUG'
        config_dict['root']['level'] = 'DEBUG'
    logging.config.dictConfig(config_dict)

    return logging.getLogger()

def get_root_logger_default_config(debug, exp_name = '', subdirectory='', filename = ''):
    logger_config_path = 'logger_config.yaml'
    config_dict = get_yaml_config(logger_config_path)

    config_dict['handlers']['file']['fn'] = filename
    config_dict['handlers']['file']['exp_name'] = exp_name
    config_dict['handlers']['file']['subdirectory'] = subdirectory
    
    if debug:
        config_dict['loggers']['src']['level'] = 'DEBUG'
        config_dict['root']['level'] = 'DEBUG'

    logging.config.dictConfig(config_dict)

    return logging.getLogger()
    
def get_console_logger(debug=True):
    logger_config_path = 'logger_config_console.yaml'
    config_dict = get_yaml_config(logger_config_path)
    if debug:
        config_dict['loggers']['__main__']['level'] = 'DEBUG'
        config_dict['loggers']['src']['level'] = 'DEBUG'
        config_dict['root']['level'] = 'DEBUG'
    logging.config.dictConfig(config_dict)

    return logging.getLogger()

def get_logger_simple():
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    return logger

