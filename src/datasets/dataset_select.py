import importlib
import inspect
import os


def import_function_or_class(module,method_name):
    module = importlib.import_module(f'{module}')
    method = getattr(module, method_name)
    return method

def update_dataset_setting(args, get_dataset_setting):
    setting = args.dataset_setting
    param_setting = get_dataset_setting(setting)
    vars(args).update(param_setting)

def filter_dict(func, kwarg_dict):
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return filtered_dict

def init_class_from_namespace(class_, namespace):
    common_kwargs = filter_dict(class_, vars(namespace))
    return class_(**common_kwargs)

def scan_dataset_list():
    s = os.path.dirname(os.path.realpath(__file__))
    modules = os.listdir(os.path.join(s,'model'))
    fns = [ f[:-3] for f in modules if not f.endswith('__init__.py') and f.endswith('.py')]
    return fns


class DatasetSelection():
    dataset_list = ['twitter15', 'twitter16']
    def __init__(self) -> None:
        pass

    @staticmethod
    def default() -> str:
        return DatasetSelection.choices()[0]

    @staticmethod
    def choices():
        return DatasetSelection.dataset_list

    @staticmethod
    def getDataset(name, args=None):
        if name not in DatasetSelection.choices():
            raise ValueError(f'No dataset named {name}')
        if name == 'twitter15':
            from .TwitterDataset.Twitter15 import Twitter15
            dl = Twitter15(args.dataset_root, args.tokenizer,args.max_token,args.max_tree,args.batch_size, args.nfold,args.determ, args.num_workers, args.seed)
        if name == 'twitter16':
            from .TwitterDataset.Twitter16 import Twitter16
            dl = Twitter16(args.dataset_root, args.tokenizer,args.max_token,args.max_tree,args.batch_size, args.nfold,args.determ, args.num_workers, args.seed)
        return dl
