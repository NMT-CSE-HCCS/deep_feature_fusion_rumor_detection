from src.models.lightning_template.base import BaseModule
class ModelSelection():
    model_list = ['RoBERTa_CNN','BERT_CNN']
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def default() -> str:
        return ModelSelection.choices()[0]

    @staticmethod
    def choices():
        return ModelSelection.model_list

    @staticmethod
    def getModel(name, args):
        if name not in ModelSelection.choices():
            raise ValueError(f'No model named {name}!')
        
        if name == 'RoBERTa_CNN':
            from .model.roberta_cnn import RoBERTa_CNN
            model = RoBERTa_CNN(args.feature_dim, args.nclass)
        if name == 'BERT_CNN':
            from .model.bert_cnn import BERT_CNN
            model = BERT_CNN(args.feature_dim, args.nclass)

        return BaseModule(args, args.nclass, model)