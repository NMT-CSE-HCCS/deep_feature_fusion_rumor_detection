import torch.nn as nn
import torch
from ..module.bert import BERT
from ..module.cnn import CNN

class BERT_CNN(nn.Module):
    def __init__(self, feature_dim, nclass) -> None:
        super().__init__()
        self.transformer = BERT('bert-base-cased')
        self.cnn = CNN(feature_dim=feature_dim)
        self.classifier_in_size = self.transformer.out_dim + self.cnn.out_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.classifier_in_size, nclass)
        )

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids, propagation_tree = batch
        text_features = self.transformer(input_ids, attention_mask,token_type_ids)
        propagation_features = self.cnn(propagation_tree)
        deep_fusion_feautures = torch.cat((text_features, propagation_features), dim=1)
        x = self.classifier(deep_fusion_feautures)
        return x