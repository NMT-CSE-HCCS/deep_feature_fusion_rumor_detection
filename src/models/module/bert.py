import torch.nn as nn
from transformers import BertModel

"""
A wrapper for BERT model.
See HuggingFace transformers for more details.
https://github.com/huggingface/transformers
"""
class BERT(nn.Module):
    def __init__(self, pretrain_model_name='bert-base-cased') -> None:
        super().__init__()
        self.model = BertModel.from_pretrained(
            pretrain_model_name, output_attentions=True
        )
        self.out_dim = self.model.config.hidden_size
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        return_dict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        h = return_dict['last_hidden_state']
        h_cls = h[:, 0]

        return h_cls
