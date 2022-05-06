import torch.nn as nn
from transformers import RobertaModel

"""
A wrapper for RoBERTa model.
See HuggingFace transformers for more details.
https://github.com/huggingface/transformers
"""
class RoBERTa(nn.Module):
    def __init__(self, pretrain_model_name='roberta-base') -> None:
        super().__init__()
        self.model = RobertaModel.from_pretrained(
            pretrain_model_name, output_attentions=True
        )
        self.out_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        return_dict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        h = return_dict['last_hidden_state']
        h_cls = h[:, 0]

        return h_cls

