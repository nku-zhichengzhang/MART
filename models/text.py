import torch
from transformers import BertTokenizer
from torchtext.legacy import data
from torchtext.legacy.data import Dataset, Example
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import tqdm

class BERT_MODEL(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, text, returnembed=False):
        output = self.bert(text)
        if returnembed:
            res = {'embeddings': output['last_hidden_state'], 'cls': output['pooler_output']}
        else:
            res = output['pooler_output']
        return res

def get_text_model(useLarge=False):

    if useLarge == 'large':
        bert = BertModel.from_pretrained('bert-large-uncased')
    else:
        bert = BertModel.from_pretrained('bert-base-uncased')

    return BERT_MODEL(bert)