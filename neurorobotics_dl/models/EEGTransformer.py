import torch
from torch import nn
import math
    
class EEGTransformer(nn.Module):
    def __init__(self,
                 d_model = 16, 
                 embedding_dim = 8, 
                 seq_len = 512,
                 encoder_layer_args={'nhead':4},
                 encoder_args=dict()):
        super().__init__()
        print('porcodio')
        self.pos_encoder = PositionalEncoding(d_model,max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model,**encoder_layer_args)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer,**encoder_args) 
        self.fc = nn.Linear(d_model,embedding_dim)
    def forward(self,x):
        h = x.squeeze(1).permute(0,2,1)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        h = h.mean(axis=1)
        h = self.fc(h)
        return h

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2)
        return self.dropout(x)