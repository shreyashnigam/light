''' This is the main file for the transformer.'''
from typing import ForwardRef
import torch
from torch import nn
import torch.nn.functional as F
import random, math, sys

class Transfromer(nn.Module): 
    # TODO: Fields, Layers, Forward etc.
    def __int__(self, embedding_dim, heads, sequence_length, mask = False, hidden_layers = 4, dropout_value = 0.5, attention_type = 'default', pos_embedding = None):
        super().__init__()
        self.attention_block = None
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.mask = mask
        self.seq_len = sequence_length

        if attention_type == 'default':
            self.attention_block = SimpleSelfAttention(self.embedding_dim, self.heads, self.mask)
        
        self.norm_layer = nn.LayerNorm(self.embedding_dim)

        self.feed_forward_layer = nn.Sequential (
            nn.Linear(self.embedding, hidden_layers*self.embedding_dim),
            nn.ReLU(), # or try nn.LeakyReLU
            nn.Linear(hidden_layers*self.embedding_dim)
        )

        self.dropout_layer = nn.Dropout(dropout_value)

    def forward(self, x):
        attention_output = self.attention_block(x)
        x = self.norm_layer(x + attention_output)
        x = self.dropout_layer(x)
        feed_forward_output = self.feed_forward_layer(x)
        x = self.norm_layer(x + feed_forward_output)
        x = self.dropout_layer(x)
        return x


class SimpleSelfAttention(nn.Module):

    def __init__(self):
        pass
    
    def forward():
        pass
