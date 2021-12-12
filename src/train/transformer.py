''' This is the main file for the transformer.'''
from typing import ForwardRef
import torch
from torch import nn
import torch.nn.functional as F
import random, math, sys
import math
from torch.autograd import Variable

# Simple Encoder-Decoder transformer with simple multi head self-attention
class Transformer(nn.Module): 
    def __int__(self, encoder_input_dims, decoder_input_dims, model_input_dims, heads, N = 1, dropout_val = 0.1):
        super().__init__()
        self.encoder_block = Encoder(encoder_input_dims, model_input_dims, heads, N, dropout_val)
        self.decoder_block = Decoder(decoder_input_dims, model_input_dims, heads, N, dropout_val)
        self.output_layer = nn.Linear(model_input_dims, decoder_input_dims)

    def forward(self, x):
        x_encoder = self.encoder_block(x)
        x_decoder = self.dencoder_block(x, x_encoder)
        x_output = self.output_layer(x_decoder)
        return x_output


class SimpleSelfAttention(nn.Module):

    def __init__(self, dim, heads, mask = None, dropout_value = 0.1):
        super.__init__()

        self.dim = dim
        self.heads = heads
        self.mask = mask;

        self.query = nn.Linear(self.dim self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(dropout_value)
        self.output_layer = nn.Linear(self.dim, self.dim)

    def attention(self, query, key, value, d_k, mask = None, dropout = None):
        result = query @ key.transpose(-2, -1)
        result /= math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            result = result.masked_fill(mask==0, -math.inf)
        
        result = F.softmax(result, dim=-1)

        if dropout is not None:
            result = dropout(result)
        
        return (result @ value)

    def forward(self, q, k, v, mask=None):
        size = q.size(0);
        d_k = self.dim//self.heads

        curr_key = self.key(k).view(size, -1, self.heads, d_k)
        curr_query = self.query(q).view(size, -1, self.heads, d_k)
        curr_value = self.value(v).view(size, -1, self.heads, d_k)

        curr_value = curr_value.transpose(1,2)
        curr_query = curr_query.transpose(1,2)
        curr_key = curr_key.transpose(1,2)

        output = self.attention(curr_query, curr_key, curr_value, d_k, mask, self.dropout)

        output = output.transpose(1,2).contiguous()
        output = output.view(size, -1, self.dim)

        return self.output_layer(output)

class WordEmbedder(nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x):
        return self.embedding_layer(x)

# referenced the most commonly used positional encoder from the internet
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class FeedForward(nn.Module):

    def __init__(self, input_dim, hidden_layers, dropout_val = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_layers)
        self.dropout = nn.Dropout(dropout_val)
        self.layer2 = nn.Linear(hidden_layers, input_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        return self.layer2(x)

class Encoder(nn.Module):

    def __init__(self, vocab_size, input_dims, heads, N = 1, mask = None, dropout_val = 0.1):
        super().__init__()
        self.word_embed = WordEmbedder(vocab_size, input_dims)
        self.pos_embed = PositionalEncoder(input_dims, dropout = dropout_val)
        # can hve multiple encoding layer blocks, make change later
        self.encoding_layer = EncoderLayer(input_dims, heads, dropout_val=dropout_val)
        self.layer_norm = nn.LayerNorm(input_dims)
    
    def forward(self, x):
        x = self.word_embed(x)
        x = self.pos_embed(x)
        x = self.encoding_layer(x)
        return self.layer_norm(x)

class Decoder(nn.Module):

    def __init__(self, vocab_size, input_dims, heads, N = 1, mask = None, dropout_val = 0.1):
        super().__init__()
        self.word_embed = WordEmbedder(vocab_size, input_dims)
        self.pos_embed = PositionalEncoder(input_dims, dropout = dropout_val)
        # can hve multiple encoding layer blocks, make change later
        self.decoding_layer = DecoderLayer(input_dims, heads, dropout_val=dropout_val)
        self.layer_norm = nn.LayerNorm(input_dims)

    def forward(self, x, x_encoding):
        x = self.word_embed(x)
        x = self.pos_embed(x)
        x = self.decoding_layer(x, x_encoding)
        return self.layer_norm(x)

class EncoderLayer(nn.Module):

    def __init__(self, input_dims, heads, hidden_layers = 1024, dropout_val = 0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dims)
        self.layer_norm2 = nn.LayerNorm(input_dims)
        self.attention_layer = SimpleSelfAttention(input_dims, heads)
        self.feed_forward_layer = FeedForward(input_dims, hidden_layers)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        x_attention = self.attention_layer(x_norm, x_norm, x_norm)
        x_attention = self.dropout(x_attention)
        x = x + x_attention

        x_norm = self.layer_norm2(x)
        x_output = self.feed_forward_layer(x_norm)
        x_output = self.dropout(x_output)

        return x + x_output

class DecoderLayer(nn.Module):

    def __init__(self, input_dims, heads, hidden_layers = 1024, dropout_val = 0.1):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(input_dims)
        self.layer_norm2 = nn.LayerNorm(input_dims)
        self.layer_norm3 = nn.LayerNorm(input_dims)
        self.attention_layer1 = SimpleSelfAttention(input_dims, heads)
        self.attention_layer2 = SimpleSelfAttention(input_dims, heads)
        self.feed_forward_layer = FeedForward(input_dims, hidden_layers)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x, x_encoder, mask = None):
        x_norm = self.layer_norm1(x)
        x_attention1 = self.attention_layer1(x_norm, x_norm, x_norm)
        x_attention1 = self.dropout(x_attention1)
        x = x + x_attention1

        x_norm = self.layer_norm2(x)
        x_attention2 = self.attention_layer2(x_norm, x_encoder, x_encoder)
        x_attention2 = self.dropout(x_attention2)
        x = x + x_attention2

        x_norm = self.layer_norm3(x)
        x_ff = self.feed_forward_layer(x_norm)
        x_ff = self.dropout(x_ff)
        x = x + x_ff
        return x












