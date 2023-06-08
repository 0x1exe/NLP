import torch
from transformers import BertTokenizer
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

import pandas as pd

import torch.nn.functional as F
import math,copy,re

from sklearn.model_selection import train_test_split

=========================== WORD EMBEDDINGS + POSITIONAL ENCODING ===========================
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        out = self.embed(x)
        return out
      
class PositionalEncoding(nn.Module):
  def __init__(self,seq_len,embed_dim):
    super().__init__()
    self.position=torch.zeros(seq_len,embed_dim)
    for pos in range(seq_len):
      for i in range(0,embed_dim,2):
        self.position[pos,i]=math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
        self.position[pos,(i+1)]=math.cos(pos/(10000 ** ((2*(i+1))/self.embed_dim)))
    self.position=self.position.unsqueeze(0)
    self.register_buffer('pe',self.position)

  def forward(self,x):
    x=x*math.sqrt(embed_dim)
    seq_len=x.size(1)
    x=x+torch.autograd.Variable(self.position[:,:seq_len],requires_grad=False)

    return x

 =========================== SELF-ATTENTION =========================== 
  
class SelfAttention(nn.Module):
  def __init__(self,embed_dim,heads):
    super().__init__()

    self.embed_dim=embed_dim
    self.heads=heads
    
    self.h1=embed_dim // heads

    self.query = nn.Linear(self.h1 ,self.h1 ,bias=False)  
    self.key = nn.Linear(self.h1 ,self.h1, bias=False)
    self.value= nn.Linear(self.h1 ,self.h1, bias=False)
    self.out = nn.Linear(self.heads*self.h1,self.embed_dim) 

  def forward(self,queries,keys,values,mask=None):

    batch_size = keys.size(0)
    seq_length = keys.size(1)

    seq_length_query = queries.size(1)
    keys = keys.view(batch_size, seq_length, self.heads, self.h1)  
    queries = queries.view(batch_size, seq_length_query, self.heads, self.h1)
    values = values.view(batch_size, seq_length, self.heads, self.h1)

    k = self.key(keys)       
    q = self.query(queries)   
    v = self.value(values)

    q = q.permute(0,2,1,3)  
    k = k.permute(0,2,3,1)  
    v = v.permute(0,2,1,3)

    product = torch.matmul(q, k)

    if mask is not None:
      product = product.masked_fill(mask == 0, float("-1e20"))
    
    product = product / math.sqrt(self.h1)
    scores = F.softmax(product, dim=-1)

    scores = torch.matmul(scores, v)
    concat = scores.permute(0,2,1,3).contiguous().view(batch_size, seq_length_query, self.h1*self.heads) 
        
    output = self.out(concat) 
       
    return output

=========================== ENCODER BLOCKS ===========================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, heads=8):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_dim, heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):
        attention_out = self.attention(key,query,value)  
        attention_residual_out = attention_out + value 
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) 

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        return norm2_out
  
class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, heads=8):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, heads) for i in range(num_layers)])
    
    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out,out,out)

        return out
      
=========================== DECODER BLOCKS ===========================

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, heads=8):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, heads)
    
    def forward(self, key, query, x,mask):
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention + x))
        
        out = self.transformer_block(key, query, value)

        
        return out
      
class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, heads=8):
        super(TransformerDecoder, self).__init__()

        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, heads=8) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        x = self.word_embedding(x)  
        x = self.position_embedding(x) 
        x = self.dropout(x)
     
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask) 

        out = F.softmax(self.fc_out(x),dim=1)

        return out
 
=========================== FINAL TRANSFORMER ===========================

class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length,device,num_layers=2, expansion_factor=4, heads=8):
        super(Transformer, self).__init__()

        self.target_vocab_size = target_vocab_size
        self.device=device

        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, heads=heads)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, heads=heads)
        
    
    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask    

    def decode(self,src,trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size,seq_len = src.shape[0],src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len): #10
            out = self.decoder(out,enc_out,trg_mask) #bs x seq_len x vocab_dim
            # taking the last token
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg).to(self.device)
        enc_out = self.encoder(src)
   
        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs
