import torch
import torch.nn as nn
import math


class MiniBERT(nn.Module):
    def __init__(self,vocab_size,max_length=128,embedding_dim=64,num_heads=8):

        super().__init__()
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.token_embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.position_embedding = nn.Embedding(max_length,embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(nn.Linear(embedding_dim, 256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256, embedding_dim))
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)


    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        token_embeddings = self.token_embedding(input_ids)
        positions = torch.arange(seq_len,device=input_ids.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)
        K = K.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)
        V = V.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q,K.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores.masked_fill(attention_mask == 0,-1e9)
        attention_weights = torch.softmax(attention_scores,dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attention_output = torch.matmul(attention_weights,V)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size,seq_len,self.embedding_dim)
        attention_output = self.out(attention_output)

        x = self.norm1(x + attention_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        return logits