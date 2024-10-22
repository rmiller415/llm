import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            d_in, d_out, 
            context_length, 
            dropout, 
            num_heads, 
            qkv_bias=False,
    ):
        super().__init__()
        
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        
        self.query_weights = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key_weights = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value_weights = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        
        b, num_tokens, d_in = x.shape
        
        keys = self.key_weights(x)
        queries = self.query_weights(x)
        values = self.value_weights(x)
        
        #Change dimensions of matrices for multihead attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        #Transpose tensors for inner product
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        #Compute attention scores using dot products for each head
        attn_scores = queries @ keys.transpose(2,3)
        
        #Tensor of positions to be masked True = mask from attention mechanism
        #False= do not mask from attention mechanism.
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        
        #Use mask to fill attention scores with -infinity
        attn_scores.masked_fill_(mask_bool,-torch.inf)
        
        #Calculated normalized attention weights from attention scores
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        #Calculate output tensor
        output = (attn_weights @ values).tranpose(1,2)
        
        #Combin heads and unroll tensor
        output = output.contiguous().view(b, num_tokens, self.d_out)
        
        #Optional output projection layer
        output = self.out_proj(output)
        
        return output
        
        
        
        