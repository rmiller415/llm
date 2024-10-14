import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out,
        context_length, dropout,
        num_heads, qkv_bias=False,
    ):
        super().__init__()
        
        assert (d_out%num_heads==0), "d_out must be divisible by num_heads with no remainder."
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                        context_length,
                        context_length
                ),
                diagonal = 1
            )
        )
        
    def forward(self,x):
        #Initialize values and weight tensors
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #Reshape and resize weight tensors to inflate them (opposite of flattning)
        # Tensors start with shape (b,num_tokens, d_out) and end with a shape of
        # (b, num_tokens, num_heads, head_dim)
        # where head_dim = d_out/num_heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #Perform matrix multiplication to get attention scores:
        # This is analagous to having a tensor, T with shape (1,2,3,4) and performing 2 dot products:
        #  first = T[0,0,:,:]
        #  first_result = first@first.T
        #
        # and then performing:
        #
        #  second = T[0,1,:,:]
        # second_result = second@second.T
        attn_scores = queries@keys.transpose(2,3)

        #Mask attention scores
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        #Normalize attention scores
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        #Dotproduct to get result
        context_vec = (attn_weights@values).transpose(1,2)
        
        #Reshape outputs back to size (b,num_tokens,d_out)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        #Projection Layer - Not necessary, but often used in LLM architecture. I'm not sure why it's used.
        # this layer projects the results from one vector space into another vector space, by some linear mapping.
        # I'm not sure if this is to save disk space/memory, or if there is some optimal dimension for the vector
        # space to decode this back to text. Maybe to decrease processing speed/cycles used.
        context_vec = self.out_proj(context_vec)