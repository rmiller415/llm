import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        '''
        Looks like tokenizer is instantiated before being passed to this class.
        '''
        
        self.input_ids = []
        self.target_ids = []
        
        #Get token id's
        self.token_ids = tokenizer.encode(txt)
        
        for i in range(0,len(self.token_ids)-max_length,stride):
            input_chunk = self.token_ids[i:i+max_length]
            target_chunk = self.token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids, self.target_ids
    
    
def create_dataloader(
    txt,
    batch_size=25,
    max_length=255,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(txt,tokenizer,max_length,stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader

def create_dataloader(
        txt, batch_size=4, max_length=256, stride=128, 
        shuffle=True, drop_last=True, num_workers=0
        ):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


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
        
        GPT_CONFIG_124M = {
            'vocab_size':50257,
            'context_length':1024,
            'emb_dim':768,
            'n_heads':12,
            'n_layers':12,
            'drop_rate':0.1,
            'qkv_bias':False,
        }
        