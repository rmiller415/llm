import torch

class GPTDataset(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt,allowed_special={'<|endoftext|>'})
        
        for i in range(0,len(token_ids)-max_length, stride):
            input_ = token_ids[i:i+max_length]
            target_ = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_))
            self.target_ids.append(torch.tensor(target_))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
