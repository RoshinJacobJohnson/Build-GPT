import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size=32
block_size=8
max_iters=5000
eval_interval=300
learning_rate=1e-2
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
print(device)
with open('input.txt',"r", encoding="utf-8") as f:
    text=f.read()


chars= sorted(list(set(text)))
vocab_size=len(chars)

stoi= {e:i for i, e in enumerate(chars)}
itos=  {i:e for i, e in enumerate(chars)}

encoder=lambda s:[stoi[e] for e in s]
decoder=lambda s:''.join(itos[e] for e in s)

data=torch.tensor(encoder(text), dtype=torch.long)
n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size, (batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x=x.to(device)
    y=y.to(device)
    return x,y


class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
        
    def forward(self,idx,targets=None):
        logits=self.token_embedding_table(idx) #B,T,C
        if targets is None:
            loss=None
        else:
            b,t,c=logits.shape
        
            pred=logits.view(b*t,c)
            actual_op=targets.view(b*t)
            loss=F.cross_entropy(pred,actual_op)
        return logits,loss

    def generate(self,idx,new_token_len):
        
        for i in range(new_token_len):
            pred,loss=self(idx)
            op=pred[:,-1,:]
            prob=F.softmax(op,dim=-1)
            idx_n=torch.multinomial(prob,num_samples=1)
            idx=torch.cat([idx,idx_n],dim=1)
        return idx
    

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y=get_batch(split)
            _,loss=model(x,y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out


model=Bigram(vocab_size)
m=model.to(device)
optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)

for steps in range(max_iters):

    if(steps%eval_interval==0):
        losses=estimate_loss()
        print("step {}: train loss {} ".format(steps,losses['train']))
    xb,yb=get_batch("train")
    
    logits,loss=m(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())


idx=torch.zeros((1,1),dtype=torch.long,device=device)
print(decoder(m.generate(idx,new_token_len=100)[0].tolist()))