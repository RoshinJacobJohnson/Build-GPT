import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size=32
block_size=8
max_iters=5000
eval_interval=300
learning_rate=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=32
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



class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)

    def forward(self,idx):
        B,T,C=idx.shape
        k=self.key(idx)
        v=self.value(idx)
        q=self.query(idx)
        wei=q@k.transpose(-2,-1)*C**-0.5
        s=torch.tril(torch.ones(T,T))
        wei=wei.masked_fill(s==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        op=wei@v
        return op


class MultiHeadAttention(nn.Module):

    def __init__(self,head_size,num_heads):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projs=nn.Linear(n_embd,n_embd)

    def forward(self,idx):
        op=torch.cat([h(idx) for h in self.heads],dim=-1)
        op=self.projs(op)
        return op

class FeedForwardNN(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net =nn.Sequential(nn.Linear(n_embd,4*n_embd),
                             nn.ReLU(),
                             nn.Linear(4*n_embd,n_embd)
                             )
    

    def forward(self,idx):
        return self.net(idx)


class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa_heads=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForwardNN(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)


    def forward(self,idx):
        x=idx+self.sa_heads(self.ln1(idx))
        x=x+self.ffwd(self.ln2(x))
        return x




class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.poition_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks= nn.Sequential(Block(n_embd,n_head=4),
                                   Block(n_embd,n_head=4),
                                   Block(n_embd,n_head=4),
                                   Block(n_embd,n_head=4),
                                   nn.LayerNorm(n_embd)
                                   )

        self.lm_head=nn.Linear(n_embd,vocab_size)

        
    def forward(self,idx,targets=None):
        B,T= idx.shape
        token_embd=self.token_embedding_table(idx) #B,T,C
        pos_embd= self.poition_embedding_table(torch.arange(T, device=device))
        x=(token_embd+pos_embd)
        x=self.blocks(x)
        logits=self.lm_head(x)
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
            pred,loss=self(idx[:,-block_size:])
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
        print("step {}: train loss {} , val loss {} ".format(steps,losses['train'],losses['val']))
    xb,yb=get_batch("train")
    
    logits,loss=m(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())


idx=torch.zeros((1,1),dtype=torch.long,device=device)
print(decoder(m.generate(idx,new_token_len=100)[0].tolist()))