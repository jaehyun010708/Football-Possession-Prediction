"""Permutation-invariant player attention -> PMA -> BiGRU -> four logits."""
import torch
from torch import nn


class SAB(nn.Module):
    def __init__(self,d,h,dropout):
        super().__init__()
        self.att=nn.MultiheadAttention(d,h,batch_first=True,dropout=dropout)
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,d))
    def forward(self,x,pad):
        z=self.att(x,x,x,key_padding_mask=pad,need_weights=False)[0]
        x=self.norm1(x+z)
        return self.norm2(x+self.ff(x))


class PlayerAttentionGRU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        c=cfg['model'];d=int(c['embed_dim']);h=int(c['heads']);dr=float(c['dropout']);k=int(c['seeds'])
        self.in_proj=nn.Linear(int(c['feature_dim']),d)
        self.layers=nn.ModuleList([SAB(d,h,dr) for _ in range(int(c['attention_layers']))])
        self.seeds=nn.Parameter(torch.randn(1,k,d)*0.1)
        self.pma=nn.MultiheadAttention(d,h,batch_first=True,dropout=dr)
        self.pool_norm=nn.LayerNorm(k*d)
        self.frame=nn.Sequential(nn.Linear(k*d,int(c['frame_dim'])),nn.GELU())
        self.empty_frame=nn.Parameter(torch.zeros(int(c['frame_dim'])))
        hidden=int(c['hidden_size']);layers=int(c['gru_layers'])
        self.gru=nn.GRU(int(c['frame_dim']),hidden,layers,batch_first=True,bidirectional=True,dropout=dr if layers>1 else 0.)
        self.head=nn.Linear(hidden*2,4)
    def encode(self,feat,mask):
        B,T,N,F=feat.shape
        x=self.in_proj(feat.reshape(B*T,N,F))
        good=mask.reshape(B*T,N).bool()
        empty=~good.any(-1)
        # Temporary safe key avoids NaNs; result replaced by empty_frame afterwards.
        safe=good.clone(); safe[empty,0]=True
        pad=~safe
        for layer in self.layers:x=layer(x,pad)
        z=self.pma(self.seeds.expand(len(x),-1,-1),x,x,key_padding_mask=pad,need_weights=False)[0]
        out=self.frame(self.pool_norm(z.flatten(1)))
        out=torch.where(empty[:,None],self.empty_frame[None,:],out)
        return out.reshape(B,T,-1)
    def forward(self,feat,mask):
        t,_=self.gru(self.encode(feat,mask))
        # logits: alive, home_given_alive, boundary_AD, boundary_HA
        return self.head(t)


def probabilities(logits):
    import torch
    p=torch.sigmoid(logits)
    alive,home=p[...,0],p[...,1]
    return torch.stack((alive*home,alive*(1-home),1-alive),-1),p[...,2:4]
