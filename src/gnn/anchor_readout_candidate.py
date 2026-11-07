"""Development-only readout preserving co-observed team-incidence nodes.

The added channels pool nodes targeted by at least two relation types. They
are structural masks, not identified contested opponents or outcome labels.
The ordinary team means are retained and the new head columns start at zero.
"""
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from .mcar_model import MCARFootballGNN


class AnchorReadoutCandidate(MCARFootballGNN):
    def __init__(self, **kwargs):
        if kwargs.get("use_current_score",False):
            raise ValueError("This isolated candidate is for the complete-match development protocol")
        super().__init__(**kwargs)
        for head in [self.cls_head,self.reg_head]:
            old=head[0]
            with torch.random.fork_rng(devices=[]):
                new=nn.Linear(old.in_features+2*self.hidden_dim,old.out_features)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:,:old.in_features].copy_(old.weight)
                new.bias.copy_(old.bias)
            head[0]=new

    def forward(self,data):
        x_dict=self.encode(data)
        pooled,incidence=[],[]
        for nt in self.NODE_TYPES:
            x=x_dict[nt]
            batch=getattr(data[nt],"batch",None)
            if batch is None:
                batch=torch.zeros(len(x),dtype=torch.long,device=x.device)
            mean=global_mean_pool(x,batch)
            available=torch.zeros((len(x),3),dtype=torch.bool,device=x.device)
            for src,rel,dst in self.EDGE_TYPES:
                if dst!=nt:
                    continue
                ei=data[(src,rel,dst)].edge_index
                if ei.numel():
                    available[ei[1],("pass","adversarial","turnover").index(rel)]=True
            mask=(available.sum(-1)>=2).to(x.dtype).unsqueeze(-1)
            total=global_add_pool(x*mask,batch,size=mean.size(0))
            count=global_add_pool(mask,batch,size=mean.size(0)).clamp_min(1.)
            pooled.append(mean);incidence.append(total/count)
        embedding=torch.cat(pooled+incidence,dim=-1)
        return self.cls_head(embedding),self.reg_head(embedding)
