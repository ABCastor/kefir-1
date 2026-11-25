"""Experimental FP32 causal pairwise recurrence; qualification required."""
import math
import torch
from kefir.layers.gdn2 import _check_recurrence_inputs

class PairwiseGDN2Backend:
 def __init__(self,chunk_size=32):
  if chunk_size<1:raise ValueError('positive chunk size required')
  self.chunk_size=chunk_size
 def __call__(self,q,k,v,log_decay,erase_gate,write_gate,initial_state=None):
  batch,sequence,heads,value_dim=_check_recurrence_inputs(q,k,v,log_decay,erase_gate,write_gate,initial_state)
  with torch.autocast(q.device.type,enabled=False):
   qf,kf,vf,ld,erase,write=[x.float() for x in (q,k,v,log_decay,erase_gate,write_gate)]
   state=initial_state.float() if initial_state is not None else torch.zeros(batch,heads,q.shape[-1],value_dim,device=q.device)
   outputs=[];scale=1/math.sqrt(q.shape[-1])
   for start in range(0,sequence,self.chunk_size):
    end=min(start+self.chunk_size,sequence);size=end-start
    qc=qf[:,start:end].transpose(1,2)*scale;kc=kf[:,start:end].transpose(1,2)
    ec=(erase[:,start:end]*kf[:,start:end]).transpose(1,2);z=(write[:,start:end]*vf[:,start:end]).transpose(1,2)
    g=ld[:,start:end].transpose(1,2).cumsum(2)
    causal=torch.ones(size,size,device=q.device,dtype=torch.bool).tril()
    delta=(g.unsqueeze(-2)-g.unsqueeze(-3)).masked_fill(~causal[...,None],0)
    decayed_key=delta.exp()*causal[...,None]*kc.unsqueeze(-3)
    a=(decayed_key*ec.unsqueeze(-2)).sum(-1).tril(-1)
    rhs=z-(ec*g.exp())@state
    identity=torch.eye(size,device=q.device,dtype=torch.float32)
    correction=torch.linalg.solve_triangular(identity+a,rhs,upper=False,unitriangular=True)
    qk=(decayed_key*qc.unsqueeze(-2)).sum(-1)
    outputs.append(((qc*g.exp())@state+qk@correction).transpose(1,2))
    final=g[:,:,-1:]
    tail_key=(final-g).exp()*kc
    state=final.squeeze(2).exp().unsqueeze(-1)*state+tail_key.transpose(-1,-2)@correction
   output=torch.cat(outputs,dim=1).to(q.dtype) if outputs else q.new_empty((batch,0,heads,value_dim))
   if not torch.isfinite(output).all() or not torch.isfinite(state).all():raise ValueError('nonfinite pairwise recurrence')
   return output,state
