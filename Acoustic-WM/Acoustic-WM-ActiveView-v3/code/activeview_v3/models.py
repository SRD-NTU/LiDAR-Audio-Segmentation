"""Real evidence calibration and a stochastic conditional observation world model."""
import torch,math
from torch import nn
from .runtime import ROOT

class EvidenceAdapter(nn.Module):
 def __init__(self,dimension=128,neural=False):
  super().__init__();self.config=dict(dimension=dimension,neural=neural)
  self.log_variance_scale=nn.Parameter(torch.zeros(dimension))
  self.log_noise=nn.Parameter(torch.zeros(dimension))
  self.mean_offset=nn.Parameter(torch.zeros(dimension))
  self.log_prior_scale=nn.Parameter(torch.zeros(()))
  # The PCA coordinates mix frequency bands. Keep the fitted transform available
  # so measured, inactive bands can be removed before evaluating real evidence.
  if dimension==128:
   import numpy as np
   with np.load(ROOT/'checkpoints/embedding.npz') as z:
    self.register_buffer('embedding_mean',torch.from_numpy(z['mean'].copy()),persistent=False)
    self.register_buffer('embedding_basis',torch.from_numpy(z['basis'].copy()),persistent=False)
    self.register_buffer('embedding_scale',torch.from_numpy(z['scale'].copy()),persistent=False)
   self.register_buffer('packing_factor',torch.tensor([1.]*4+[2.**.5]*12),persistent=False)
  if neural:
   self.correction=nn.Sequential(nn.Linear(128+2*dimension+24,48),nn.LayerNorm(48),nn.GELU(),nn.Linear(48,1))
   nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
 def prior(self,batch):
  scores=batch['base'].float()*self.log_prior_scale.exp().clamp(.25,4)
  return scores.masked_fill(~batch['valid'],-1e4)
 def distribution(self,batch):
  mean=batch['mu'].float()+.25*self.mean_offset.tanh()
  variance=batch['var'].float()*self.log_variance_scale.clamp(-4,4).exp()+self.log_noise.clamp(-5,5).exp()
  return mean,variance.clamp(.005,100)
 def component_log_likelihood(self,batch,mean,var,observation,measured=False):
  """Gaussian component score; measured audio uses its 48-band observation mask.

  The diagonal covariance after projection is an approximation. The projected
  residual is exactly zero when only an inactive band in the measured packed
  covariance changes. A complete mask recovers the original likelihood.
  """
  if measured and 'raw_target' in batch and 'band_reliability' in batch:
   if not hasattr(self,'embedding_basis'):raise ValueError('Masked likelihood requires the fitted 128-D embedding')
   b,k,h,d=mean.shape
   active=(batch['band_reliability'].float()>0).repeat_interleave(16,-1).float()
   raw=batch['raw_target'].float().reshape(b,768)*self.packing_factor.repeat(48)
   reconstructed=self.embedding_mean+(mean*self.embedding_scale)@self.embedding_basis.T
   residual=((raw[:,None,None]-reconstructed)*active[:,None,None])@self.embedding_basis/self.embedding_scale
   retained=(active@self.embedding_basis.square()).clamp(0,1)
   # The projected masked noise variance scales with the retained basis mass.
   # The log(retained) term is shared by all components, so omit it.
   return -.5*((residual.square()/(var*retained[:,None,None].clamp_min(1e-8)))+
               retained[:,None,None]*var.log()).sum(-1)
  return -.5*(((observation[...,None,None,:]-mean).square()/var)+var.log()).sum(-1)
 def scores(self,batch,observation=None,scale=1.,measured=None):
  mean,var=self.distribution(batch)
  measured=(observation is None) if measured is None else measured
  observation=batch['target'].float() if observation is None else observation
  extra=observation.ndim==3
  if extra:mean=mean[:,None];var=var[:,None];sites=batch['sites'].float()[:,None]
  else:sites=batch['sites'].float()
  likelihood=self.component_log_likelihood(batch,mean,var,observation,measured=measured and not extra)
  ll=torch.logsumexp(sites+likelihood,-1)
  if self.config['neural']:
   component_mu,component_var=self.distribution(batch);site=batch['sites'].float().exp()
   average=(component_mu*site[...,None]).sum(2)
   spread=((component_var+(component_mu-average[:,:,None]).square())*site[...,None]).sum(2)
   latent=batch['latent'].float();residual=batch['residual'].float();support=batch['support'].float()/10
   action=batch['actions'].float().clone();action[:,:3]/=3
   action=action[:,None].expand(-1,average.shape[1],-1)
   if extra:
    n=observation.shape[1]
    average=average[:,None];spread=spread[:,None]
    latent=latent[:,None].expand(-1,n,-1,-1);residual=residual[:,None].expand(-1,n,-1,-1)
    support=support[:,None].expand(-1,n,-1,-1);action=action[:,None].expand(-1,n,-1,-1)
   if measured and not extra and 'raw_target' in batch and 'band_reliability' in batch:
    active=batch['band_reliability'].float().gt(0).repeat_interleave(16,-1).float()
    raw=batch['raw_target'].float().reshape(len(average),768)*self.packing_factor.repeat(48)
    reconstruction=self.embedding_mean+(average*self.embedding_scale)@self.embedding_basis.T
    residual=((raw[:,None]-reconstruction)*active[:,None])@self.embedding_basis/self.embedding_scale
    retained=(active@self.embedding_basis.square()).clamp_min(1e-8)
    error=(residual/(spread*retained[:,None]).sqrt()).clamp(-4,4)
   else:error=((observation[...,None,:]-average)/spread.sqrt()).clamp(-4,4)
   context=torch.cat([latent,residual.clamp(-4,4),support,action],-1)
   # Subtract the zero-residual output: a learned context-only offset cannot
   # silently replace the reference selector with a new object classifier.
   correction=self.correction(torch.cat([error,context],-1)).squeeze(-1)
   neutral=self.correction(torch.cat([torch.zeros_like(error),context],-1)).squeeze(-1)
   ll=ll+8*torch.tanh(correction-neutral)
  prior=self.prior(batch)
  if extra:prior=prior[:,None]
  audible=batch.get('audible')
  if audible is not None and observation.ndim==2:ll=torch.where(audible[:,None],ll,torch.zeros_like(ll))
  return prior+scale*ll

class ObservationWorld(nn.Module):
 def __init__(self,dimension=128,hidden=48,correction=True):
  super().__init__();self.config=dict(dimension=dimension,hidden=hidden,correction=correction)
  # Frozen memory, reference acoustic residual, candidate site, action, prior gap, site belief.
  self.context=nn.Sequential(nn.Linear(128+dimension+3+12+2,hidden),nn.LayerNorm(hidden),nn.GELU(),
                             nn.Linear(hidden,hidden),nn.GELU())
  self.out=nn.Linear(hidden,dimension*2);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,batch,evidence):
  mu,var=evidence.distribution(batch)
  if not self.config['correction']:return mu,var
  b,k,h,d=mu.shape
  actions=batch['actions'].float().clone();actions[:,:3]/=3
  gap=((batch['base'].float()-batch['base'].float().max(-1,keepdim=True).values)/10).clamp(-5,0)
  x=torch.cat([batch['latent'].float()[:,:,None].expand(-1,-1,h,-1),
   batch['residual'].float()[:,:,None].expand(-1,-1,h,-1),
   batch['support'].float().reshape(b,k,h,3)/10,
   actions[:,None,None].expand(-1,k,h,-1),gap[:,:,None,None].expand(-1,-1,h,-1),
   batch['sites'].float()[...,None]/5],-1)
  raw=self.out(self.context(x));delta,noise=raw.chunk(2,-1)
  # Conservative residual; the reference likelihood is a frozen physical prior.
  return mu+.5*torch.tanh(delta),var*torch.exp(.7*torch.tanh(noise))

def target_distribution(batch):
 target=batch['iou'].float().clamp_min(0).pow(4)*batch['valid']
 return target/target.sum(-1,keepdim=True).clamp_min(1e-9),target.sum(-1)>0

def evidence_loss(model,batch):
 scores=model.scores(batch);target,usable=target_distribution(batch)
 ce=-(target*scores.log_softmax(-1)).sum(-1)
 return (ce*usable).sum()/usable.sum().clamp_min(1)

def world_loss(world,evidence,batch,posterior_weight=.2,scale=1.):
 mean,var=world(batch,evidence);e=batch['target'].float()
 log_density=evidence.component_log_likelihood(batch,mean,var,e,measured=True)
 log_object=torch.logsumexp(batch['sites'].float()+log_density,-1)
 target,usable=target_distribution(batch);usable=usable&batch['audible']
 nll=-(target*log_object).sum(-1)/mean.shape[-1]
 # Infer a component from the real TRAIN observation. Sample its posterior predictive
 # outcome and require the frozen real-audio updater to preserve the observed belief.
 with torch.no_grad():
  teacher=evidence.scores(batch,scale=scale).softmax(-1)
  responsibilities=(batch['sites'].float()+log_density).softmax(-1)
  probability=(target[:,:,None]*responsibilities).flatten(1)
  probability=probability+1e-8*batch['valid'][:,:,None].expand_as(responsibilities).flatten(1)
  component=torch.multinomial(probability,1).squeeze(1)
 rr=torch.arange(len(mean),device=mean.device);chosen_mean=mean.flatten(1,2)[rr,component]
 chosen_var=var.flatten(1,2)[rr,component]
 imagined=chosen_mean+chosen_var.sqrt()*torch.randn_like(chosen_mean)
 imagined_log=evidence.scores(batch,imagined,scale=scale).log_softmax(-1)
 kl=(teacher*(teacher.clamp_min(1e-9).log()-imagined_log)).sum(-1)
 prior_mu,prior_var=evidence.distribution(batch)
 penalty=((mean-prior_mu).square()/prior_var).mean((1,2,3))
 loss=((nll+posterior_weight*kl+.02*penalty)*usable).sum()/usable.sum().clamp_min(1)
 return loss,dict(nll=float((nll*usable).sum().detach()/usable.sum().clamp_min(1)),
                  posterior_kl=float((kl*usable).sum().detach()/usable.sum().clamp_min(1)))

@torch.no_grad()
def imagined_utility(world,evidence,batch,draws=16,scale=1.,seed=179):
 """Only pre-acquisition fields are consumed; samples are model-generated audio features."""
 clean={k:v for k,v in batch.items() if k not in ['target','audible','iou','null_iou']}
 mean,var=world(clean,evidence);b,k,h,d=mean.shape
 prior=evidence.prior(clean).softmax(-1)
 component_probability=(prior[:,:,None]*clean['sites'].float().exp()).flatten(1)
 generator=torch.Generator(device=mean.device).manual_seed(seed)
 # Common random quantiles/noise across poses reduce Monte Carlo action-ranking noise.
 quantile=torch.rand(draws,device=mean.device,generator=generator)
 cumulative=component_probability.cumsum(-1).contiguous()
 component=torch.searchsorted(cumulative,quantile[None].expand(b,-1).contiguous()).clamp_max(k*h-1)
 index=component[...,None].expand(-1,-1,d)
 selected_mean=mean.flatten(1,2).gather(1,index);selected_var=var.flatten(1,2).gather(1,index)
 noise=torch.randn(draws,d,device=mean.device,generator=generator)
 imagined=selected_mean+selected_var.sqrt()*noise[None]
 posterior=evidence.scores(clean,imagined,scale).softmax(-1)
 chosen=posterior.argmax(-1)
 overlap=clean['overlap'].float()
 selected_overlap=overlap.gather(1,chosen[...,None].expand(-1,-1,k))
 future_quality=(posterior*selected_overlap).sum(-1).mean(-1)
 initial=prior.argmax(-1)
 current_quality=(prior*overlap[torch.arange(b,device=mean.device),initial]).sum(-1)
 return future_quality-current_quality

class DirectValue(nn.Module):
 def __init__(self,dimension=128,hidden=64,full_information=False):
  super().__init__();self.config=dict(dimension=dimension,hidden=hidden,full_information=full_information)
  if full_information:
   # Same pre-acquisition object/site fields as ObservationWorld. Pooling makes
   # the regressor independent of the number and ordering of candidate masks.
   self.site_encoder=nn.Sequential(nn.Linear(3*dimension+145,hidden),nn.LayerNorm(hidden),nn.GELU(),
                                   nn.Linear(hidden,hidden),nn.GELU())
   self.network=nn.Sequential(nn.Linear(3*hidden+14,hidden),nn.LayerNorm(hidden),nn.GELU(),
                              nn.Dropout(.2),nn.Linear(hidden,1))
  else:
   # Retain compatibility for the archived direct baseline checkpoint.
   self.network=nn.Sequential(nn.Linear(128+2*dimension+14,hidden),nn.LayerNorm(hidden),nn.GELU(),
                              nn.Dropout(.2),nn.Linear(hidden,1))
 def forward(self,batch,evidence):
  p=evidence.prior(batch).softmax(-1);mu,var=evidence.distribution(batch)
  if self.config['full_information']:
   b,k,h,d=mu.shape
   site=batch['sites'].float().softmax(-1)
   action=batch['actions'].float().clone();action[:,:3]/=3
   gap=((batch['base'].float()-batch['base'].float().max(-1,keepdim=True).values)/10).clamp(-5,0)
   features=torch.cat([mu,var.log().clamp(-5,5),batch['support'].float().reshape(b,k,h,3)/10,
      batch['sites'].float()[...,None],batch['latent'].float()[:,:,None].expand(-1,-1,h,-1),
      batch['residual'].float()[:,:,None].expand(-1,-1,h,-1),gap[:,:,None,None].expand(-1,-1,h,-1),
      action[:,None,None].expand(-1,k,h,-1)],-1)
   encoded=self.site_encoder(features)
   objects=(encoded*site[...,None]).sum(2)
   average=(objects*p[...,None]).sum(1)
   spread=((objects-average[:,None]).square()*p[...,None]).sum(1)
   maximum=objects.masked_fill(~batch['valid'][...,None],-1e4).amax(1)
   entropy=-(p*p.clamp_min(1e-9).log()).sum(-1)
   return self.network(torch.cat([average,spread,maximum,action,entropy[:,None]/5,
                                  batch['distance'].float()[:,None]/3],-1)).squeeze(-1)
  site=batch['sites'].float().exp();mean=(mu*site[...,None]).sum(2)
  variance=((var+(mu-mean[:,:,None]).square())*site[...,None]).sum(2)
  average=(mean*p[...,None]).sum(1)
  between=((mean-average[:,None]).square()*p[...,None]).sum(1)
  within=(variance*p[...,None]).sum(1)
  latent=(batch['latent'].float()*p[...,None]).sum(1)
  entropy=-(p*p.clamp_min(1e-9).log()).sum(-1)
  action=batch['actions'].float().clone();action[:,:3]/=3
  features=torch.cat([latent,between.clamp(0,10),within.log().clamp(-5,5),action,
                      entropy[:,None]/5,batch['distance'].float()[:,None]/3],-1)
  return self.network(features).squeeze(-1)
