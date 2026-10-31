"""Fine-band source-invariant spatial observations for predictive world learning."""
import numpy as np
from scipy.signal import stft
from .spatial_features import FREQUENCIES,KEEP,NFFT,RATE,pack

EDGES=np.linspace(150,3000.01,49)
POOL48=np.stack([((FREQUENCIES>=a)&(FREQUENCIES<b)).astype(np.float32) for a,b in zip(EDGES[:-1],EDGES[1:])])
POOL48/=POOL48.sum(1,keepdims=True)

def fine_features(rir,wave):
    multiple=int(np.ceil(rir.shape[1]/NFFT));h=np.fft.rfft(rir,n=multiple*NFFT,axis=-1)[:,::multiple][:,KEEP].T
    q=h[:,:,None]*h[:,None,:].conj();q/=np.maximum((abs(h)**2).sum(-1)[:,None,None],1e-20)
    transfer=np.einsum('bf,fmn->bmn',POOL48,q)
    observed,weight=covariance48_from_wave(wave)
    return pack(transfer),observed,weight

def covariance48_from_wave(wave):
    wave=np.asarray(wave)
    if wave.ndim!=2 or wave.shape[1]!=4 or len(wave)<NFFT or not np.isfinite(wave).all():raise ValueError('Expected finite 44100 Hz audio with four channels')
    _,_,x=stft(wave.T,fs=RATE,nperseg=NFFT,noverlap=NFFT//2,boundary=None)
    x=x[:,KEEP];c=np.einsum('mft,nft->fmn',x,x.conj(),optimize=True)/x.shape[-1]
    power=np.trace(c,axis1=-2,axis2=-1).real;c/=np.maximum(power[:,None,None],1e-20)
    observed=np.einsum('bf,fmn->bmn',POOL48,c);energy=POOL48@power
    weight=(energy>max(energy.max()*1e-4,1e-20)).astype(np.float32);weight/=max(weight.sum(),1)
    return pack(observed),weight

def geometric_tokens(xyz,features):
    """Nearest/farthest measured point per angular bin; never invented means."""
    radius=np.linalg.norm(xyz,axis=1)
    az=np.floor((np.arctan2(xyz[:,1],xyz[:,0])+np.pi)/(2*np.pi)*48).astype(int)%48
    el=np.clip(((np.arctan2(xyz[:,2],np.linalg.norm(xyz[:,:2],axis=1))+np.pi/2)/np.pi*4).astype(int),0,3)
    ids=az*4+el;order=np.lexsort((radius,ids));sorted_ids=ids[order]
    first=np.r_[True,sorted_ids[1:]!=sorted_ids[:-1]];last=np.r_[sorted_ids[1:]!=sorted_ids[:-1],True]
    chosen=np.r_[order[first],order[last]];chosen=np.unique(chosen)
    points=np.zeros((384,3),np.float32);feat=np.zeros((384,256),np.float16);valid=np.zeros(384,bool)
    n=len(chosen);points[:n]=xyz[chosen];feat[:n]=features[chosen];valid[:n]=True
    return points,feat,valid
