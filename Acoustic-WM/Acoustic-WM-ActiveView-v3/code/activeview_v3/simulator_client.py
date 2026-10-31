"""Geometry sampling and committed acquisition across the two Python environments."""
from .runtime import ROOT,UTILITY,output_path
import subprocess,json,os

class Simulator:
 def __init__(self,log='logs/simulator-server.log',server='scripts/renderer_server.py'):
  self.log=open(output_path(log),'a',buffering=1)
  self.process=subprocess.Popen([str(UTILITY/'environments/audiolidar/bin/python'),'-B','-u',str(ROOT/server)],
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
    env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OPENBLAS_NUM_THREADS='1'))
 def request(self,**command):
  self.process.stdin.write(json.dumps(command)+'\n');self.process.stdin.flush()
  for line in self.process.stdout:
   if not line.startswith('V3RPC '):self.log.write(line);continue
   reply=json.loads(line[6:])
   if not reply['ok']:raise RuntimeError(reply['error'])
   return reply['result']
  raise RuntimeError('Renderer exited: '+str(self.process.poll()))
 def reference(self,sample):return self.request(command='reference',sample=sample)['listener']
 def geometry(self,scene,listener):return self.request(command='geometry',scene=scene,listener=listener)
 def sample(self,rng,count,centre=None,scale=None):
  return self.request(command='sample',seed=int(rng.integers(2**32)),count=count,
    centre=None if centre is None else centre.tolist(),scale=None if scale is None else scale.tolist())
 def acquire(self,pose,output,save_audio=False):
  return self.request(command='acquire',pose=pose,output=str(output),save_audio=save_audio)
 def close(self):
  try:
   if self.process.poll() is None:self.request(command='close');self.process.wait(timeout=30)
  finally:
   if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=10)
   self.log.close()
