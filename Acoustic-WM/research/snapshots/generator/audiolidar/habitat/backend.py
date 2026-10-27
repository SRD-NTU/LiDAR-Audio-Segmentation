"""Pinned Habitat AudioSensor backend. No analytic or mono fallback."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from audiolidar.geometry.frames import quaternion_xyzw


class Backend:
    def __init__(self, scene: dict, cfg: dict, diagnostic: bool = False, mesh=None):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Habitat rendering must run in a Slurm GPU allocation')
        import quaternion  # Official workaround for audio branch ABI initialization.
        import habitat_sim as hs
        self.hs, self.scene, self.cfg = hs, scene, cfg
        sc = hs.SimulatorConfiguration()
        sc.scene_id = scene['visual']
        if not diagnostic:
            if mesh is None:raise ValueError('Original semantic mesh required for acoustic adapter')
            from audiolidar.habitat.acoustic_mesh import prepare
            sc.scene_id=str(prepare(mesh,scene,cfg))
        sc.use_semantic_textures=diagnostic
        sc.scene_dataset_config_file = cfg['dataset_config']
        sc.enable_physics = False
        sc.load_semantic_mesh = True
        sc.gpu_device_id = 0
        ac = hs.agent.AgentConfiguration()
        # Required to force semantic mesh loading and independently audit ray labels.
        specs = []
        kinds=[('semantic',hs.SensorType.SEMANTIC),('depth',hs.SensorType.DEPTH)]
        if diagnostic:kinds.append(('texture_loader',hs.SensorType.COLOR))
        for name, kind in kinds:
            spec = hs.CameraSensorSpec()
            spec.uuid=name; spec.sensor_type=kind
            spec.resolution=cfg.get('visualization',{}).get('camera_resolution',[128,128]) if diagnostic else [128,128]
            spec.position=[0.,0.,0.]; spec.hfov=90
            specs.append(spec)
        ac.sensor_specifications=specs
        self.sim=hs.Simulator(hs.Configuration(sc,[ac]))
        self.sim.pathfinder.load_nav_mesh(scene['navmesh'])
        self.sim.pathfinder.seed(cfg['seed'])
        if not self.sim.pathfinder.is_loaded or not self.sim.semantic_scene.objects:
            self.close(); raise RuntimeError('Missing navmesh or semantic objects')
        audio=hs.AudioSensorSpec(); audio.uuid='audio_sensor'; audio.position=[0.,0.,0.]
        audio.enableMaterials=cfg['soundspaces']['enableMaterials']
        mode=cfg['audio']['mode']
        # ``Mono`` models an omnidirectional point microphone.  A caller can
        # render it repeatedly at calibrated offsets to form a physical array;
        # this is fundamentally different from four ambisonic coefficients.
        layout={'mono':('Mono',1),'binaural':('Binaural',2),'ambisonics':('Ambisonics',4)}
        if mode not in layout:
            raise ValueError('Unsupported audio mode: '+str(mode))
        name,count=layout[mode]
        self.channel_count=count
        audio.channelLayout.type=getattr(hs.sensor.RLRAudioPropagationChannelLayoutType,name)
        audio.channelLayout.channelCount=count
        for k,v in cfg['soundspaces']['acoustics'].items():
            if not hasattr(audio.acousticsConfig,k):raise ValueError('Unsupported acoustic parameter: '+k)
            setattr(audio.acousticsConfig,k,v)
        audio.acousticsConfig.sampleRate=cfg['audio']['sample_rate']
        self.sim.add_sensor(audio)
        self.audio=self.sim.get_agent(0)._sensors['audio_sensor']
        materials=Path(cfg['soundspaces']['materials_json'])
        if not materials.is_absolute(): materials=Path(cfg['config_path']).parent.parent/materials
        if not materials.is_file():raise FileNotFoundError(materials)
        self.audio.setAudioMaterialsJSON(str(materials))

    def pose(self, position, yaw):
        import quaternion
        state=self.hs.AgentState()
        state.position=np.asarray(position,dtype=float)
        q=quaternion_xyzw(yaw)
        state.rotation=np.quaternion(q[3],q[0],q[1],q[2])
        state.sensor_states={}
        self.sim.get_agent(0).set_state(state,True)

    def render(self, source, position, yaw) -> np.ndarray:
        self.pose(position,yaw)
        self.audio.setAudioSourceTransform(np.asarray(source,dtype=float))
        result=np.asarray(self.sim.get_sensor_observations()['audio_sensor'],dtype='float32')
        expected=self.channel_count
        if result.ndim!=2 or result.shape[0]!=expected or not np.isfinite(result).all() or not np.any(result):
            raise RuntimeError('Invalid SoundSpaces impulse response: '
                               f'shape={result.shape}, finite={bool(np.isfinite(result).all())}, '
                               f'nonzero={int(np.count_nonzero(result))}')
        return result

    def summary(self) -> dict:
        objects=[]
        for o in self.sim.semantic_scene.objects:
            if o is None:continue
            objects.append({'habitat_id':o.id,'semantic_id':o.semantic_id,
                            'category':o.category.name(), 'category_id':o.category.index(),
                            'center':np.asarray(o.aabb.center).tolist(),
                            'size':np.asarray(o.aabb.sizes).tolist()})
        return dict(scene_id=self.scene['scene_id'],objects=objects,object_count=len(objects),
                    category_count=len({o['category'] for o in objects}),
                    navigable_area=float(self.sim.pathfinder.navigable_area))

    def close(self):
        self.sim.close()
