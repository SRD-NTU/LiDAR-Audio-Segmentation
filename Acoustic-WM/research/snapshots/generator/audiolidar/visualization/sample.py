"""Headless geometry and binaural signal inspection."""
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audiolidar.utils.io import read_json
from audiolidar.geometry.frames import sensor_rotation


def preview(sample: Path, output: Path = None, rgb: Path = None):
    sample=Path(sample); m=read_json(sample/'metadata.json')
    with np.load(sample/'lidar.npz') as p:
        xyz=p['xyz_world']; mask=p['sound_mask'].astype(bool)
    wave,rate=sf.read(sample/'audio.wav',always_2d=True)
    if rgb:
        fig=plt.figure(figsize=(18,15),layout='constrained')
        rgb_axis=fig.add_subplot(331)
        rgb_axis.imshow(plt.imread(rgb));rgb_axis.axis('off');rgb_axis.set_title('Textured HM3D RGB view; sounding object overlay in red')
        scene_index,detail_index,wave_index,left_index,right_index,top_index=332,333,334,335,336,337
    else:
        fig=plt.figure(figsize=(18,10),layout='constrained')
        scene_index,detail_index,wave_index,left_index,right_index,top_index=231,232,233,234,235,236
    ax=fig.add_subplot(scene_index,projection='3d')
    bg=xyz[~mask][::max(1,int((~mask).sum())//15000)]
    # Display X/Z as floor plane and Y as height.
    ax.scatter(bg[:,0],bg[:,2],bg[:,1],s=.3,c='#a0a0a0',alpha=.3)
    q=xyz[mask];ax.scatter(q[:,0],q[:,2],q[:,1],s=3,c='#e63946',label='Sounding instance')
    source=np.asarray(m['source']['position_world']);listener=np.asarray(m['listener']['position_world'])
    ax.scatter(*source[[0,2,1]],s=90,marker='*',c='orange',label='Emitter')
    ax.scatter(*listener[[0,2,1]],s=40,c='blue',label='Listener')
    forward=sensor_rotation(m['listener']['yaw_rad'])[:,0]
    ax.quiver(*listener[[0,2,1]],*forward[[0,2,1]],length=.7,color='blue')
    ax.set(xlabel='world X (m)',ylabel='world Z (m)',zlabel='world Y (m)');ax.legend(fontsize=8)
    ax.set_title(f"{m['scene_id']} / instance {m['source']['instance_id']} / {m['difficulty']['setting']}")
    ax=fig.add_subplot(detail_index,projection='3d')
    center=np.asarray(m['source']['center']);near=xyz[np.linalg.norm(xyz-center,axis=1)<1.]
    ax.scatter(near[:,0],near[:,2],near[:,1],s=1,c='#a0a0a0',alpha=.15)
    ax.scatter(q[:,0],q[:,2],q[:,1],s=9,c='#e63946',label=f'{len(q)} source returns')
    ax.scatter(*source[[0,2,1]],s=70,marker='*',c='orange');ax.legend(fontsize=8)
    ax.set(xlabel='world X (m)',ylabel='world Z (m)',zlabel='world Y (m)',title=f"Source detail: {m['source']['category']}")
    ax=fig.add_subplot(top_index)
    ax.scatter(bg[:,0],bg[:,2],s=.3,c='#aaa',alpha=.25)
    ax.scatter(q[:,0],q[:,2],s=6,c='#e63946')
    ax.scatter(*source[[0,2]],marker='*',s=100,c='orange');ax.scatter(*listener[[0,2]],s=40,c='blue')
    ax.arrow(listener[0],listener[2],forward[0]*.6,forward[2]*.6,color='blue',width=.03)
    ax.plot([listener[0],source[0]],[listener[2],source[2]],'--',color='orange',lw=1)
    ax.set(xlabel='world X (m)',ylabel='world Z (m)',title='Top view / receiver heading');ax.set_aspect('equal',adjustable='datalim')
    ax=fig.add_subplot(wave_index)
    time=np.arange(len(wave))/rate
    for i in range(wave.shape[1]):ax.plot(time,wave[:,i],lw=.4,label=f'Channel {i}')
    ax.set(xlabel='Time (s)',ylabel='Amplitude',title=f"Azimuth {m['relative_source']['azimuth_deg']:.1f}°, {m['relative_source']['distance_m']:.2f} m")
    ax.legend()
    for i,index in enumerate([left_index,right_index]):
        ax=fig.add_subplot(index); ax.specgram(wave[:,i],NFFT=512,Fs=rate,noverlap=384,cmap='magma')
        ax.set(ylim=(0,12000),xlabel='Time (s)',ylabel='Frequency (Hz)',title=['Left','Right'][i])
    fig.savefig(output or sample/'preview.png',dpi=150);plt.close(fig)


def preview_parts(sample: Path, output_dir: Path):
    """Write diagnostic panels as independent PNG files."""
    sample=Path(sample); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    m=read_json(sample/'metadata.json')
    with np.load(sample/'lidar.npz') as p:
        xyz=p['xyz_world']; mask=p['sound_mask'].astype(bool)
    wave,rate=sf.read(sample/'audio.wav',always_2d=True)
    bg=xyz[~mask][::max(1,int((~mask).sum())//15000)]
    q=xyz[mask]; source=np.asarray(m['source']['position_world']); listener=np.asarray(m['listener']['position_world'])
    forward=sensor_rotation(m['listener']['yaw_rad'])[:,0]
    def save(fig,name):
        fig.savefig(output_dir/name,dpi=150); plt.close(fig)
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection='3d')
    ax.scatter(bg[:,0],bg[:,2],bg[:,1],s=.3,c='#a0a0a0',alpha=.3); ax.scatter(q[:,0],q[:,2],q[:,1],s=3,c='#e63946',label='Sounding instance')
    ax.scatter(*source[[0,2,1]],s=90,marker='*',c='orange',label='Emitter'); ax.scatter(*listener[[0,2,1]],s=40,c='blue',label='Listener'); ax.quiver(*listener[[0,2,1]],*forward[[0,2,1]],length=.7,color='blue')
    ax.set(xlabel='world X (m)',ylabel='world Z (m)',zlabel='world Y (m)',title=f"{m['scene_id']} / instance {m['source']['instance_id']}"); ax.legend(fontsize=8); save(fig,'lidar_scene.png')
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection='3d'); center=np.asarray(m['source']['center']); near=xyz[np.linalg.norm(xyz-center,axis=1)<1.]
    ax.scatter(near[:,0],near[:,2],near[:,1],s=1,c='#a0a0a0',alpha=.15); ax.scatter(q[:,0],q[:,2],q[:,1],s=9,c='#e63946',label=f'{len(q)} source returns'); ax.scatter(*source[[0,2,1]],s=70,marker='*',c='orange'); ax.legend(fontsize=8); ax.set(xlabel='world X (m)',ylabel='world Z (m)',zlabel='world Y (m)',title=f"Source detail: {m['source']['category']}"); save(fig,'source_detail.png')
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(bg[:,0],bg[:,2],s=.3,c='#aaa',alpha=.25); ax.scatter(q[:,0],q[:,2],s=6,c='#e63946'); ax.scatter(*source[[0,2]],marker='*',s=100,c='orange'); ax.scatter(*listener[[0,2]],s=40,c='blue'); ax.arrow(listener[0],listener[2],forward[0]*.6,forward[2]*.6,color='blue',width=.03); ax.plot([listener[0],source[0]],[listener[2],source[2]],'--',color='orange',lw=1); ax.set(xlabel='world X (m)',ylabel='world Z (m)',title='Top view / receiver heading'); ax.set_aspect('equal',adjustable='datalim'); save(fig,'top_view.png')
    time=np.arange(len(wave))/rate; fig,ax=plt.subplots(figsize=(8,4))
    for i in range(wave.shape[1]): ax.plot(time,wave[:,i],lw=.4,label=f'Channel {i}')
    ax.set(xlabel='Time (s)',ylabel='Amplitude',title=f"Azimuth {m['relative_source']['azimuth_deg']:.1f}°, {m['relative_source']['distance_m']:.2f} m"); ax.legend(); save(fig,'waveform.png')
    for i,name in enumerate(('spectrogram_left.png','spectrogram_right.png')):
        fig,ax=plt.subplots(figsize=(8,4)); ax.specgram(wave[:,i],NFFT=512,Fs=rate,noverlap=384,cmap='magma'); ax.set(ylim=(0,12000),xlabel='Time (s)',ylabel='Frequency (Hz)',title=['Left','Right'][i]); save(fig,name)
