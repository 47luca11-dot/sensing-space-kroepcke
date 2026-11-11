import json, sys
import numpy as np
from shapely.geometry import Polygon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT='C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
l=json.load(open(ROOT+'data/lod2_buildings.json',encoding='utf8'))
d=json.load(open(ROOT+'data/osm_features.json',encoding='utf8'))
osm={w['id']:w for w in d['ways']}
B=[b for b in l['buildings'] if b['id'].startswith('ID_6388af31')][0]
json.dump(B, open(ROOT+'pipeline/landmarks/anzeiger/anz_lod2.json','w'), indent=0)
fig,ax=plt.subplots(1,2,figsize=(18,10))
cm=plt.get_cmap('viridis')
for r in B['roofs']:
    p=np.array(r['ext']); x=p[:,0]; z=-p[:,1]; zz=p[:,2]
    c=cm((zz.mean()-60)/47)
    ax[0].fill(x,z,color=c,ec='k',lw=0.3)
for r in B['roofs']:
    p=np.array(r['ext'])
    if p[:,2].max()<100:
        ax[0].text(p[:,0].mean(), -p[:,1].mean(), f"{p[:,2].mean():.1f}", fontsize=6)
g=np.array(B['grounds'][0]['ext']); ax[0].plot(g[:,0],-g[:,1],'r-',lw=1)
o=np.array(osm[304133007]['pts']); ax[0].plot(o[:,0],o[:,1],'m--',lw=1)
for wid in (166991229,208147781):
    o=np.array(osm[wid]['pts']); ax[0].plot(o[:,0],o[:,1],'c--',lw=1)
ax[0].set_aspect('equal'); ax[0].invert_yaxis(); ax[0].set_title('top view game x(right) z(down=south)... z inverted so north up')
ax[0].grid(True)
# elevation from east: plot walls projected onto z (game), y
for w in B['walls']:
    p=np.array(w['ext'])
    ax[1].fill(-p[:,1], p[:,2], fc='none', ec='k', lw=0.3)
ax[1].set_aspect('equal'); ax[1].grid(True); ax[1].set_title('elevation seen from east: horizontal = dN (north to right?)')
plt.savefig(ROOT+'pipeline/landmarks/anzeiger/lod2_plot.png',dpi=110)
print(len(B['walls']),'walls', len(B['roofs']),'roofs')
print('ground ext', [[round(p[0],2),round(-p[1],2),round(p[2],2)] for p in B['grounds'][0]['ext']])
hs=sorted(set(round(p[2],1) for r in B['roofs'] for p in r['ext']))
print('roof heights', hs)
