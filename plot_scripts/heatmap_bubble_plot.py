### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-12
### Usage: Please cite RNAsmol when you use this script


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.cluster import hierarchy
from matplotlib.colors import Normalize

df = pd.read_csv('27sequence_metrics.csv')

ylabels = ['ROC-AUC', 'PR-AUC', 'ACC', 'SEN', 'SPE', 'F1', 'PRE', 'REC', 'MCC']
xlabels = ['NRAS', 'TERRA', 'EWSR1', 'AKTIP', 'Zika_NS5', 'Zika3PrimeUTR', 'FGFR', 'KLF6_wt', 'KLF6_mut', 'BCL_XL',
           'BCL_XL_SS', 'RRE2B', 'RRE2B_MeA', 'Pre_miR_21', 'Pre_miR_17', 'Pre_miR_31', 'HIV_SL3', 'HBV', 'Pro_wt',
           'Pro_mut', 'PreQ1', 'SAM_ll', 'ZTP', 'TPP', 'Glutamine_RS', 'MALAT1', 'ENE_A9']


data_matrix = df.values  


dendrogram_col = hierarchy.dendrogram(hierarchy.linkage(data_matrix, method='average'), no_plot=True)

data_matrix = data_matrix[dendrogram_col['leaves']]

fig, ax = plt.subplots(figsize=(17, 5))

R = data_matrix.T / data_matrix.T.max() / 2.5
x, y = np.meshgrid(np.arange(data_matrix.shape[0]), np.arange(data_matrix.shape[1]))
circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]

norm = Normalize(vmin=0.7, vmax=1)
col = PatchCollection(circles, array=data_matrix.T.flatten(), cmap="PuOr", alpha=0.5,norm=norm)
ax.add_collection(col)


xlabels_dendrogram=[]
for ind in dendrogram_col['ivl']:
    xlabels_dendrogram.append(xlabels[int(ind)])

#ax.set(xticks=np.arange(27), yticks=np.arange(9),xticklabels=xlabels_dendrogram, yticklabels=ylabels)
ax.set(xticks=np.arange(27), yticks=np.arange(9),xticklabels=[], yticklabels=[])


ax.set_xticks(np.arange(27+1)-0.5, minor=True)
ax.set_yticks(np.arange(9+1)-0.5,minor=True)
ax.grid(which='minor')


plt.xticks(rotation=90)
'''
#hierarchy.dendrogram(hierarchy.linkage(data_matrix, method='average'), ax=ax, above_threshold_color='gray', orientation='bottom')
axins = ax.inset_axes([0, -0.93, 1, 0.5])
hierarchy.dendrogram(hierarchy.linkage(data_matrix, method='average'), ax=axins, orientation='bottom')



#axins.set_xticks([])
axins.set_xticklabels([])
axins.set_axis_off()
'''

#fig.colorbar(col,ticks=[0.7,0.8,0.9,1])
fig.colorbar(col, ticks=[0.7, 0.8, 0.9, 1], pad=0.02)


plt.show()

################################################################################################################################


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


xlabels = ['NRAS', 'TERRA', 'EWSR1', 'AKTIP', 'Zika_NS5', 'Zika3PrimeUTR', 'FGFR', 'KLF6_wt', 'KLF6_mut', 'BCL_XL',
           'BCL_XL_SS', 'RRE2B', 'RRE2B_MeA', 'Pre_miR_21', 'Pre_miR_17', 'Pre_miR_31', 'HIV_SL3', 'HBV', 'Pro_wt',
           'Pro_mut', 'PreQ1', 'SAM_ll', 'ZTP', 'TPP', 'Glutamine_RS', 'MALAT1', 'ENE_A9']

secs = ['G-quadruplex'] * 6 + ['Hairpin'] * 14 + ['Pseudoknot'] * 3 + ['Three-way junction'] * 2 + ['Triple helix'] * 2


xlabel_to_sec = dict(zip(xlabels, secs))


new_order = ['Pro_mut', 'ZTP', 'TPP', 'BCL_XL', 'Pre_miR_21', 'Pre_miR_17', 'FGFR', 'Pre_miR_31', 'ENE_A9', 
             'Zika3PrimeUTR', 'SAM_ll', 'HIV_SL3', 'Glutamine_RS', 'EWSR1', 'AKTIP', 'Pro_wt', 'PreQ1', 
             'RRE2B_MeA', 'BCL_XL_SS', 'Zika_NS5', 'NRAS', 'RRE2B', 'KLF6_wt', 'KLF6_mut', 'HBV', 'TERRA', 'MALAT1']


cmap = plt.cm.coolwarm
sec_colors = {
    'G-quadruplex': cmap(0.1, alpha=0.5),
    'Hairpin': cmap(0.3, alpha=0.5),
    'Pseudoknot': cmap(0.5, alpha=0.5),
    'Three-way junction': cmap(0.7, alpha=0.5),
    'Triple helix': cmap(0.9, alpha=0.5)
}


color_bar = [sec_colors[xlabel_to_sec[xlabel]] for xlabel in new_order]

fig, ax = plt.subplots(figsize=(20, 0.5))

cmap = mcolors.ListedColormap(color_bar)
bounds = range(len(new_order) + 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

cb = ax.imshow([list(range(len(new_order)))], cmap=cmap, norm=norm, aspect='auto')
ax.set_yticks([])  
ax.set_xticks([])
ax.set_xticklabels([], rotation=90, ha='center')

ax.axis('off')

#plt.colorbar(cb, orientation='horizontal', fraction=0.046, pad=0.04)

plt.show()

