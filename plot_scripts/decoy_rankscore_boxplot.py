### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-10
### Usage: Please cite RNAsmol when you use this script



#sns.boxplot(data=[RNAmigos_lst,RNAsmol_lst])
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from statannot import add_stat_annotation
from scipy.stats import mannwhitneyu
import pandas as pd
import pickle
import random 

matplotlib.rc('font', family='Arial')


sns.set_style('white', {'axes.linewidth': 1})
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
#plt.rcParams['xtick.major.width'] = 5
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

ax=plt.figure(figsize=(8,12))


with open("Figure2a_data/RNAmigos_tenfold_eu_ranklst.pkl", "rb") as f:
    RNAmigos_tenfold_eu_ranklst = pickle.load(f)

with open("Figure2a_data/RNAmigos2_onlypocket_tenfold_mixed_ranklst.pkl", "rb") as f:
    RNAmigos2_tenfold_mixed_ranklst = pickle.load(f)

with open("Figure2a_data/rDock_tenfold_ranklst.pkl", "rb") as f:
    rdock_tenfold_ranklst = pickle.load(f)

with open("Figure2a_data/RNAsmol_tenfold_ranklst.pkl", "rb") as f:
    RNAsmol_tenfold_ranklst = pickle.load(f)


df = pd.DataFrame({'RNAmigos': RNAmigos_tenfold_eu_ranklst,'RNAmigos2': RNAmigos2_tenfold_mixed_ranklst,'rDock': rdock_tenfold_ranklst, 'RNAsmol': RNAsmol_tenfold_ranklst})

colors=['#8098BB','#9DC4A9','#D88CAF','#A99DC4']
ax=sns.boxplot(data=df,palette=colors,fliersize=0,boxprops=dict(alpha=.6))


ax=sns.stripplot(data=df,color='grey',marker="o",size=5,alpha=0.8,linewidth=0.2)


box_pairs = [ ("RNAmigos", "RNAsmol"),("RNAmigos2", "RNAsmol"), ("rDock", "RNAsmol")]


add_stat_annotation(ax, data=df, box_pairs=box_pairs, test='Mann-Whitney-ls', text_format='star',linewidth=3,line_height=0.02, loc='inside', verbose=2)


#add_stat_annotation(ax, data=df, box_pairs=[("RNAmigos_eu","RNAmigos_ma","RNAmigos_co","RNAsmol")],test='Mann-Whitney-gt', text_format='star',line_height=0.05, loc='inside',verbose=2)

ax.grid(False)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
#plt.xlabel('Model',fontsize=20)
#plt.ylabel('Rank (top percentage)',fontsize=20)
#plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.yticks([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0],fontsize=0)
plt.xticks(fontsize=0,rotation=45)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)


plt.show()

#plt.ylim([0,1])
#add_stat_annotation(ax, data=df, box_pairs=[("RNAmigos", "RNAsmol")],test='Mann-Whitney-gt', text_format='star', loc='inside', line_height=0.05,verbose=2)
