### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2024-01
### Usage: Please cite RNAsmol when you use this script


### calibration_curve

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# 读取目录
total_dirs = os.listdir('./calibration_curve/total/')
if '.DS_Store' in total_dirs:
    total_dirs.remove('.DS_Store')

#colors = [ '#C59E9B', '#cbd5e1']  
colors=["#b0b0b0","#8f8fbc"]
alphas = [ 0.9]  


for idx, dir in enumerate(total_dirs):
    
    properties = dir.split('_')
    dbcat = properties[2]
    files = os.listdir(f'./calibration_curve/total/{dir}/model/')
    files = [f'./calibration_curve/total/{dir}/model/{file}' for file in files]
    
    label_files = [file for file in files if 'test_label' in file]
    pred_files = [file for file in files if 'test_pred' in file]
    
    
    label_files_dict = {file.split('_epoch')[1].split('_test_label')[0]: file for file in label_files}
    pred_files_dict = {file.split('_epoch')[1].split('_test_pred')[0]: file for file in pred_files}

    fig, ax = plt.subplots(figsize=(6,6))
    common_bins = np.linspace(0, 1, 11)

    for epoch in label_files_dict.keys():
        if epoch in pred_files_dict:
            label_file = label_files_dict[epoch]
            pred_file = pred_files_dict[epoch]
            
            label = np.load(label_file)
            pred = np.load(pred_file)
            
            prob_true, prob_pred = calibration_curve(label, pred, n_bins=10)
            
            ax.plot(prob_pred, prob_true, marker='o', label=f'{dbcat} model (epoch {epoch})', color='black', alpha=0.5)
        
            ax_twin = ax.twinx()
            ax_twin.hist(pred, bins=common_bins, color=colors[idx % len(colors)], alpha=alphas[idx % len(alphas)], edgecolor='black')
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax_twin.set_ylabel('')

            ax.set_xticks([])
            ax.set_yticks([])
            ax_twin.set_yticks([])




plt.show()
# plt.savefig('./calibration_figures/' + dbcat + '_' + rnatype + '.png')
# fig, ax = plt.subplots()
# ax.plot(pred,label,marker='o')