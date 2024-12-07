### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-12
### Usage: Please cite RNAsmol when you use this script


import os
import numpy as np
import torch
from torch_geometric.data import Batch
import pandas as pd
from tqdm import tqdm
from model_MCNN_multi_diffusion_bilinearattention import MGraphDTA, MCNN_GCN
from dataset import *
from utils import *

import torch.nn.functional as F
import pickle


class GradAAM():
   def __init__(self, model, module):
       self.model = model
       module.register_forward_hook(self.save_hook)
       self.target_feat = None
   def save_hook(self, md, fin, fout):
       #self.target_feat = fout.x  
       self.target_feat=fout
   def __call__(self, data):
       self.model.eval()
       a,b,c,d=self.model(data)
       #print(self.model(data))
       output=F.softmax(d,dim=-1)
       #print(output)\n",
       output, indices=torch.max(output,dim=-1)
       output[indices==0]=1 - output[indices==0]
       #print(output)
       pred_prob=output.view(-1)
       #output = d.view(-1)
       #class_output=output[1]
       grad = torch.autograd.grad(pred_prob, self.target_feat,allow_unused=True)[0]
       #print(grad)
       #print(grad.shape)
       if grad is None:
           grad=torch.zeros_like(self.target_feat)
       channel_weight = torch.mean(grad, dim=(0,2), keepdim=True)
       channel_weight = normalize(channel_weight)
       weighted_feat = self.target_feat * channel_weight
       cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
       cam = normalize(cam)
       #print(cam)
       return output.detach().cpu().numpy(), cam

def main():
   device = torch.device('cpu')


   fpath = os.path.join('RNAsmol/visualization', '4casestudy')
   test_df = pd.read_csv(os.path.join(fpath, 'raw', 'data_test.csv'))
   test_set = GNNDataset(fpath, train=False)
   model = MCNN_GCN(3, 25 + 1, embedding_size=96, filter_num=32, out_dim=2,ban_heads=2).to(device)
   #model = MCNN_GCN(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
   params=model.state_dict()
   for k,v in params.items():
       print(k)
   #print(\"************\")
   
    
    
   model.load_state_dict(torch.load('save/pdbrnaprotein_proteinbinder_posneg11_rnafold_ppr_ban/model/epoch-31, loss-0.1603, val_loss-0.2719, test_loss-0.2871, test_acc-0.8824, test_sen-0.9425, test_spe-0.8193,test_pre-0.8454, test_rec-0.9425, test_f1-0.8913, test_rocauc-0.9648, test_prauc-0.9699, test_mcc-0.7693.pt',map_location=torch.device('cpu')))
   
   #gradcam = GradAAM(model, module=model.protein_encoder.block_list[0].inc[0])
   gradcam=GradAAM(model,module=model.protein_encoder.embed)
   #bottom = cm.get_cmap('Blues_r', 256)
   #top = cm.get_cmap('Oranges', 256)
   #newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
   #newcmp = ListedColormap(newcolors, name='OrangeBlue')
   seq_list = list(test_df['target_sequence'].unique())
   progress_bar = tqdm(total=len(seq_list))
   seq_atom_att_dict = {}
   for idx in range(len(test_set)):
       seq = test_df.iloc[idx]['target_sequence']
       if len(seq_list) == 0:
           break
       if seq in seq_list:
           seq_list.remove(seq)
       else:
           continue
       data = Batch.from_data_list([test_set[idx]])
       data = data.to(device)
       #print(gradcam(data))
       _, atom_att = gradcam(data)
       #print(len(seq))
       #print(atom_att[0])
       seq_atom_att_dict[seq] = atom_att[0][:len(seq)].tolist()

   with open("pdb_proteinbinder_casestudy_rnaseq_ntgradaam.pkl", 'wb') as f:
        pickle.dump(seq_atom_att_dict,f)

       #atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) for idx in range(len(atom_att))])
       #radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
       #img = clourMol()

       seq_ele=list(str(seq))
       mat=logomaker.sequence_to_matrix(str(seq))
       fig, ax = plt.subplots(figsize=(20, 2))
       logo=logomaker.Logo(mat,color_scheme='gray',ax=ax)
       color_lst=[]
       for idxx in range(len(atom_att[0])):
           #print(newcmp(atom_att[0][idx])[:3])
           color_lst.append(newcmp(atom_att[0][idxx])[:3])
       #print(color_lst)   \n",
       for i in range(len(seq_ele)):
           if i <=len(color_lst):
              #print(color_lst[i])
              logo.highlight_position(p=i,color=color_lst[i], alpha=.9)
       #logo.fig.savefig('results/'+str(idx)+'.png')
       #print(logo.fig)
       #cv2.imwrite(os.path.join('results', f'{idx}.png'), img)

       progress_bar.update(1)

if __name__ == '__main__':
   main()

   