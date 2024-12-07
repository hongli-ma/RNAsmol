### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-12
### Usage: Please cite RNAsmol when you use this script


import os
import numpy as np
import torch
from torch_geometric.data import Batch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
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
        output, indices=torch.max(output,dim=-1)
        output[indices==0]=1 - output[indices==0]
        pred_prob=output.view(-1)
        #output = d.view(-1)
        #class_output=output[1]
        grad = torch.autograd.grad(pred_prob, self.target_feat,allow_unused=True)[0]
        #print(grad.size())
        if grad is None:
            grad=torch.zeros_like(self.target_feat)
        channel_weight = torch.mean(grad, dim=0, keepdim=True)
        channel_weight = normalize(channel_weight)
        weighted_feat = self.target_feat * channel_weight
        #print(weighted_feat.size())
        cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
        cam = normalize(cam)
        #print(cam)
        return output.detach().cpu().numpy(), cam



def main():
    device = torch.device('cpu')

    test_df=pd.read_csv('RNAsmol/visualization/4casestudy/raw/data_test.csv')
    test_set = GNNDataset('RNAsmol/visualization/4casestudy/', train=False)
    #test_df = pd.read_csv(os.path.join(fpath, 'raw', 'data_test.csv'))
    #test_set = GNNDataset(fpath, train=False)

    model = MCNN_GCN(3, 25 + 1, embedding_size=96, filter_num=32, out_dim=2,ban_heads=2).to(device)
    #model = MCNN_GCN(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
 
   
    #for k,v in model.state_dict().items():
    #   print(k)

   
    model.load_state_dict(torch.load('save/pdbrnaprotein_netshuffle_posneg11_rnafold_ppr_ban/model/epoch-0, loss-0.6926, val_loss-0.6962, test_loss-0.6956, test_acc-0.4615, test_sen-1.0000, test_spe-0.0000,test_pre-0.4615, test_rec-1.0000, test_f1-0.6316, test_rocauc-0.5442, test_prauc-0.4942, test_mcc-0.0000.pt',map_location=torch.device('cpu')))
    
    #module=model['protein_encoder.block_list.0.inc.conv_layer0.inc.0.weight']
  
    #gradcam = GradAAM(model, module=module)
    #gradcam = GradAAM(model, module=model.protein_encoder.block_list[0].inc[0])
    gradcam = GradAAM(model, module=model.ligand_encoder.conv1)


    smile_list = list(test_df['compound_iso_smiles'].unique())

    progress_bar = tqdm(total=len(smile_list))

    smile_atom_att_dict = {}
    for idx in range(len(test_set)):
        smile = test_df.iloc[idx]['compound_iso_smiles']

        if len(smile_list) == 0:
            break
        if smile in smile_list:
            smile_list.remove(smile)
        else:
            continue

        data = Batch.from_data_list([test_set[idx]])
        data = data.to(device)
        _, atom_att = gradcam(data)
        #print(smile)
        #print(len(smile))
        #print(atom_att)
        smile_atom_att_dict[smile] = atom_att.tolist()
        progress_bar.update(1)

    with open("pdb_netshuffle_casestudy_molsmile_atomgradaam.pkl", 'wb') as f:
       pickle.dump(smile_atom_att_dict,f)


if __name__ == '__main__':
    main()