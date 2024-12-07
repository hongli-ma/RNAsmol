### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-09
### Usage: Please cite RNAsmol when you use this script

### bothaug
### augment train, val dataset by rMSA target sequence and molecule MACCS fp tanimoto similarity (rhon, rhor, rhom)



import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import pandas as pd
import json
import numpy as np
import csv
import re
import os
from sklearn.model_selection import train_test_split
import glob
import random
import multiprocess



afa_files=glob.glob("rnadrug_dataset/ROBIN/ROBIN_27RNA/*.afa_0.8")
file_name_key=[]
for file in afa_files:
    file_name_key.append(file.rsplit("_",3)[0].split("/")[-1])


smm_na_sequence=pd.read_csv("rnadrug_dataset/ROBIN/SMM_full_results/SMM_Sequence_Table_hit_rates.csv")
### only retain RNA
smm_na_sequence=smm_na_sequence[smm_na_sequence['Biomolecule Type']=='RNA'].reset_index(drop=True)

smm_target_hits=pd.read_csv("rnadrug_dataset/ROBIN/SMM_full_results/SMM_Target_Hits.csv")
### drop DNA targets    
smm_target_hits.drop(columns=['KRAS_hit','MTOR_hit','MYB_hit','MYC_Pu22_hit','VEGF_hit','RB1_hit','BCL2_hit','CKIT_hit','MYCN_hit'],inplace=True)


smm_target_hits_new=smm_target_hits.iloc[:,2:].astype(float).fillna(0)
hit_mol_ind_list = smm_target_hits_new.loc[~(smm_target_hits_new==0.0).all(axis=1)].index.tolist()
smm_target_hit_df=smm_target_hits.loc[smm_target_hits.index[hit_mol_ind_list]].reset_index(drop=True)


cols=smm_target_hit_df.columns
compound_iso_smiles=[]
target_sequence=[]
affinity=[]
name_list=[]

for i in range(len(smm_target_hit_df)):
    for col in cols[2:]:
        name=col.rsplit("_",1)[0]
        ind=smm_na_sequence['Nucleic Acid Target'].tolist().index(name)
        name_list.append(name)
        compound_iso_smiles.append(smm_target_hit_df['Smile'][i])
        target_sequence.append(smm_na_sequence["Sequence (5' to 3')"][ind].replace('(m6A)','').replace(' and ',''))
        affinity.append(smm_target_hit_df[col][i])
        
          
df=pd.DataFrame({'target_name':name_list,'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)        

df_pos=df[(df['affinity']==1.0)].reset_index(drop=True)
df_nohit_neg=df[(df['affinity']==0.0)].reset_index(drop=True)


train, test = train_test_split(df_pos, test_size=0.2)

train,val = train_test_split(train,test_size=0.25)
train=train.reset_index(drop=True)
val=val.drop(['target_name'], axis=1).reset_index(drop=True)
test=test.drop(['target_name'], axis=1).reset_index(drop=True)



def cal_hpdf(fps,smiles,mol_fps):
    sim_list = []
    for i in range(len(fps)):
        similarity = DataStructs.TanimotoSimilarity(fps[i],mol_fps)
        sim_list.append([similarity,smiles[i]])
    sim_hp = pd.DataFrame(sim_list, columns=['similarity', 'smiles'])
    return sim_hp

smiles=pd.read_csv('rnadrug_dataset/in-vitro.smi')['SMILES'].tolist() 

decoyset_fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)) for x in smiles]


compound_iso_smiles=[]
target_sequence=[]
affinity=[]
name_list=[]


for i in range(len(train)):
    name_list.append(train['target_name'][i])
    compound_iso_smiles.append(train['compound_iso_smiles'][i])
    target_sequence.append(train['target_sequence'][i].strip().replace('(m6A)','').replace(' and ',''))
    affinity.append(train['affinity'][i])
    try:
        mol=train['compound_iso_smiles'][i]
        mol_fp = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(mol))
        mol_df = cal_hpdf(decoyset_fps,smiles,mol_fp)
        top5 = mol_df.sort_values(by = 'similarity',ascending=False,inplace=False).head(5)
        top5=top5.reset_index(drop=True)
        for ii in range(len(top5)):
            compound_iso_smiles.append(top5['smiles'][ii])
            target_sequence.append(train['target_sequence'][i].strip().replace('(m6A)','').replace(' and ',''))
            affinity.append(train['affinity'][i])
            name_list.append(train['target_name'][i])
    except:
        pass


df_train_aug_temp1=pd.DataFrame({'target_name':name_list,'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


compound_iso_smiles=[]
target_sequence=[]
affinity=[]

for i in range(len(train)):

    name = train['target_name'][i]

    try:
        ind=file_name_key.index(name)

        with open(afa_files[ind]) as f:
            lines=f.readlines()
            for line in lines:
                if '>' not in line:
                    compound_iso_smiles.append(train['compound_iso_smiles'][i])
                    target_sequence.append(line.strip().replace('(m6A)','').replace(' and ',''))
                    affinity.append(train['affinity'][i])
    except:
        pass

df_train_aug_temp2=pd.DataFrame({'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


df_train_aug=pd.concat([df_train_aug_temp1,df_train_aug_temp2]).drop('target_name',axis=1).reset_index(drop=True)


compound_iso_smiles=[]
target_sequence=[]
affinity=[]
name_list=[]


for i in range(len(val)):
    name_list.append(val['target_name'][i])
    compound_iso_smiles.append(val['compound_iso_smiles'][i])
    target_sequence.append(val['target_sequence'][i].strip().replace('(m6A)','').replace(' and ',''))
    affinity.append(val['affinity'][i])
    try:
        mol=val_pos['compound_iso_smiles'][i]
        top5 = mol_set_dict[mol]
        for ii in range(len(top5)):
            compound_iso_smiles.append(top5[ii])
            target_sequence.append(val['target_sequence'][i].strip().replace('(m6A)','').replace(' and ',''))
            affinity.append(val['affinity'][i])
            name_list.append(val['target_name'][i])
    except:
        pass


df_val_aug_temp1=pd.DataFrame({'target_name':name_list,'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


compound_iso_smiles=[]
target_sequence=[]
affinity=[]

for i in range(len(val)):

    name = val['target_name'][i]

    try:
        ind=file_name_key.index(name)

        with open(afa_files[ind]) as f:
            lines=f.readlines()
            for line in lines:
                if '>' not in line:
                    compound_iso_smiles.append(val['compound_iso_smiles'][i])
                    target_sequence.append(line.strip().replace('(m6A)','').replace(' and ',''))
                    affinity.append(val['affinity'][i])
    except:
        pass

df_val_aug_temp2=pd.DataFrame({'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


df_val_aug=pd.concat([df_val_aug_temp1,df_val_aug_temp2]).drop('target_name',axis=1).reset_index(drop=True)



compound_iso_smiles=[]
target_sequence=[]
affinity=[]

for i in range(len(df_nohit_neg)):
    compound_iso_smiles.append(df_nohit_neg['compound_iso_smiles'][i])
    target_sequence.append(df_nohit_neg['target_sequence'][i])
    affinity.append(df_nohit_neg['affinity'][i])
    name = df_nohit_neg['target_name'][i]
    try:
        ind=file_name_key.index(name)
        mol=df_nohit_neg['compound_iso_smiles'][i]
        mol_fp = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(mol))
        mol_df = cal_hpdf(decoyset_fps,smiles,mol_fp)
        top5 = mol_df.sort_values(by = 'similarity',ascending=False,inplace=False).head(5)
        top5=top5.reset_index(drop=True)

        for ii in range(len(top5)):
            compound_iso_smiles.append(top5['smiles'][ii])
            target_sequence.append(df_nohit_neg['target_sequence'][i])
            affinity.append(df_nohit_neg['affinity'][i])

        with open(afa_files[ind]) as f:
            lines=f.readlines()
            for line in lines:
                if '>' not in line:
                    compound_iso_smiles.append(df_nohit_neg['compound_iso_smiles'][i])
                    target_sequence.append(line.strip().replace('(m6A)','').replace(' and ',''))
                    affinity.append(df_nohit_neg['affinity'][i])
                   
    except:
        pass

df_aug_nohit=pd.DataFrame({'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


### rohn of augmented RNA target sequence and small molecule ligand
df_aug_nohit_ind=df_aug_nohit.index.tolist()
df_nohit_neg=df_nohit_neg.drop(['target_name'], axis=1).reset_index(drop=True)
df_nohit_neg_ind=df_nohit_neg.index.tolist()

random_1=random.sample(df_nohit_neg_ind, len(test))
test_nohit=pd.concat([test,df_nohit_neg.loc[random_1]])

random_2=random.sample(df_nohit_neg_ind, len(val))
val_aug_nohit=pd.concat([df_val_aug,df_aug_nohit.loc[random_2]])

random_3=random.sample(df_aug_nohit_ind, len(df_train_aug))
train_aug_nohit=pd.concat([df_train_aug,df_aug_nohit.loc[random_3]])

outdir1 = 'data/robin_bothaug_rnaaug_nohit'
if not os.path.exists(outdir1):
    os.mkdir(outdir1)
    os.mkdir(outdir1+'/raw')

train_aug_nohit.to_csv(outdir1+'/raw/data_train.csv',index=False)
val_aug_nohit.to_csv(outdir1+'/raw/data_val.csv',index=False)
test_nohit.to_csv(outdir1+'/raw/data_test.csv',index=False)


### rohr of augmented RNA target sequence and small molecule ligand
outdir2 = 'data/robin_bothaug_rnaaug_dinu'
if not os.path.exists(outdir2):
    os.mkdir(outdir2)
    os.mkdir(outdir2+'/raw')

index=0
with open(outdir2+"/raw/data_train.csv","w+") as fcsv:
    fcsv.write("compound_iso_smiles,target_sequence,affinity"+"\n")
    for i in range(len(df_train_aug)):
        neg=dinuclShuffle(df_train_aug['target_sequence'][i]).replace('T','U')
        fcsv.write(df_train_aug['compound_iso_smiles'][i]+','+df_train_aug['target_sequence'][i]+','+'1'+'\n')
        fcsv.write(df_train_aug['compound_iso_smiles'][i]+','+neg+','+'0'+'\n')
        index+=1

index=0
with open(outdir2+"/raw/data_val.csv","w+") as fcsv:
    fcsv.write("compound_iso_smiles,target_sequence,affinity"+"\n")
    for i in range(len(df_val_aug)):
        neg=dinuclShuffle(df_val_aug['target_sequence'][i]).replace('T','U')
        fcsv.write(df_val_aug['compound_iso_smiles'][i]+','+df_val_aug['target_sequence'][i]+','+'1'+'\n')
        fcsv.write(df_val_aug['compound_iso_smiles'][i]+','+neg+','+'0'+'\n')
        index+=1

index=0
with open(outdir2+"/raw/data_test.csv","w+") as fcsv:
    fcsv.write("compound_iso_smiles,target_sequence,affinity"+"\n")
    for i in range(len(test)):
        neg=dinuclShuffle(test['target_sequence'][i]).replace('T','U')
        fcsv.write(test['compound_iso_smiles'][i]+','+test['target_sequence'][i]+','+'1'+'\n')
        fcsv.write(test['compound_iso_smiles'][i]+','+neg+','+'0'+'\n')
        index+=1


### rohm of augmented RNA target sequence and small molecule ligand
outdir3 = 'data/robin_bothaug_rnaaug_proteinbinder'
if not os.path.exists(outdir3):
    os.mkdir(outdir3)
    os.mkdir(outdir3+'/raw')

df_proteinbinder=pd.read_csv('rnadrug_dataset/robinrnabinder_bindingdbproteinbinder.csv')
df_proteinbinder_neg=df_proteinbinder[df_proteinbinder['target']==0.0].reset_index(drop=True)

###bothaug's positive set is larger than fda negative set, so we need to duplicate proteinbinder negative set
df_proteinbinder_neg=pd.concat([df_proteinbinder_neg,df_proteinbinder_neg]).reset_index(drop=True)


index=0
with open(outdir3+"/raw/data_train.csv","w+") as fcsv:
    fcsv.write("compound_iso_smiles,target_sequence,affinity"+"\n")
    for i in range(len(df_train_aug)):
        neg=df_proteinbinder_neg['SMILES'][i]
        fcsv.write(df_train_aug['compound_iso_smiles'][i]+','+df_train_aug['target_sequence'][i]+','+'1'+'\n')
        fcsv.write(neg+','+df_train_aug['target_sequence'][i]+','+'0'+'\n')
        index+=1

index=0
with open(outdir3+"/raw/data_val.csv","w+") as fcsv:
    fcsv.write("compound_iso_smiles,target_sequence,affinity"+"\n")
    for i in range(len(df_val_aug)):
        neg=df_proteinbinder_neg['SMILES'][i]
        fcsv.write(df_val_aug['compound_iso_smiles'][i]+','+df_val_aug['target_sequence'][i]+','+'1'+'\n')
        fcsv.write(neg+','+df_val_aug['target_sequence'][i]+','+'0'+'\n')
        index+=1

index=0
with open(outdir3+"/raw/data_test.csv","w+") as fcsv:
    fcsv.write("compound_iso_smiles,target_sequence,affinity"+"\n")
    for i in range(len(test)):
        neg=df_proteinbinder_neg['SMILES'][i]
        fcsv.write(test['compound_iso_smiles'][i]+','+test['target_sequence'][i]+','+'1'+'\n')
        fcsv.write(neg+','+test['target_sequence'][i]+','+'0'+'\n')
        index+=1
