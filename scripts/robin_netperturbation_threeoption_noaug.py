### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-09
### Usage: Please cite RNAsmol when you use this script

###  different nohit option for robin dataset

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os

smm_na_sequence=pd.read_csv("ROBIN/SMM_full_results/SMM_Sequence_Table_hit_rates.csv")
### only retain RNA
smm_na_sequence=smm_na_sequence[smm_na_sequence['Biomolecule Type']=='RNA'].reset_index(drop=True)

smm_target_hits=pd.read_csv("ROBIN/SMM_full_results/SMM_Target_Hits.csv")

### drop DNA targets    
smm_target_hits.drop(columns=['KRAS_hit','MTOR_hit','MYB_hit','MYC_Pu22_hit','VEGF_hit','RB1_hit','BCL2_hit','CKIT_hit','MYCN_hit'],inplace=True)



smm_target_hits_new=smm_target_hits.iloc[:,2:].astype(float).fillna(0)
nohit_mol_ind_list = smm_target_hits_new.loc[(smm_target_hits_new==0.0).all(axis=1)].index.tolist()
hit_mol_ind_list = smm_target_hits_new.loc[~(smm_target_hits_new==0.0).all(axis=1)].index.tolist()
smm_target_nohit_df=smm_target_hits.loc[smm_target_hits.index[nohit_mol_ind_list]].reset_index(drop=True)
smm_target_hit_df=smm_target_hits.loc[smm_target_hits.index[hit_mol_ind_list]].reset_index(drop=True)



cols=smm_target_hits.columns
compound_iso_smiles=[]
target_sequence=[]
affinity=[]

for i in range(len(smm_target_hits)):
    for col in cols[2:]:
        name=col.rsplit("_",1)[0]
        ind=smm_na_sequence['Nucleic Acid Target'].tolist().index(name)
        compound_iso_smiles.append(smm_target_hits['Smile'][i])
        target_sequence.append(smm_na_sequence["Sequence (5' to 3')"][ind].replace('U','T').replace('(m6A)','').replace(' and ',''))
        affinity.append(smm_target_hits[col][i])
          
        
df=pd.DataFrame({'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


cols=smm_target_hit_df.columns
compound_iso_smiles=[]
target_sequence=[]
affinity=[]

for i in range(len(smm_target_hit_df)):
    for col in cols[2:]:
        name=col.rsplit("_",1)[0]
        ind=smm_na_sequence['Nucleic Acid Target'].tolist().index(name)
        compound_iso_smiles.append(smm_target_hit_df['Smile'][i])
        target_sequence.append(smm_na_sequence["Sequence (5' to 3')"][ind].replace('U','T').replace('(m6A)','').replace(' and ',''))
        affinity.append(smm_target_hit_df[col][i])
          
        
df_hit=pd.DataFrame({'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


cols=smm_target_nohit_df.columns
compound_iso_smiles=[]
target_sequence=[]
affinity=[]

for i in range(len(smm_target_nohit_df)):
    for col in cols[2:]:
        name=col.rsplit("_",1)[0]
        ind=smm_na_sequence['Nucleic Acid Target'].tolist().index(name)
        compound_iso_smiles.append(smm_target_nohit_df['Smile'][i])
        target_sequence.append(smm_na_sequence["Sequence (5' to 3')"][ind].replace('U','T').replace('(m6A)','').replace(' and ',''))
        affinity.append(smm_target_nohit_df[col][i])


df_nohit=pd.DataFrame({'compound_iso_smiles':compound_iso_smiles,'target_sequence':target_sequence,'affinity':affinity}).fillna(0)


df_pos=df[(df['affinity']==1.0)].reset_index(drop=True)
df_nohit_neg1=df_hit[(df_hit['affinity']==0.0)].reset_index(drop=True)
df_nohit_neg2=df_nohit[(df_nohit['affinity']==0.0)].reset_index(drop=True)
df_nohit_neg3=df[(df['affinity']==0.0)].reset_index(drop=True)

### random sampling 10 times

neg1_ind=df_nohit_neg1.index.tolist()
neg2_ind=df_nohit_neg2.index.tolist()
neg3_ind=df_nohit_neg3.index.tolist()


for i in range(10):
    random_neg1=random.sample(neg1_ind, len(df_pos))
    df_nohit_option1=pd.concat([df_pos,df_nohit_neg1.loc[random_neg1]])
    random_neg2=random.sample(neg2_ind, len(df_pos))
    df_nohit_option2=pd.concat([df_pos,df_nohit_neg2.loc[random_neg2]])
    random_neg3=random.sample(neg3_ind, len(df_pos))
    df_nohit_option3=pd.concat([df_pos,df_nohit_neg3.loc[random_neg3]])
    train, test = train_test_split(df_nohit_option1, test_size=0.2)
    train, val=train_test_split(train,test_size=0.25)
    outdir1 = 'data/robin_netperturbation_option1_'+str(i)
    if not os.path.exists(outdir1):
        os.mkdir(outdir1)
        os.mkdir(outdir1+'/raw')
    test.to_csv(outdir1+'/raw/data_test.csv',index=False)
    train.to_csv(outdir1+'/raw/data_train.csv',index=False)
    val.to_csv(outdir1+'/raw/data_val.csv',index=False)
    outdir2 = 'data/robin_netperturbation_option2_'+str(i)
    if not os.path.exists(outdir2):
        os.mkdir(outdir2)
        os.mkdir(outdir2+'/raw')
    train, test = train_test_split(df_nohit_option2, test_size=0.2)
    train, val=train_test_split(train,test_size=0.25)
    test.to_csv(outdir2+'/raw/data_test.csv',index=False)
    train.to_csv(outdir2+'/raw/data_train.csv',index=False)
    val.to_csv(outdir2+'/raw/data_val.csv',index=False)
    outdir3 = 'data/robin_netperturbation_option3_'+str(i)
    if not os.path.exists(outdir3):
        os.mkdir(outdir3)
        os.mkdir(outdir3+'/raw')
    train, test = train_test_split(df_nohit_option3, test_size=0.2)
    train, val=train_test_split(train,test_size=0.25)
    test.to_csv(outdir3+'/raw/data_test.csv',index=False)
    train.to_csv(outdir3+'/raw/data_train.csv',index=False)
    val.to_csv(outdir3+'/raw/data_val.csv',index=False)




