### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-10
### Usage: Please cite RNAsmol when you use this script

### pdbrnaprotein data preparation from pdb2fasta chain seq>10 (rhon, rhor, rhom)(pos:neg=1:1,1:2,1:5)

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os

pdbrnaprotein_nonredundent_df_pos=pd.read_csv("rnadrug_dataset/pdb_rnaprotein_ligandsmile_rnaseq_affinity_pos_nonredundent",sep='\t',header=None)
pdbrnaprotein_nonredundent_df_pos.columns=['compound_iso_smiles','target_sequence','affinity']


node_set1 = pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'].tolist()
node_set2 = pdbrnaprotein_nonredundent_df_pos['target_sequence'].tolist()
edge_set = pdbrnaprotein_nonredundent_df_pos[['compound_iso_smiles', 'target_sequence']].apply(tuple, axis=1).tolist()

i=0
complete_edges=[]
for node1 in node_set1:
    j=0
    for node2 in node_set2:
        if i !=j:
            complete_edges.append((node1, node2,0))
        else:
            pass
        j=j+1
    i=i+1
complete_edges=pd.DataFrame(complete_edges,columns=['compound_iso_smiles','target_sequence','affinity'])

### rhon
nohit_ind_lst1=random.sample(complete_edges.index.tolist(), len(df_pos))
nohit_ind_lst2=random.sample(complete_edges.index.tolist(), 2*len(df_pos))
nohit_ind_lst3=random.sample(complete_edges.index.tolist(), 5*len(df_pos))

df_pdbrnaprotein_nohit_posneg11=pd.concat([pdbrnaprotein_nonredundent_df_pos,complete_edges.loc[nohit_ind_lst1]])
df_pdbrnaprotein_nohit_posneg12=pd.concat([pdbrnaprotein_nonredundent_df_pos,complete_edges.loc[nohit_ind_lst2]])
df_pdbrnaprotein_nohit_posneg15=pd.concat([pdbrnaprotein_nonredundent_df_pos,complete_edges.loc[nohit_ind_lst3]])                              

train1, test1 = train_test_split(df_pdbrnaprotein_nohit_posneg11, test_size=0.2)
train2, test2 = train_test_split(df_pdbrnaprotein_nohit_posneg12, test_size=0.2)
train3, test3 = train_test_split(df_pdbrnaprotein_nohit_posneg15, test_size=0.2)

train1, val1=train_test_split(train1,test_size=0.25)
train2, val2=train_test_split(train2,test_size=0.25)
train3, val3=train_test_split(train3,test_size=0.25)

outdir1 = 'data/pdbrnaprotein_netshuffle_posneg11'
if not os.path.exists(outdir1):
    os.mkdir(outdir1)
    os.mkdir(outdir1+'/raw')

test1.to_csv(outdir1+'/raw/data_test.csv',index=False)
train1.to_csv(outdir1+'/raw/data_train.csv',index=False)
val1.to_csv(outdir1+'/raw/data_val.csv',index=False)

outdir2 = 'data/pdbrnaprotein_netshuffle_posneg12'
if not os.path.exists(outdir2):
    os.mkdir(outdir2)
    os.mkdir(outdir2+'/raw')

test2.to_csv(outdir2+'/raw/data_test.csv',index=False)
train2.to_csv(outdir2+'/raw/data_train.csv',index=False)
val2.to_csv(outdir2+'/raw/data_val.csv',index=False)

outdir3 = 'data/pdbrnaprotein_netshuffle_posneg15'
if not os.path.exists(outdir3):
    os.mkdir(outdir3)
    os.mkdir(outdir3+'/raw')

test3.to_csv(outdir3+'/raw/data_test.csv',index=False)
train3.to_csv(outdir3+'/raw/data_train.csv',index=False)
val3.to_csv(outdir3+'/raw/data_val.csv',index=False)


### rhor
posneg11_lst=[]
posneg12_lst=[]
posneg15_lst=[]


index=0
for i in range(len(pdbrnaprotein_nonredundent_df_pos)):
    neg=dinuclShuffle(pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]).replace('T','U')
    neg2=dinuclShuffle(pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]).replace('T','U')
    neg3=dinuclShuffle(pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]).replace('T','U')
    neg4=dinuclShuffle(pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]).replace('T','U')
    neg5=dinuclShuffle(pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]).replace('T','U')

    posneg11_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]+','+'1')
    posneg11_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg+','+'0')
    posneg12_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]+','+'1')
    posneg12_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg+','+'0')
    posneg12_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg2+','+'0')
    posneg15_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+pdbrnaprotein_nonredundent_df_pos['target_sequence'][i]+','+'1')
    posneg15_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg+','+'0')
    posneg15_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg2+','+'0')
    posneg15_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg3+','+'0')
    posneg15_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg4+','+'0')
    posneg15_lst.append(pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'][i]+','+neg5+','+'0')

  
train1, test1 = train_test_split(posneg11_lst, test_size=0.2)
train2, test2 = train_test_split(posneg12_lst, test_size=0.2)
train3, test3 = train_test_split(posneg15_lst, test_size=0.2)

train1, val1=train_test_split(train1,test_size=0.25)
train2, val2=train_test_split(train2,test_size=0.25)
train3, val3=train_test_split(train3,test_size=0.25)

uni_cols=['compound_iso_smiles','target_sequence','affinity']
outdir1 = 'data/pdbrnaprotein_dinu_posneg11'
if not os.path.exists(outdir1):
    os.mkdir(outdir1)
    os.mkdir(outdir1+'/raw')

pd.DataFrame([sub.split(",") for sub in test1],columns=uni_cols).to_csv(outdir1+'/raw/data_test.csv',index=False)
pd.DataFrame([sub.split(",") for sub in train1],columns=uni_cols).to_csv(outdir1+'/raw/data_train.csv',index=False)
pd.DataFrame([sub.split(",") for sub in val1],columns=uni_cols).to_csv(outdir1+'/raw/data_val.csv',index=False)

outdir2 = 'data/pdbrnaprotein_dinu_posneg12'
if not os.path.exists(outdir2):
    os.mkdir(outdir2)
    os.mkdir(outdir2+'/raw')

pd.DataFrame([sub.split(",") for sub in test2],columns=uni_cols).to_csv(outdir2+'/raw/data_test.csv',index=False)
pd.DataFrame([sub.split(",") for sub in train2],columns=uni_cols).to_csv(outdir2+'/raw/data_train.csv',index=False)
pd.DataFrame([sub.split(",") for sub in val2],columns=uni_cols).to_csv(outdir2+'/raw/data_val.csv',index=False)

outdir3 = 'data/pdbrnaprotein_dinu_posneg15'
if not os.path.exists(outdir3):
    os.mkdir(outdir3)
    os.mkdir(outdir3+'/raw')

pd.DataFrame([sub.split(",") for sub in test3],columns=uni_cols).to_csv(outdir3+'/raw/data_test.csv',index=False)
pd.DataFrame([sub.split(",") for sub in train3],columns=uni_cols).to_csv(outdir3+'/raw/data_train.csv',index=False)
pd.DataFrame([sub.split(",") for sub in val3],columns=uni_cols).to_csv(outdir3+'/raw/data_val.csv',index=False)


### rhom

df_proteinbinder=pd.read_csv('rnadrug_dataset/robinrnabinder_bindingdbproteinbinder.csv')
df_proteinbinder_neg=df_proteinbinder[df_proteinbinder['target']==0.0].reset_index(drop=True)


lst=df_proteinbinder_neg.index.tolist()
random_negind1=random.sample(lst, len(pdbrnaprotein_nonredundent_df_pos))
random_negind2=random.sample(lst, 2*len(pdbrnaprotein_nonredundent_df_pos))
random_negind3=random.sample(lst, 5*len(pdbrnaprotein_nonredundent_df_pos))

negsmile1=df_proteinbinder_neg.loc[random_negind1]['SMILES'].tolist()
negsmile2=df_proteinbinder_neg.loc[random_negind2]['SMILES'].tolist()
negsmile3=df_proteinbinder_neg.loc[random_negind3]['SMILES'].tolist()

negaff1=df_proteinbinder_neg.loc[random_negind1]['target'].tolist()
negaff2=df_proteinbinder_neg.loc[random_negind2]['target'].tolist()
negaff3=df_proteinbinder_neg.loc[random_negind3]['target'].tolist()

df_neg1=pd.DataFrame({'compound_iso_smiles':negsmile1,'target_sequence':pdbrnaprotein_nonredundent_df_pos['target_sequence'].tolist(),'affinity':negaff1})
df_neg2=pd.DataFrame({'compound_iso_smiles':negsmile2,'target_sequence':pdbrnaprotein_nonredundent_df_pos['target_sequence'].tolist()*2,'affinity':negaff2})
df_neg3=pd.DataFrame({'compound_iso_smiles':negsmile3,'target_sequence':pdbrnaprotein_nonredundent_df_pos['target_sequence'].tolist()*5,'affinity':negaff3})

df_pdbrnaprotein_proteinbinder_posneg11=pd.concat([pdbrnaprotein_nonredundent_df_pos,df_neg1])
df_pdbrnaprotein_proteinbinder_posneg12=pd.concat([pdbrnaprotein_nonredundent_df_pos,df_neg2])
df_pdbrnaprotein_proteinbinder_posneg15=pd.concat([pdbrnaprotein_nonredundent_df_pos,df_neg3])

train1, test1 = train_test_split(df_pdbrnaprotein_proteinbinder_posneg11, test_size=0.2)
train2, test2 = train_test_split(df_pdbrnaprotein_proteinbinder_posneg12, test_size=0.2)
train3, test3 = train_test_split(df_pdbrnaprotein_proteinbinder_posneg15, test_size=0.2)

train1, val1=train_test_split(train1,test_size=0.25)
train2, val2=train_test_split(train2,test_size=0.25)
train3, val3=train_test_split(train3,test_size=0.25)

outdir1 = 'data/pdbrnaprotein_proteinbinder_posneg11'
if not os.path.exists(outdir1):
    os.mkdir(outdir1)
    os.mkdir(outdir1+'/raw')

test1.to_csv(outdir1+'/raw/data_test.csv',index=False)
train1.to_csv(outdir1+'/raw/data_train.csv',index=False)
val1.to_csv(outdir1+'/raw/data_val.csv',index=False)

outdir2 = 'data/pdbrnaprotein_proteinbinder_posneg12'
if not os.path.exists(outdir2):
    os.mkdir(outdir2)
    os.mkdir(outdir2+'/raw')

test2.to_csv(outdir2+'/raw/data_test.csv',index=False)
train2.to_csv(outdir2+'/raw/data_train.csv',index=False)
val2.to_csv(outdir2+'/raw/data_val.csv',index=False)

outdir3 = 'data/pdbrnaprotein_proteinbinder_posneg15'
if not os.path.exists(outdir3):
    os.mkdir(outdir3)
    os.mkdir(outdir3+'/raw')

test3.to_csv(outdir3+'/raw/data_test.csv',index=False)
train3.to_csv(outdir3+'/raw/data_train.csv',index=False)
val3.to_csv(outdir3+'/raw/data_val.csv',index=False)

