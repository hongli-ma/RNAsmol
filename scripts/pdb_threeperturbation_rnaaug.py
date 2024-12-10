### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-10
### Usage: Please cite RNAsmol when you use this script


### rnaaug
### only augment train, val RNA target in dataset by cmsearch (rhon, rhor, rhom)

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os
import networkx as nx
import random

pdbrnaprotein_nonredundent_df_pos=pd.read_csv("datasets/pdb_rnaprotein_ligandsmile_rnaseq_affinity_pos_nonredundent",sep='\t',header=None)
pdbrnaprotein_nonredundent_df_pos.columns=['compound_iso_smiles','target_sequence','affinity']

ribo_homo_pos=pd.read_csv('datasets/data_rna_homo.csv',header=None)


train, test = train_test_split(pdbrnaprotein_nonredundent_df_pos, test_size=0.2)

train,val = train_test_split(train,test_size=0.25)
train=train.reset_index(drop=True)
val=val.reset_index(drop=True)
test=test.reset_index(drop=True)



df_ribo_homo_pos=ribo_homo_pos.iloc[:,1:]
df_ribo_homo_pos.columns=['target_sequence','compound_iso_smiles','affinity']

df1 = df_ribo_homo_pos.sample(frac=8/9, random_state=42)
df2 = df_ribo_homo_pos.drop(df1.index)

df_train_aug=pd.concat([train,df1]).reset_index(drop=True)
df_val_aug=pd.concat([val,df2]).reset_index(drop=True)

df_val_aug.drop_duplicates(inplace=True)
df_val_aug=df_val_aug.reset_index(drop=True,inplace=True)

df_train_aug=pd.concat([train,df_ribo_homo_pos]).reset_index(drop=True)
df_train_aug.drop_duplicates(inplace=True)
df_train_aug=df_train_aug.reset_index(drop=True,inplace=True)

### rhon

pdbrnaprotein_node_set1 = pdbrnaprotein_nonredundent_df_pos['compound_iso_smiles'].tolist()
pdbrnaprotein_node_set2 = pdbrnaprotein_nonredundent_df_pos['target_sequence'].tolist()
pdbrnaprotein_edge_set = pdbrnaprotein_nonredundent_df_pos[['compound_iso_smiles', 'target_sequence']].apply(tuple, axis=1).tolist()


def calculate_complete_interaction_remove_known_edges(node_set1,node_set2,edge_set):
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
    return complete_edges


pdbrnaprotein_complete_edges=calculate_complete_interaction_remove_known_edges(pdbrnaprotein_node_set1,pdbrnaprotein_node_set2,pdbrnaprotein_edge_set)


# Define the two node sets
riboaug_node_set1 = list(set(df_train_aug['compound_iso_smiles'].tolist()))
riboaug_node_set2 = list(set(df_train_aug['target_sequence'].tolist()))

# Create a complete bipartite graph
G = nx.complete_bipartite_graph(riboaug_node_set1, riboaug_node_set2)

# Define a list of known edges to be removed
known_edges_to_remove = df_train_aug[['compound_iso_smiles', 'target_sequence']].apply(tuple, axis=1).tolist()

# Remove the known edges from the graph
G.remove_edges_from(known_edges_to_remove)


riboaug_complete_edges_tuples=list(G.edges)
riboaug_complete_edges = [(t[0], t[1], 0) for t in riboaug_complete_edges_tuples]
riboaug_complete_edges=pd.DataFrame(riboaug_complete_edges,columns=['compound_iso_smiles','target_sequence','affinity'])


df_aug_netshuffle_neg_ind=riboaug_complete_edges.index.tolist()

df_netshuffle_neg_ind=pdbrnaprotein_complete_edges.index.tolist()

random_1=random.sample(df_netshuffle_neg_ind, len(test))
test_netshuffle=pd.concat([test,pdbrnaprotein_complete_edges.loc[random_1]])

random_2=random.sample(df_aug_netshuffle_neg_ind, len(df_val_aug))
val_aug_netshuffle=pd.concat([df_val_aug,riboaug_complete_edges.loc[random_2]])

random_3=random.sample(df_aug_netshuffle_neg_ind, len(df_train_aug))
train_aug_netshuffle=pd.concat([df_train_aug,riboaug_complete_edges.loc[random_3]])

outdir1 = 'data/pdbrnaprotein_riboaug_netshuffle'
if not os.path.exists(outdir1):
    os.mkdir(outdir1)
    os.mkdir(outdir1+'/raw')

train_aug_netshuffle.to_csv(outdir1+'/raw/data_train.csv',index=False)
val_aug_netshuffle.to_csv(outdir1+'/raw/data_val.csv',index=False)
test_netshuffle.to_csv(outdir1+'/raw/data_test.csv',index=False)


### rhor
outdir2 = 'data/pdbrnaprotein_riboaug_dinu'
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

### rhom
outdir3 = 'data/pdbrnaprotein_riboaug_proteinbinder'
if not os.path.exists(outdir3):
    os.mkdir(outdir3)
    os.mkdir(outdir3+'/raw')

df_proteinbinder=pd.read_csv('rnadrug_dataset/robinrnabinder_bindingdbproteinbinder.csv')
df_proteinbinder_neg=df_proteinbinder[df_proteinbinder['target']==0.0].reset_index(drop=True)

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
