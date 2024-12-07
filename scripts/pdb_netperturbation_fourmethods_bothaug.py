### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-10
### Usage: Please cite RNAsmol when you use this script


### generate rhon interaction of augmented RNA target sequence
### method1: nested for loops to write other edges (failed), but we use it for the molaug data generation, it works for small dataset

riboaug_node_set1 = df_train_aug['compound_iso_smiles'].tolist()
riboaug_node_set2 = df_train_aug['target_sequence'].tolist()
riboaug_edge_set = df_train_aug[['compound_iso_smiles', 'target_sequence']].apply(tuple, axis=1).tolist()


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

### failed: The Kernel crashed 
riboaug_complete_edges=calculate_complete_interaction_remove_known_edges(riboaug_node_set1,riboaug_node_set2,riboaug_edge_set)



### generate rhon interaction of augmented RNA target sequence
### method2: use networkx digraph adjacency matrix conversion (failed)

import networkx as nx
from scipy.sparse import coo_array

G=nx.DiGraph()
edge_list=df_train_aug.values.tolist()

for i in range(len(edge_list)):
    G.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])

adj=nx.adjacency_matrix(G).A
adj_conv=np.where(adj<0.5,1,0)

### failed: MemoryError: Unable to allocate 74.1 GiB for an array with shape (4969470873, 2) and data type int64
nx.from_numpy_matrix(adj_conv)

riboaug_complete_edges=[]
#for line in nx.generate_edgelist(G,data=["weight"]):
    #print(line)
    #riboaug_complete_edges.append(line)



### generate rhon interaction of augmented RNA target sequence
### method3: use dataframe pivot and melt (failed)

vals=np.unique(df_train_aug[['compound_iso_smiles','target_sequence']])

### failed: The Kernel crashed 
temp=df_train_aug.pivot(index='compound_iso_smiles',columns='target_sequence',values='affinity').reindex(index=vals,columns=vals).fillna(0)

ss=temp.rename_axis('compound_iso_smiles')\
  .reset_index()\
  .melt('compound_iso_smiles', value_name='affinity', var_name='target_sequence')\
  .query('compound_iso_smiles != target_sequence')\
  .reset_index(drop=True)

ss1=ss[ss['compound_iso_smiles'].str.contains('G')==False]

ss2=ss1[ss1['target_sequence'].str.contains('\(')==False]
riboaug_complete_edges=ss2[ss2['affinity']==0].values.tolist()



### generate rhon interaction of augmented RNA target sequence
### method4: use networkx bipartite graph and rempve known edges (success)
 
import networkx as nx

# Define the two node sets
node_set1 = list(set(df_train_aug['compound_iso_smiles'].tolist()))
node_set2 = list(set(df_train_aug['target_sequence'].tolist()))

# Create a complete bipartite graph
G = nx.complete_bipartite_graph(node_set1, node_set2)

# Define a list of known edges to be removed
known_edges_to_remove = df_train_aug[['compound_iso_smiles', 'target_sequence']].apply(tuple, axis=1).tolist()

# Remove the known edges from the graph
G.remove_edges_from(known_edges_to_remove)


riboaug_complete_edges_tuples=list(G.edges)
riboaug_complete_edges = [(t[0], t[1], 0) for t in riboaug_complete_edges_tuples]