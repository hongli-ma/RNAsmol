
import os
import os.path as osp
import sys
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


dataset_name=sys.argv[1]


'''
Molecular graphs generation
'''

VOCAB_PROTEIN = { "A": 1, "C": 2, "G": 3,"T": 4,"U": 4,"a": 5,"c": 6,"g": 7,"t": 8,"u":8,"D": 9,"d":9, "B": 9,"b":9, "F": 9,"f":9, "I": 9, "i":9,"H": 9,"h":9, "K": 9,"k":9, "M": 9,"m":9, "L": 9,"l":9, "O": 9,"o":9, "N": 9,"n":9, "Q": 9,"q":9, "P": 9,"p":9, "S": 9,"s":9, "R": 9,"r":9, "E": 9,"e":9, "W": 9,"w":9, "V": 9,"v":9, "Y": 9, "y":9,"X": 9,"x":9, "Z": 9,"z":9 }

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target]


def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'])
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

    return np.array(encoding)

def mol_to_graph(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature/np.sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if len(edges) == 0:
        return features, [[0, 0]]

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    adj_matrix=np.zeros(shape=(len(g.nodes),len(g.nodes)))
    for i, j in zip(edge_index[0],edge_index[1]):
        adj_matrix[i-1,j-1]=1
    return features, edge_index,adj_matrix


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

class GNNDataset(InMemoryDataset):
    def __init__(self, root, types='train', alpha: float = 0.1, t: float=5.0, k: int = 5, eps: float = None, transform=None, pre_transform=None, pre_filter=None):
        
        self.alpha = alpha
        self.t = t
        self.k = k
        self.eps = eps
        
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif types == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

        
    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_val.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process_data(self, data_path, graph_dict):
        df = pd.read_csv(data_path)

        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence']
            label = row['affinity']

            if graph_dict.get(smi) == None:
                print("Unable to process: ", smi)
                delete_list.append(i)
                continue
            try:
               x, edge_index, adj_matrix = graph_dict[smi]
            except Exception as e:
               print(f"Error while processing: {smi}, Error: {e}")
               delete_list.append(i)
               continue

            ppr_matrix = get_ppr_matrix(adj_matrix,alpha=self.alpha)
            #heat_matrix = get_heat_matrix(adj_matrix,t=self.t)

            if self.k:
                #print(f'Selecting top {self.k} edges per node.')
                ppr_matrix = get_top_k_matrix(ppr_matrix, k=self.k)
                #heat_matrix = get_top_k_matrix(heat_matrix, k=self.k)
            elif self.eps:
                #print(f'Selecting edges with weight greater than {self.eps}.')
                ppr_matrix = get_clipped_matrix(ppr_matrix, eps=self.eps)
                #heat_matrix = get_clipped_matrix(heat_matrix, eps=self.eps)
            else:
                raise ValueError

            target = seqs2int(sequence)
            target_len = 500
            if len(target) < target_len:
                target = np.pad(target, (0, target_len- len(target)))
            else:
                target = target[:target_len]

            edges_i = []
            edges_j = []
            edge_attr = []
            for i, row in enumerate(ppr_matrix):
                for j in np.where(row > 0)[0]:
                    edges_i.append(i)
                    edges_j.append(j)
                    edge_attr.append(ppr_matrix[i, j])
            edge_index = [edges_i, edges_j]

            data = DATA.Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index),
                    edge_attr=torch.FloatTensor(edge_attr),
                    #edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    y=torch.FloatTensor([label]),
                    target=torch.LongTensor([target])
                )

            data_list.append(data)

        if len(delete_list) > 0:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)

        return data_list

    def process(self):
        df_train = pd.read_csv(self.raw_paths[0])
        df_val = pd.read_csv(self.raw_paths[1])
        df_test = pd.read_csv(self.raw_paths[2])
        df = pd.concat([df_train, df_val, df_test])
        smiles = df['compound_iso_smiles'].unique()

        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            if mol == None:
                print("Unable to process: ", smile)
                continue
            graph_dict[smile] = mol_to_graph(mol)

        train_list = self.process_data(self.raw_paths[0], graph_dict)
        val_list = self.process_data(self.raw_paths[1], graph_dict)
        test_list = self.process_data(self.raw_paths[2], graph_dict)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            val_list = [val for val in val_list if self.pre_filter(val)]
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            val_list = [self.pre_transform(val) for val in val_list]
            test_list = [self.pre_transform(test) for test in test_list]

        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), self.processed_paths[0])

        # save preprocessed val data:
        data, slices = self.collate(val_list)
        torch.save((data, slices), self.processed_paths[1])

        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), self.processed_paths[2])

if __name__ == "__main__":
    #data_split_train_val_test(data_root='data', data_set='human')
    #data_split_train_val_test(data_root='data', data_set='celegans')    
    GNNDataset(root='data/'+dataset_name)
    
