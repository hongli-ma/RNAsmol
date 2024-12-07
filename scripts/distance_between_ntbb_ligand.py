### Author: Letian Gao 2024-10
### Usage: Please cite RNAsmol when you use this script


import Bio.PDB as PDB
import numpy as np
import os
import json
import math
import tqdm
import warnings
import rdkit.Chem as Chem
warnings.filterwarnings('ignore')
# set workdir to the directory of this script
os.chdir(os.path.dirname(__file__))
ion = 'CA,HG,K,NA,ZN,MG,CL,SR,I,CD,BA,N,CS,TL,IR,BR,MN'.split(',')
backbone = 'A+C+T+G+U'.split('+')

chem_smiles = json.load(open('./chemcomps_id.json','r'))
chem2smiles = {chem_smiles['data']["chem_comps"][i]["chem_comp"]["id"]: chem_smiles['data']["chem_comps"][i]["rcsb_chem_comp_descriptor"]["SMILES_stereo"] for i in range(len(chem_smiles['data']["chem_comps"]))}
smiles2chem = {value:key for key,value in chem2smiles.items()}

def distance_between_ntbb_ligand(pdb_id,rnaseq,targetsmiles,ligname=None):
    # download the structure
    if not os.path.exists('./ciffiles'):
        os.makedirs('./ciffiles')
    if not os.path.exists(f'./ciffiles/{pdb_id.lower()}.cif'):
        PDB.PDBList().retrieve_pdb_file(pdb_id, pdir='./ciffiles', file_format='mmCif')
    # load the structure
    structure = PDB.MMCIFParser().get_structure(pdb_id, f'./ciffiles/{pdb_id.lower()}.cif')
    # get the first model
    model = structure[0]
    bb_chains = {}
    ligs = []
    for chain in model:
        bb_res = []
        for res in chain:
            if res.resname in backbone:
                bb_res.append(res)
            elif res.resname in ion + ['HOH']:
                continue
            else:
                if ligname:
                    if res.resname == ligname:
                        ligs.append(res)
                else:
                    smiles = chem2smiles[res.resname]
                    if Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == Chem.MolToSmiles(Chem.MolFromSmiles(targetsmiles)):
                        ligs.append(res)
        seq = ''.join([res.resname for res in bb_res])
        if seq == rnaseq:
            bb_chains[chain.get_id()] = bb_res
    distmaps = []
    flag = False
    for bb_chain in bb_chains:
        flag = True
        for lig in ligs:
            dist = []
            N1_dist = []
            lig2nt_dist = []
            lig2nt_N1dist = []
            for bb_res in bb_chains[bb_chain]:
                tmp = []
                N1_tmp = math.inf
                for la in lig.child_list:
                    try:
                        N1_tmp = la-bb_res['N1']
                    except:
                        pass
                    for ba in bb_res.child_list:
                        tmp.append(ba-la)
                dist.append(min(tmp))
                N1_dist.append(N1_tmp)
            for la in lig.child_list:
                la_tmp = []
                lan1_tmp = []
                for bb_res in bb_chains[bb_chain]:
                    tmp = []
                    N1_tmp = math.inf
                    try:
                        N1_tmp = la-bb_res['N1']
                    except:
                        pass
                    for ba in bb_res.child_list:
                        tmp.append(ba-la)
                    la_tmp.append(min(tmp))
                    lan1_tmp.append(N1_tmp)
                lig2nt_dist.append(la_tmp)
                lig2nt_N1dist.append(lan1_tmp)
            if min(dist) <= 6.0:
                distmaps.append(
                    {'ligand':lig.resname, 
                     'chain':bb_chain, 
                     'dist':np.array(dist), 
                     'N1_dist':np.array(N1_dist), 
                     'lig2nt':np.array(lig2nt_dist),
                     'lig2nt_N1':np.array(lig2nt_N1dist)
                    }
                )
        break
    if not flag:
        print(f'{pdb_id} not found matched sequence')
    else:
        print(f'{pdb_id} found matched sequence')
    # del model, structure, bb_chains, ligs, seq, dist, N1_dist, flag
    return distmaps

def single_process(data):
    return distance_between_ntbb_ligand(data[0], data[2])

if __name__ == '__main__':

    import pickle
    with open('./pdb_rnaprotein_pdbid_ligandsmile_rnaseq_affinity_pos', 'r') as f:
        lines = f.readlines()
    datas = [line.rstrip().split('\t') for line in lines]
    datas = sorted(datas, key=lambda x: len(x[2]), reverse=False)

    items = []
    for data in tqdm.tqdm(datas):
        pdbid = data[0]
        # if pdbid in visited:
        #     continue
        # visited.add(pdbid)
        # if os.path.exists(f'./distmaps/{pdbid}.pkl'):
        #     continue
        # else:
        rnaseq = data[2]
        ligsmiles = data[1]
        if ligsmiles in smiles2chem:
            ligname = smiles2chem[ligsmiles]
        else:
            ligname = None
        results = distance_between_ntbb_ligand(pdbid,rnaseq,ligsmiles,ligname)
        # save pickle
        items.append({(pdbid,ligsmiles):results})
    with open(f'./distmaps.pkl', 'wb') as f:
        pickle.dump(items, f)

# test_pdbid = datas[0][0]
# items = []
# visited = set()
# for data in tqdm.tqdm(datas):
#     pdbid = data[0]
#     if pdbid in visited:
#         continue
#     visited.add(pdbid)
#     distmaps = distance_between_ntbb_ligand(pdbid, data[2])
#     items.append({pdbid:distmaps})
    

# # save pickle
# import pickle
# with open('./distmaps.pkl', 'wb') as f:
#     pickle.dump(items, f)
