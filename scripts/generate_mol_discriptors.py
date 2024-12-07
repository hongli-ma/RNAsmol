### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-11
### Usage: Please cite RNAsmol when you use this script

###generate descriptors of robin mol 

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

robin_mol=list(set(df['compound_iso_smiles'].tolist()))

robin_mol_lst = []
for smi in robin_mol:
    try:
        mol = Chem.MolFromSmiles(smi)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        mollogp = Descriptors.MolLogP(mol)
        rota_bonds = Descriptors.NumRotatableBonds(mol)
        wei = Descriptors.ExactMolWt(mol)
        robin_other_mol_lst.append((hba, hbd, mollogp, rota_bonds, wei))
    except:
        pass

np.save('datasets/robin_mol.npy',np.array(robin_mol_lst))