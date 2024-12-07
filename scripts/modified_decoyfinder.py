### decoyfinder.py remove dict2pkl and log version
### modified from https://github.com/URV-cheminformatics/DecoyFinder/blob/master/find_decoys.py



import numpy as np
import pandas as pd
import os
import rdkit
import pickle
from tqdm import *
from multiprocessing import Pool
from multiprocessing import Process
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdmolfiles
from rdkit import DataStructs
from decimal import Decimal
from random import shuffle

# Define dict containing parameter calculation functions
calc_functs = {
    True:{
    'mw': lambda mol: Decimal(mol.GetProp('MW')).quantize(Decimal('0.000')),
    'hba': lambda mol: int(mol.GetProp('H_A')),
    'hbd': lambda mol: int(mol.GetProp('H_D')),
    'clogp': lambda mol: Decimal(mol.GetProp('LogP')).quantize(Decimal('0.000')),
    'rot': lambda mol: int(mol.GetProp('R_B'))
    },
    False:{
    'mw': lambda mol: Decimal(Descriptors.MolWt(mol)).quantize(Decimal('0.000')),
    'hba': lambda mol: Descriptors.NumHAcceptors(mol),
    'hbd': lambda mol: Descriptors.NumHDonors(mol),
    'clogp': lambda mol: Decimal(Descriptors.MolLogP(mol)).quantize(Decimal('0.000')),
    'rot': lambda mol: Descriptors.NumRotatableBonds(mol)
    }
}

# Define ComparableMol class with dict
class ComparableMol(object):

    def __init__(self,
                 mol, # Mol object in RDKit
                 use_original=False, # Whether to read the 5 decoy-determining parameters from smi file; read from smi file when True
                 kekulize = False, # Whether to kekulize SMILES expression; kekulize when True
                 #fp = 'MACCS',  # Representation method for molecular fingerprint
                 similarity = 'tanimoto' # Method for calculating similarity between molecules
                 ):
        self.mol = mol
        self.sim = similarity

        self.mw = calc_functs[use_original]['mw'](self.mol)
        self.hba = calc_functs[use_original]['hba'](self.mol)
        self.hbd = calc_functs[use_original]['hbd'](self.mol)
        self.clogp = calc_functs[use_original]['clogp'](self.mol)
        self.rot = calc_functs[use_original]['rot'](self.mol)

        self.fp = MACCSkeys.GenMACCSKeys(self.mol)
        self.name = self.mol.GetProp('_Name')
        self.smiles = Chem.MolToSmiles(self.mol, kekuleSmiles=kekulize)


    def __or__(self, other):
        return DataStructs.TanimotoSimilarity(self.fp, other.fp)
    
# Read in active ligands
def read_active_ligands(filename,
                        echo_error = False,
                        delimiter = ' ',
                        titleLine = True,
                        use_original_props = False):
    file = str(filename)
    active_ligands_comparable = []
    active_ligand_deposit = rdmolfiles.SmilesMolSupplier(data = file,
                                                         delimiter = delimiter,
                                                         titleLine = titleLine)

    for i, lig in enumerate(active_ligand_deposit):
        try:
            mol = ComparableMol(lig, use_original=use_original_props)
            active_ligands_comparable.append(mol)
        except Exception as e:
            if echo_error:
                print('Invalid molecule in active ligand set: line ' + str(i+1))

    print('Found %d recognizable active ligands in input file.' % (len(active_ligand_deposit)))
    print('Generated %d comparable active ligands from input file.'%(len(active_ligands_comparable)))
    return active_ligands_comparable

# Judge whether two molecules are too similar in chemical properties via Tanimoto coef.
def is_too_similar(ligand,
                   db_mol,
                   tanimoto_t = 0.75):

    if (ligand | db_mol) >= tanimoto_t:
        return True
    else:
        return False

# Judge whether two molecules are decoys
def is_decoy(ligand,
             db_mol,
             HBA_t = 2,
             HBD_t= 1,
             ClogP_t = 1.0,
             MW_t= 25,
             RB_t= 1,
             tanimoto_t = 0.75):
    ClogP_t = Decimal(ClogP_t)
    if ligand.hbd - HBD_t <= db_mol.hbd <= ligand.hbd + HBD_t:
        if ligand.mw - MW_t <= db_mol.mw <= ligand.mw + MW_t:
            if ligand.rot - RB_t <= db_mol.rot <= ligand.rot + RB_t:
                if ligand.hba - HBA_t <= db_mol.hba <= ligand.hba + HBA_t:
                    if ligand.clogp - ClogP_t <= db_mol.clogp <= ligand.clogp + ClogP_t:
                        if not is_too_similar(ligand, db_mol, tanimoto_t):
                            return True
    return False

# Generator for iterating molecules from database
def parse_database(db_dir,
                   echo_error=False,
                   delimiter=' ',
                   titleLine=True,
                   use_original_props=False,
                   do_shuffle=True
                   ):
    db_file_list = os.listdir(db_dir)
    db_file_list = [f for f in db_file_list if f.split('.')[1]=='smi']
    if do_shuffle:
        shuffle(db_file_list)
    for db_file in db_file_list:
        print('>>>Parsing decoy candidates from: ' + db_file)
        generate_mol =  rdmolfiles.SmilesMolSupplier(data = os.path.join(db_dir, db_file),
                                                 delimiter = delimiter,
                                                 titleLine = titleLine)
        for mol in generate_mol:
            try:
                cp_mol = ComparableMol(mol, use_original=use_original_props)
                yield cp_mol
            except Exception as e:
                if echo_error:
                    print('>Failed to generate CompMol from database,')
                    print('>Exception:', e)

# Convert MACCS fingerprint to numpy array
def fp2array(mol):
    fp_list = mol.fp.ToList()
    fp_arr = np.float64(np.array(fp_list[:-1]))
    return fp_arr

# Generate decoy-dict (MACCS fingerprints + SMILES)
def make_decoy_dict(decoy_set):
    fp_decoy_dict, smiles_decoy_dict = {}, {}
    for key, value in decoy_set.items():
        fp_decoy_dict[key.name] = (fp2array(key),
                                [fp2array(decoy) for decoy in value])
        smiles_decoy_dict[key.name] = (key.smiles,
                                [decoy.smiles for decoy in value])
    return fp_decoy_dict, smiles_decoy_dict

# Generate run log
def make_run_log(ligands_dict, output_log_name, output_log_dir = ''):
    df = pd.DataFrame(columns=['Name', 'MW', 'H_A', 'H_D', 'R_B', 'cLogP', 'Decoys_Found'])
    for i, mol in enumerate(ligands_dict):
        df.loc[i] = [mol.name, float(mol.mw), mol.hba, mol.hbd, mol.rot, float(mol.clogp), ligands_dict[mol]]
    # Write to csv file
    df.to_csv(os.path.join(output_log_dir, output_log_name + '.csv'), index=False)

# Write results to pkl file
def decoy_dict2pkl(decoy_dict, output_file_name, output_file_dir = ''):
    with open(os.path.join(output_file_dir, output_file_name + '.p'), 'wb') as f:
        pickle.dump(decoy_dict, f, -1)

# Iterate database
def find_decoys(active_set,
                db_generator,
                output_maccs_dir = 'maccs_dicts/',
                output_smiles_dir = 'smi_dicts/',
                output_log_dir = 'logs/',
                output_file_name = 'decoys_set',
                output_log_name = 'decoyfinder_log',
                HBA_t = 2,
                HBD_t = 1,
                ClogP_t = 1.0,
                tanimoto_t = 0.75,
                tanimoto_d = 0.9,
                MW_t = 25,
                RB_t = 1,
                mind = 36,
                unique = False,
                max_db_mols = 1000000): # max_db_mols: maximum number of db_mols to search; 0 if no restriction

    nactive_ligands = len(active_set)  # Number of active ligands provided
    total_min = nactive_ligands * mind # Number of decoys needed in total
    complete_ligand_sets = 0 # Number of active ligands with decoy set completed (at least #mind decoys)
    ndecoys = 0 # Number of decoys found in total
    ligands_max = set() # Set of active ligands with all decoys found
    #minreached = False # Indicator of whether all active ligands have enough decoys

    # Dictionary containing results; keys: active ligands names; values: decoys for each ligand in smi format
    decoy_set = {}
    ligands_dict = {}
    for mol in active_set:
        decoy_set[mol] = []
        ligands_dict[mol] = 0

    # Iterate database
    db_mol_count = 0  # Number of db_mols iterated
    for db_mol in db_generator:
        db_mol_count += 1
        if db_mol_count % 10000 == 0:
            print('Finished search for %d db_mols'%(db_mol_count))
            print('%d decoy sets finished out of %d active ligands.' %(complete_ligand_sets, nactive_ligands))
            print('%d decoys found in total; %d decoys required in total.' %(sum(ligands_dict.values()), total_min))

        if (max_db_mols > 0) & (db_mol_count > max_db_mols):
            print('Reached maximum number of db_mols. Search ends.')
            break
        if complete_ligand_sets >= nactive_ligands:
            #minreached = True
            print('All decoy sets finished for %d active ligands. Search ends.'%(nactive_ligands))
            break

        if ndecoys < total_min:
            try:
                #is_decoy_bool = [is_decoy(ligand, db_mol, tanimoto_t) for ligand in ligands_dict]
                for ligand in ligands_dict:

                    if ligand not in ligands_max:


                        if is_decoy(ligand, db_mol, HBA_t, HBD_t, ClogP_t, MW_t, RB_t, tanimoto_t):
                            too_similar_to_existing = False  # Judge whether the decoy is too similar to existing decoys for the active ligand
                            if decoy_set[ligand]:
                                for found_decoy in decoy_set[ligand]:
                                    if is_too_similar(found_decoy, db_mol, tanimoto_t=tanimoto_d):
                                        too_similar_to_existing = True
                                        break
                            if not too_similar_to_existing:
                                ligands_dict[ligand] += 1
                                decoy_set[ligand].append(db_mol)
                                ndecoys += 1
                                if ligands_dict[ligand] >= mind: # If decoy set complete for the active ligand
                                    complete_ligand_sets += 1
                                    ligands_max.add(ligand)
                                if unique:
                                    break
            except Exception as e:
                print('Search failed for db_mol %d.'%db_mol_count)
                print(e)
                continue
    else:
        print('Iterated all db_mols. Search ends.')

    # Print brief summary of results
    print('%d decoy sets finished out of %d active ligands.' %(complete_ligand_sets, nactive_ligands))
    print('%d decoys found in total; %d decoys required in total.' %(sum(ligands_dict.values()), total_min))

    # Generate results
    fp_decoy_dict, smiles_decoy_dict = make_decoy_dict(decoy_set)

    # Write log file
    #make_run_log(ligands_dict, output_log_name, output_log_dir)

    # Write results
    #decoy_dict2pkl(fp_decoy_dict, output_file_name, output_maccs_dir)
    #decoy_dict2pkl(smiles_decoy_dict, output_file_name, output_smiles_dir)

    return smiles_decoy_dict

