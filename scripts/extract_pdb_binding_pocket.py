### Get exemplary binding site for all RNA molecules and store binding site as xyz-file
### copied and modified from https://github.com/lkmoeller/rna_binding/blob/main/rna_binding/data_splitting/binding_site_extraction.py



import logging, os, random
import numpy as np
import pandas as pd
from moleculekit.molecule import Molecule
from scipy.spatial import distance
from rna_binding.utils import (
    check_frames,
    reshape_coords,
    pickle_load,
    decompose_mol,
    get_single_lig,
    BASE_PATH,
    SITE_DIST
)



def write2xyz(mol, out_path):
    """ Generate xyz file based on molecule (mol, moleculekit object) as input
        Store xyz file under out_path
        Note: file is different from write2xyz in utils!
    """
    with open(out_path, 'w') as f_out:
        f_out.write(str(mol.numAtoms) + '\n\n')
        for a, (x,y,z) in zip(mol.element, mol.coords):
            f_out.write(str(a)+'\t'+str(x[0])+'\t'+str(y[0])+'\t'+str(z[0])+'\n')


def get_site_atoms(single_lig, rna_coords):
    """ Function to extract indices of all atoms that consitute binding site for corresponding ligand
    Args:
        single_lig (moleculekit object): ligand of interest
        rna_coords (np.array): coordinates of all rna atoms in reshaped form
    Returns:
        list with indices of all rna atoms of corresponding binding site    
    """
    # calculate distance to nearest ligand atom for all rna atoms
    single_lig_coords = reshape_coords(single_lig.coords)
    atom_dist = distance.cdist(single_lig_coords, rna_coords, 'euclidean')
    atom_dist = np.amin(atom_dist, axis=0)

    # get indices of atoms that are within site_dist of ligand
    return np.asarray(atom_dist < SITE_DIST).nonzero()[0]


def get_binding_sites(rna, lig, suitable_lig_list, suitable_res_list):
    """ Function to select and generate binding site used for clustering
    
    Args:
        rna, lig (moleculekit objects): rna and ligand objects
        suitable_lig_list (list): list of ligand ids that can be used for binding site generation
        suitable_lig_res (list): list of residue ids that can be used for binding site generation
    
    Returns:
        site_rna (moleculekit object): selected binding site
    """
    # reshape rna coordinates for analysis
    rna_coords = reshape_coords(rna.coords)

    # search for rna building up binding site
    site_index_dict = {}
    ligand_counter = {key: 0 for key in suitable_lig_list}
    ligand_res = {key: [] for key in suitable_lig_list}
    for res in suitable_res_list:
        single_lig = get_single_lig(lig, res)
        lig_name = np.unique(single_lig.resname)[0]
        ligand_counter[lig_name] += 1
        ligand_res[lig_name].append(res)
        site_index_list = get_site_atoms(single_lig, rna_coords)
        site_index_dict[res] = site_index_list

    # get binding site from most abundant ligand
    most_abundant_lig = max(ligand_counter, key=lambda key: ligand_counter[key])
    selected_res = random.choice(ligand_res[most_abundant_lig])
    site_rna = rna.copy()
    site_rna.filter('index ' + ' '.join([str(index) for index in list(set(site_index_dict[selected_res]))]))
    return site_rna



if __name__ == '__main__':

    # bsub -W 4:00 -R "rusage[mem=8192]" -n 1 -sp 90 python data_splitting/binding_site_extraction.py

    # silence warnings and infos
    logging.getLogger('moleculekit.molecule').setLevel(logging.ERROR)
    logging.getLogger('__name__').setLevel(logging.ERROR)

    # paths to pdb files
    pdb_path = os.path.join(BASE_PATH, 'data/mol/rna/pdb')

    # path to output files
    save_path = os.path.join(BASE_PATH, 'data/mol/rna/xyz')
    os.makedirs(save_path, exist_ok=True)

    # get list of rna molecules
    dataset_dict = pickle_load(os.path.join(BASE_PATH, 'data/dataset_preparation/rna/rna_result_dicts.pkl'))
    dataset_df = pd.read_csv(os.path.join(BASE_PATH, 'data/dataset_preparation/rna/rna_dataset_filter1.csv'), delimiter=',', index_col=None) 
    mol_ids = dataset_df['pdb id'].values

    link_dict = dataset_dict['link']
    ligand_dict = dataset_dict['lig_filtered']
    res_dict = dataset_dict['res_filtered']

    # loop over molecules
    for mol_id in mol_ids:
        # path to store xyz file
        path_2_save = os.path.join(save_path, mol_id + '.xyz')

        # load molecule
        mol = Molecule(os.path.join(pdb_path, mol_id + '.pdb'))
        check_frames(mol)
        
        try:
            # get site
            rna, _, lig = decompose_mol(mol, ligand_dict[mol_id], link_dict[mol_id])
            site_rna = get_binding_sites(rna, lig, ligand_dict[mol_id], res_dict[mol_id])
        
            # write to xyz file
            write2xyz(site_rna, path_2_save)
        except:
            print(mol_id)