### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2024-10
### Usage: Please cite RNAsmol when you use this script

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_filter_data():
    smm_na_sequence = pd.read_csv("/path/to/SMM_Sequence_Table_hit_rates.csv")
    smm_na_sequence = smm_na_sequence[smm_na_sequence['Biomolecule Type'] == 'RNA'].reset_index(drop=True)

    smm_target_hits = pd.read_csv("/path/to/SMM_Target_Hits.csv")
    smm_target_hits.drop(columns=['KRAS_hit','MTOR_hit','MYB_hit','MYC_Pu22_hit','VEGF_hit','RB1_hit','BCL2_hit','CKIT_hit','MYCN_hit'], inplace=True)  
    return smm_na_sequence, smm_target_hits

def extract_data(target_hits, na_sequence):
    data = {'compound_iso_smiles': [], 'target_sequence': [], 'affinity': []}
    for i in range(len(target_hits)):
        for col in target_hits.columns[2:]:
            name = col.rsplit("_", 1)[0]
            ind = na_sequence['Nucleic Acid Target'].tolist().index(name)
            data['compound_iso_smiles'].append(target_hits['Smile'][i])
            data['target_sequence'].append(na_sequence["Sequence (5' to 3')"][ind].replace('U', 'T').replace('(m6A)', '').replace(' and ', ''))
            data['affinity'].append(target_hits[col][i])
    return pd.DataFrame(data).fillna(0)

def create_csv(df, output_dir, split_name):
    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    df.to_csv(f"{output_dir}/raw/data_{split_name}.csv", index=False)

def process_and_save_data(target_hits, na_sequence, output_dir):
    nohit = target_hits[(target_hits.iloc[:, 2:] == 0).all(axis=1)]
    hit = target_hits[~(target_hits.iloc[:, 2:] == 0).all(axis=1)]

    df_nohit = extract_data(nohit, na_sequence)
    df_hit = extract_data(hit, na_sequence)

    df_pos = df_hit[(df_hit['affinity'] == 1.0)].reset_index(drop=True)
    df_nohit_neg1 = df_hit[(df_hit['affinity'] == 0.0)].reset_index(drop=True)
    df_nohit_neg2 = df_nohit[(df_nohit['affinity'] == 0.0)].reset_index(drop=True)
    df_nohit_neg3 = df[(df['affinity'] == 0.0)].reset_index(drop=True)

    for i, df in enumerate([df_nohit, df_hit]):
        train, test = train_test_split(df, test_size=0.2)
        train, val = train_test_split(train, test_size=0.25)

        create_csv(train, f"{output_dir}/nohit_option{i}", "train")
        create_csv(val, f"{output_dir}/nohit_option{i}", "val")
        create_csv(test, f"{output_dir}/nohit_option{i}", "test")

    return df_pos, df_nohit_neg1, df_nohit_neg2, df_nohit_neg3

def load_negatives(file_path):
    if 'bindingdbproteinbinder' in file_path:
        return pd.read_csv(file_path)[pd.read_csv(file_path)['target'] == 0.0]['SMILES']
    elif 'chbrbb' in file_path:
        return pd.read_csv(file_path, sep=' ', header=None)[0]
    elif 'COCONUT_DB' in file_path:
        return pd.read_csv(file_path, sep=' ', header=None)[0]
    elif 'in-vitro' in file_path:
        return pd.read_csv(file_path)['SMILES']
    else:
        raise ValueError("Unsupported file type")

def write_negative_samples(df, neg_samples, output_dir, split_name):
    with open(f"{output_dir}/raw/data_{split_name}.csv", "w") as f:
        f.write("compound_iso_smiles,target_sequence,affinity\n")
        for _, row in df.iterrows():
            neg = neg_samples.sample().values[0]
            f.write(f"{row['compound_iso_smiles']},{row['target_sequence']},1\n")
            f.write(f"{neg},{row['target_sequence']},0\n")

def main():
    smm_na_sequence, smm_target_hits = load_and_filter_data()
    df_pos, df_nohit_neg1, df_nohit_neg2, df_nohit_neg3 = process_and_save_data(smm_target_hits, smm_na_sequence, "/output/path")

    # load unknown neg
    neg_files = [
        '/path/to/robinrnabinder_bindingdbproteinbinder.csv',
        '/path/to/chbrbb_p0.smi',
        '/path/to/COCONUT_DB.smi',
        '/path/to/in-vitro.smi'
    ]

    neg_samples_list = [load_negatives(file) for file in neg_files]

neg_sample_files = ['bindingdbproteinbinder','chbrbb','coconut','bioactive']

for i in range(10):  
    for j, df in enumerate([df_nohit_neg1, df_nohit_neg2, df_nohit_neg3]):
        for k, neg_samples in enumerate(neg_samples_list):
            
            neg_sample_type = neg_sample_files[k]  
            
            write_negative_samples(df, neg_samples, f"/output/path/robin_nohit_option{j}_{neg_sample_type}_{i}", "train")
            write_negative_samples(df, neg_samples, f"/output/path/robin_nohit_option{j}_{neg_sample_type}_{i}", "val")
            write_negative_samples(df, neg_samples, f"/output/path/robin_nohit_option{j}_{neg_sample_type}_{i}", "test")


main()
