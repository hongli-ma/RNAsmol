### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2024-11
### Usage: Please cite RNAsmol when you use this script

### rmalign-split

import os
from collections import defaultdict
import re
import pandas as pd


path = 'rmalign-split/'  

files = [f for f in os.listdir(path) if f.endswith('.output')]

file_set = set(files)

rm_pattern = re.compile(r"RMscore\s*=\s*([0-9]*\.[0-9]+)")

score_dict = {}

for file in files:
  
    pdbid1, pdbid2 = file.split('.')[0].split('_')
    
 
    reverse_file = f"{pdbid2}_{pdbid1}.output"
    if reverse_file in file_set:
   
        try:
            with open(os.path.join(path, file), 'r') as f1, open(os.path.join(path, reverse_file), 'r') as f2:
           
                line1 = f1.readlines()[4].strip()
                line2 = f2.readlines()[4].strip()

             
                match1 = rm_pattern.search(line1)
                match2 = rm_pattern.search(line2)
                
                if match1 and match2:
                    rm_score1 = float(match1.group(1))
                    rm_score2 = float(match2.group(1))
                    
                    avg_score = (rm_score1 + rm_score2) / 2
                    score_dict[(pdbid1, pdbid2)] = avg_score
        
        except (IndexError, ValueError, FileNotFoundError) as e:
            print(f"Error processing files {file} and {reverse_file}: {e}")

df = pd.DataFrame(list(score_dict.items()), columns=['PDB Pair', 'Average RMscore'])


df_ = pd.DataFrame([(key[0], key[1], value) for key, value in score_dict.items()], columns=['pdbid1', 'pdbid2', 'Average RMscore'])


print("DataFrame columns:", df.columns)


try:
    df_matrix = df_.pivot_table(index='pdbid1', columns='pdbid2', values='Average RMscore')
    
    df_matrix.to_csv('rmscore_pairs_matrix.csv')
except KeyError as e:
    print(f"KeyError: {e}. Could not create matrix.")