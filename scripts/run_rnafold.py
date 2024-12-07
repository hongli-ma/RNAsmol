import os
import pandas as pd
import numpy as np
import subprocess
import multiprocess
import re
import sys

file=sys.argv[1]



def vienna_rnafold(seq):
    command=['RNAfold','-d2']
    rna_fold=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    rna_fold_output=rna_fold.communicate(seq.encode('utf-8'))[0]
    #print(rna_fold_output)
    if len(rna_fold_output) > len(seq):
        #energy=re.findall(r"[^AUCG(-\d+]",str(rna_fold_output))
        SEC=(str(rna_fold_output).split("\\n")[1]).split(" ")[0]
    else:
        print('Error encountered while doing rnafold. Skipping...')
        #energy=None
        SEC=None
    #return energy
    return SEC

name_lst=file.rsplit('/',3)
df=pd.read_csv(file)
sec_str_lst=[]
mol_lst=[]
aff_lst=[]
seqs=df['target_sequence'].tolist()
for i in range(len(seqs)):
    sequence=seqs[i].replace('U','T').replace('(m6A)','').replace(' and ','')
    try:
        sec=vienna_rnafold(sequence)
        seq_sep=[]
        for ii in range(len(sec)):
            if sec[ii]=='.':
                seq_sep.append(sequence[ii].lower())
            else:
                seq_sep.append(sequence[ii].upper())
        sec_str_lst.append(''.join(seq_sep))
        mol_lst.append(df['compound_iso_smiles'][i])
        aff_lst.append(df['affinity'][i])
    except:
        pass
    
df_new=pd.DataFrame([mol_lst,sec_str_lst,aff_lst]).T
df_new.columns=['compound_iso_smiles','target_sequence','affinity']
outdir = name_lst[0]+'/'+name_lst[1]+'_rnafold'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    os.mkdir(outdir+'/raw')
df_new.to_csv(name_lst[0]+'/'+name_lst[1]+'_rnafold/raw/'+name_lst[3],index=False)

