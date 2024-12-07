#!/bin/bash
dir_path="rmalign-split/pdb/"  
RNAalign_path="RMalign/RNAalign"


export RNAalign_path


find "$dir_path" -name "*.pdb" | while read -r file1; do
    find "$dir_path" -name "*.pdb" | while read -r file2; do
        if [[ "$file1" != "$file2" ]]; then
            echo "$file1 $file2"
            echo "$file2 $file1" 
        fi
    done
done | xargs -n 2 -P 160 bash -c '
file1="$0"
file2="$1"
base1=$(basename "$file1" .pdb)
base2=$(basename "$file2" .pdb)
"$RNAalign_path" -A "$file1" -B "$file2" -o "${base1}_${base2}.output"
'

