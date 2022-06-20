#!/usr/bin/env bash

# cross-reference bibtex reference keys between report and bibliography (error reporting is horrible and mangles line numbers?)

for key in $(./scripts/citekeys.py report/chapters/*.tex | uniq)
do 
    if ! grep -Fq "$key" report/references.bib
    then 
        echo "$key"
    fi
done