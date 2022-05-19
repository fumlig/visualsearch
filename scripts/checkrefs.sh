#!/usr/bin/env bash

for key in $(./scripts/citekeys.py report/chapters/*.tex | uniq)
do 
    if ! grep -Fq "$key" report/references.bib
    then 
        echo "$key"
    fi
done