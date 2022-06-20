#!/usr/bin/env python3

import fileinput

labels = set()

for line in fileinput.input():
    for word in line.split():
        if word.startswith("\\label"):
            print("label:", word)
            if word in labels:
                print("duplicate:", word)

            labels.add(word)
