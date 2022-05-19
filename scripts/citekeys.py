#!/usr/bin/env python3

import fileinput

for line in fileinput.input():

    if not "\\cite" in line:
        continue

    for word in line.split():
        skip = word.find("\\cite")

        if skip < 0:
            continue

        word = word[skip:]
         
        begin = word.find("{") + 1
        end = word.find("}")

        keys = word[begin:end]

        for key in keys.split(","):
            print(key)