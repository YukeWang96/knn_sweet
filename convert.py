#!/usr/bin/env python
import os
import sys

dirpth = sys.argv[1]
outpth = sys.argv[2]
files = os.listdir(dirpth)

for fname in files:
    if "05" in fname: continue
    fp = open(os.path.join(dirpth, fname))
    fo = open(os.path.join(outpth, fname), "w")
    print(fname)
    for line in fp:
        fo.write(" ".join(line.split(",")))
