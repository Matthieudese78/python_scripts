#!/usr/bin/python3
import sys
import os
import subprocess

# Identify input and output extensions:
in_ext = sys.argv[1].lstrip(".").lower()
out_ext = sys.argv[2].lstrip(".").lower()
print(f"input file extension {in_ext}")
print(f"output file extension {out_ext}")

#
for fname in os.listdir("."):
    base = os.path.splitext(fname)[0]
    newfile = f"{base}.{out_ext}"
    print(f"output file name {newfile}")
    subprocess.run(["convert", fname, newfile])
