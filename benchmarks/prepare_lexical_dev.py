#!/usr/bin/env python3
"""Create a train/development split exclusively from WiLI's official train split."""
from pathlib import Path
import argparse
from collections import defaultdict
import random

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workdir",default="/tmp/wili")
    p.add_argument("--train-count",type=int,default=400)
    p.add_argument("--seed",type=int,default=97234)
    a=p.parse_args()
    root=Path(a.workdir)
    groups=defaultdict(list)
    for line in (root/"train.fasttext").read_text(encoding="utf8").splitlines():
        label,text=line.split(" ",1)
        groups[label].append((label,text))
    train=[];dev=[]
    for label,examples in groups.items():
        rng=random.Random(a.seed+sum(ord(c) for c in label))
        rng.shuffle(examples)
        train+=examples[:a.train_count]
        dev+=examples[a.train_count:]
    random.Random(a.seed).shuffle(train)
    random.Random(a.seed+1).shuffle(dev)
    def write(name,data):
        (root/(name+".fasttext")).write_text(
           "".join(label+" "+text+"\n" for label,text in data),encoding="utf8")
        (root/(name+".txt")).write_text(
           "".join(text+"\n" for label,text in data),encoding="utf8")
    write("lex-train",train);write("lex-dev",dev)
    for split in ["test"]:
        lines=(root/(split+".fasttext")).read_text(encoding="utf8").splitlines()
        (root/(split+".txt")).write_text(
            "".join(line.split(" ",1)[1]+"\n" for line in lines),encoding="utf8")
    print("LEXICAL_SPLIT train=",len(train),"dev=",len(dev),"labels=",len(groups))
if __name__=="__main__": main()
