#!/usr/bin/env python3
"""Generate controlled domain-shift variants of the untouched WiLI test split.

No altered examples are used for training or hyperparameter selection.
"""
from pathlib import Path
import argparse
import random

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workdir",default="/tmp/wili")
    p.add_argument("--seed",type=int,default=2741)
    args=p.parse_args()
    root=Path(args.workdir)
    samples=[]
    for line in (root/"test.fasttext").read_text(encoding="utf8").splitlines():
        label,text=line.split(" ",1)
        samples.append((label,text))
    rng=random.Random(args.seed)
    def write(name,items):
        with (root/f"robust-{name}-gold.txt").open("w",encoding="utf8") as g, \
             (root/f"robust-{name}-text.txt").open("w",encoding="utf8") as x:
            for label,txt in items:
                txt=txt.replace("\n"," ").strip()
                g.write(label+" "+txt+"\n")
                x.write(txt+"\n")
    write("original",samples)
    for size in (32,64,128):
        write(f"short{size}",[
            (l,x.encode("utf8")[:size].decode("utf8","ignore")) for l,x in samples])
    write("oddcase",[(l,"".join(c.upper() if i%2 else c.lower()
                                 for i,c in enumerate(x))) for l,x in samples])
    replacements={"a":"@","o":"0","i":"1","l":"|","e":"3","s":"5"}
    corrupted=[]
    for l,txt in samples:
        chars=list(txt)
        for j,c in enumerate(chars):
            if rng.random()<0.08:
                chars[j]=replacements.get(c.lower(),c+"~")
        corrupted.append((l,"".join(chars)))
    write("noisy8",corrupted)
    write("webprefix",[(l,"<header>Donate | Contact | Log in | Language </header> " + x)
                       for l,x in samples])
    print(f"ROBUST generated={len(samples)} cases=7",flush=True)

if __name__=="__main__":
    main()
