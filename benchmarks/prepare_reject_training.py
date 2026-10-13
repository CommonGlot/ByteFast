#!/usr/bin/env python3
"""Create open-set training with explicit und/zxx classes.

15 OOD languages provide 'und' training; 20 *other* OOD languages remain
completely unseen, with their held-out official WiLI test paragraphs.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import random

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workdir",default="/tmp/wili")
    p.add_argument("--und-languages",type=int,default=15)
    args=p.parse_args()
    root=Path(args.workdir)
    out=root/"reject"
    out.mkdir(exist_ok=True)
    train_unknown=sorted({
        s.split(" ",1)[0].removeprefix("__label__")
        for s in (root/"open-dev-unknown.txt").read_text(encoding="utf8").splitlines()
    })[:args.und_languages]
    training_set=set(train_unknown)
    if not train_unknown: raise ValueError("No unknown languages for und training")
    all_train=defaultdict(list)
    for line in (root/"train.fasttext").read_text(encoding="utf8").splitlines():
        lab,text=line.split(" ",1)
        all_train[lab.removeprefix("__label__")].append(text)
    originals=(root/"open-train.txt").read_text(encoding="utf8").splitlines()
    und=[]
    rng=random.Random(4180)
    for lab in training_set:
        vals=list(all_train[lab])
        rng.shuffle(vals)
        und.extend(f"__label__und {text}" for text in vals[:150])

    # Distinct generators/seeds from the held-out synthetic dev and test noise.
    letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    symbols="!@#$%^&*[]{}~_-=+|/"
    noise=[]
    for i in range(2000):
        n=rng.randint(24,180)
        if i%3==0: s="".join(rng.choices(letters,k=n))
        elif i%3==1: s="".join(rng.choices(symbols+"0123456789",k=n))
        else: s="<script>"+ "".join(rng.choices(letters,k=n))+"</script>"
        noise.append(f"__label__zxx {s}")
    combined=originals+und+noise
    rng.shuffle(combined)
    (out/"open-reject-train.txt").write_text("\n".join(combined)+"\n",encoding="utf8")
    (out/"open-control-train.txt").write_text("\n".join(originals)+"\n",encoding="utf8")

    for suffix in ("dev-known","test-known"):
        for ext in (".txt","-text.txt"):
            name="open-"+suffix+ext
            (out/name).write_bytes((root/name).read_bytes())
    for suffix in ("dev","test"):
        src=root/("open-"+suffix+"-unknown.txt")
        remaining=[]
        for line in src.read_text(encoding="utf8").splitlines():
            label=line.split(" ",1)[0].removeprefix("__label__")
            if label not in training_set: remaining.append(line)
        (out/("open-"+suffix+"-unknown.txt")).write_text(
            "\n".join(remaining)+"\n",encoding="utf8")
        (out/("open-"+suffix+"-unknown-text.txt")).write_text(
            "".join(line.split(" ",1)[1]+"\n" for line in remaining),encoding="utf8")
        noise_file="open-"+suffix+"-noise-text.txt"
        (out/noise_file).write_bytes((root/noise_file).read_bytes())
    print(f"REJECT train_known={len(originals)} train_und={len(und)}"
          f" train_zxx={len(noise)} train_negative_languages={len(training_set)}"
          f" unseen_evaluation_languages={len(set(all_train)-training_set)-200}",
          flush=True)

if __name__=="__main__": main()
