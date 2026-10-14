#!/usr/bin/env python3
"""Prepare known-language and unseen-language evaluation from WiLI TRAIN only.

The official WiLI test partition is never used to choose open-set thresholds.
"""
import argparse
import collections
import hashlib
from pathlib import Path
import random

def write(path, pairs):
    with open(path, "w", encoding="utf8") as out:
        for label, text in pairs:
            out.write("__label__" + label + " " + text + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/tmp/wili")
    p.add_argument("--known", type=int, default=200)
    p.add_argument("--train-per-language", type=int, default=400)
    p.add_argument("--dev-per-language", type=int, default=50)
    p.add_argument("--seed", type=int, default=731)
    args = p.parse_args()
    root = Path(args.workdir)
    by_lang = collections.defaultdict(list)
    for line in (root/"train.fasttext").read_text(encoding="utf8").splitlines():
        label, text = line.split(" ", 1)
        by_lang[label[len("__label__"):]].append(text)
    languages = sorted(by_lang)
    random.Random(args.seed).shuffle(languages)
    known = set(languages[:args.known])
    unknown = set(languages[args.known:])
    if not unknown:
        raise ValueError("Need at least one withheld language")
    train, dev_known, dev_unknown = [], [], []
    for lang, lines in by_lang.items():
        rng = random.Random(args.seed + int.from_bytes(
            hashlib.blake2b(lang.encode(), digest_size=8).digest(), "little"))
        lines = list(lines)
        rng.shuffle(lines)
        if lang in known:
            train += [(lang, x) for x in lines[:args.train_per_language]]
            dev_known += [(lang,x) for x in
                          lines[args.train_per_language:args.train_per_language + args.dev_per_language]]
        else:
            dev_unknown += [(lang,x) for x in lines[:args.dev_per_language]]
    rng = random.Random(args.seed)
    for rows in (train,dev_known,dev_unknown):
        rng.shuffle(rows)
    write(root/"open-train.txt",train)
    write(root/"open-dev-known.txt",dev_known)
    write(root/"open-dev-unknown.txt",dev_unknown)
    # The untouched official WiLI test labels determine which examples count
    # as known and which count as open-set negatives.
    test_known,test_unknown = [],[]
    for line in (root/"test.fasttext").read_text(encoding="utf8").splitlines():
        label,text=line.split(" ",1)
        lang=label[len("__label__"):]
        (test_known if lang in known else test_unknown).append((lang,text))
    write(root/"open-test-known.txt",test_known)
    write(root/"open-test-unknown.txt",test_unknown)

    def as_plain(src,dst):
        rows=(root/src).read_text(encoding="utf8").splitlines()
        (root/dst).write_text("".join(row.split(" ",1)[1]+"\n" for row in rows),
                              encoding="utf8")
    for name in ("dev-known","dev-unknown","test-known","test-unknown"):
        as_plain("open-"+name+".txt","open-"+name+"-text.txt")

    # Controlled nonlinguistic negatives. Different seeds for dev and test.
    chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    symbols="!@#$%^&*[]{}~_-=+|/"
    for split, seed in (("dev",args.seed+100),("test",args.seed+200)):
        rg=random.Random(seed)
        noise=[]
        for i in range(500):
            n=rg.randint(24,180)
            mode=i % 4
            if mode == 0:
                t="".join(rg.choices(chars,k=n))
            elif mode == 1:
                t="".join(rg.choices(symbols+"0123456789",k=n))
            elif mode == 2:
                t="https://"+ "".join(rg.choices(chars,k=12)) + ".invalid/?id=" + \
                  "".join(rg.choices(chars+symbols,k=n))
            else:
                t="<html><script>var s='"+"".join(rg.choices(chars,k=n))+"';</script></html>"
            noise.append(t)
        (root/("open-"+split+"-noise-text.txt")).write_text("\n".join(noise)+"\n",encoding="utf8")
    print(f"OPEN known_languages={len(known)} unknown_languages={len(unknown)}"
          f" train={len(train)} dev_known={len(dev_known)}"
          f" dev_unknown={len(dev_unknown)} test_known={len(test_known)}"
          f" test_unknown={len(test_unknown)}",flush=True)

if __name__=="__main__":
    main()
