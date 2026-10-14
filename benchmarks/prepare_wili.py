#!/usr/bin/env python3
"""Prepare WiLI-2018 official splits for fastText/ByteFast.

Dataset source: https://zenodo.org/records/841984 (ODbL)
No examples from the test split are used for training.
"""
import argparse
import collections
import csv
import pathlib
import urllib.request
import zipfile

URL = "https://zenodo.org/records/841984/files/wili-2018.zip?download=1"

def find_file(root, name):
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}, found {len(matches)}")
    return matches[0]

def convert(root, split, destination, max_per_language, max_bytes):
    x = find_file(root, f"x_{split}.txt")
    y = find_file(root, f"y_{split}.txt")
    count = collections.Counter()
    seen = 0
    with x.open(encoding="utf-8") as texts, y.open(encoding="utf-8") as labels, destination.open("w",encoding="utf-8") as out:
        for text,label in zip(texts,labels):
            label = label.strip()
            text = text.strip().replace("\t"," ").replace("\n"," ")
            if not label or not text or count[label] >= max_per_language:
                continue
            # Preserve a valid Unicode string before ByteFast/fastText handle encoding.
            encoded = text.encode("utf-8")[:max_bytes] if max_bytes else text.encode("utf-8")
            text = encoded.decode("utf-8","ignore")
            if not text: continue
            out.write(f"__label__{label} {text}\n")
            count[label]+=1
            seen+=1
    print(f"{split}: {seen} examples, {len(count)} labels",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workdir",default="data/wili")
    p.add_argument("--train-per-language",type=int,default=100)
    p.add_argument("--test-per-language",type=int,default=100)
    p.add_argument("--max-bytes",type=int,default=512)
    a=p.parse_args()
    root=pathlib.Path(a.workdir)
    root.mkdir(parents=True,exist_ok=True)
    archive=root/"wili-2018.zip"
    if not archive.exists():
        print("Downloading official WiLI-2018 archive (62.4 MB)...",flush=True)
        urllib.request.urlretrieve(URL,archive)
    with zipfile.ZipFile(archive) as z:
        safe=[]
        for m in z.infolist():
            dest=(root/m.filename).resolve()
            if not dest.is_relative_to(root.resolve()):
                raise RuntimeError("Unsafe archive path")
            safe.append(m)
        z.extractall(root,members=safe)
    convert(root,"train",root/"train.fasttext",a.train_per_language,a.max_bytes)
    convert(root,"test",root/"test.fasttext",a.test_per_language,a.max_bytes)

if __name__=="__main__":
    main()
