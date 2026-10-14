#!/usr/bin/env python3
"""Score fastText-compatible label predictions: strict accuracy and macro F1."""
import argparse
from collections import Counter

p=argparse.ArgumentParser()
p.add_argument("gold")
p.add_argument("predictions")
p.add_argument("--prefix",default="__label__")
args=p.parse_args()
def labels(path):
    with open(path,encoding="utf8") as f:
        for line in f:
            if not line.strip(): continue
            yield line.split()[0].removeprefix(args.prefix)
gold=list(labels(args.gold))
pred=list(labels(args.predictions))
if len(gold)!=len(pred): raise SystemExit(f"length mismatch: gold={len(gold)} pred={len(pred)}")
tp=Counter();fp=Counter();fn=Counter();confusions=Counter()
for a,b in zip(gold,pred):
    if a==b:tp[a]+=1
    else:
        fn[a]+=1
        fp[b]+=1
        confusions[(a,b)]+=1
languages=set(gold) | set(pred)
f1={lang:(2*tp[lang]/(2*tp[lang]+fp[lang]+fn[lang]) if 2*tp[lang]+fp[lang]+fn[lang] else 0) for lang in languages}
print(f"samples={len(gold)} labels={len(languages)} accuracy={sum(tp.values())/len(gold):.6f} macro_f1={sum(f1.values())/len(f1):.6f}")
print("worst_f1:", sorted(f1.items(),key=lambda x:x[1])[:10])
print("top_confusions:", confusions.most_common(15))
