#!/usr/bin/env python3
"""Fit a reject rule on WiLI train-dev, evaluate on untouched WiLI test.

Reports both language accuracy and false positive rates under non-uniform
language prevalence. No model is trained on test or CommonLID examples.
"""
import argparse
from collections import Counter
import json

def scores(path, fasttext=False):
    result = []
    with open(path, encoding="utf8") as stream:
        for line in stream:
            x=line.strip().split()
            if fasttext:
                if len(x)<4:
                    raise ValueError("Expected fastText top2 predict-prob fields")
                name=x[0].removeprefix("__label__")
                confidence=float(x[1])
                margin=confidence-float(x[3])
            else:
                if len(x)<3:
                    raise ValueError("Expected ByteFast predict-scores fields")
                name=x[0].removeprefix("__label__")
                confidence=float(x[1])
                margin=float(x[2])
            result.append((name,confidence,margin))
    return result

def gold(path):
    with open(path, encoding="utf8") as stream:
        return [line.split(" ",1)[0].removeprefix("__label__").strip()
                for line in stream if line.strip()]

def accepts(rows,t,m):
    return [(name if p>=t and margin>=m and name not in {"und","zxx"}
             else "und") for name,p,margin in rows]

def fraction(n,d):
    return n/d if d else 0.0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("prefix",help="path prefix: '-dev-known.scores', etc.")
    p.add_argument("--workdir",default="/tmp/wili")
    p.add_argument("--fasttext",action="store_true")
    args=p.parse_args()
    root=args.workdir.rstrip("/")
    name=args.prefix
    def read(split, group):
        return scores(f"{root}/{name}-{split}-{group}.scores",args.fasttext)
    dk,du,dn=read("dev","known"),read("dev","unknown"),read("dev","noise")
    tk,tu,tn=read("test","known"),read("test","unknown"),read("test","noise")
    def seen(rows,t,m):
        return sum(p>=t and gap>=m and name not in {"und","zxx"} for name,p,gap in rows)
    # At least 90% validation coverage on known languages; minimize
    # open-set false acceptance. Selection only sees train-dev data.
    candidates=[]
    for t in [0,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98]:
        for m in [0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5]:
            coverage=fraction(seen(dk,t,m),len(dk))
            if coverage>=0.90:
                risk=(fraction(seen(du,t,m),len(du))+
                      fraction(seen(dn,t,m),len(dn)))/2
                candidates.append((risk,-coverage,t,m))
    if not candidates:
        raise ValueError("Cannot obtain 90% known validation coverage")
    _,_,t,m=min(candidates)

    gk=gold(f"{root}/open-test-known.txt")
    gu=gold(f"{root}/open-test-unknown.txt")
    assert len(gk)==len(tk) and len(gu)==len(tu)
    pk=accepts(tk,t,m)
    pu=accepts(tu,t,m)
    pn=accepts(tn,t,m)
    known_accept=fraction(sum(x!="und" for x in pk),len(pk))
    correct=sum(a==b for a,b in zip(gk,pk))
    accepted=sum(x!="und" for x in pk)
    all_gold=gk+gu+["zxx"]*len(tn)
    all_pred=pk+pu+pn
    fp=Counter()
    for gold_label,pred_label in zip(all_gold,all_pred):
        if pred_label!="und" and pred_label!=gold_label:
            fp[pred_label]+=1
    langs=set(gk)
    denom=len(all_gold)
    fprs=[fp[l]/max(1,denom-all_gold.count(l)) for l in langs]
    out={
        "model":name, "validation_confidence_threshold":t,
        "validation_margin_threshold":m,
        "test_known_count":len(tk), "test_unseen_count":len(tu),
        "test_noise_count":len(tn),
        "test_known_coverage":round(known_accept,6),
        "test_known_selective_accuracy":round(fraction(correct,accepted),6),
        "test_known_correct_fraction":round(fraction(correct,len(pk)),6),
        "test_unseen_language_false_accept_rate":round(fraction(sum(x!="und" for x in pu),len(pu)),6),
        "test_noise_false_accept_rate":round(fraction(sum(x!="und" for x in pn),len(pn)),6),
        "test_macro_language_fpr":round(fraction(sum(fprs),len(fprs)),7),
        "test_max_language_fpr":round(max(fprs),7),
    }
    print("OPEN_SET_RESULT",json.dumps(out,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
