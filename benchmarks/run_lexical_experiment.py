#!/usr/bin/env python3
"""Run ByteFast + lexical sidecar dev selection and held-out test.

All lexical vocabulary and hyperparameter choices come exclusively from the
official WiLI train partition. The official test partition is scored once.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics
import subprocess
import time

FOCUS={"bos", "hrv", "srp", "hbs", "cnr"}

def load_labels(path):
    return [
        line.split(maxsplit=1)[0].removeprefix("__label__")
        for line in Path(path).read_text(encoding="utf8").splitlines() if line
    ]

def scores(gold,pred):
    if len(gold)!=len(pred):raise ValueError("Unmatched test lengths")
    tp=collections.Counter()
    fp=collections.Counter()
    fn=collections.Counter()
    for a,b in zip(gold,pred):
        if a==b:tp[a]+=1
        else:fp[b]+=1;fn[a]+=1
    langs=set(gold)
    f1={l: (2*tp[l]/(2*tp[l]+fp[l]+fn[l])
            if (2*tp[l]+fp[l]+fn[l]) else 0.0) for l in langs}
    focus=FOCUS & langs
    return {
       "accuracy":round(sum(tp.values())/len(gold),6),
       "macro_f1":round(sum(f1.values())/len(langs),6),
       "bcms_f1":round(sum(f1[x] for x in focus)/len(focus),6) if focus else None,
       "bcms_per_label":{x:round(f1[x],5) for x in sorted(focus)},
       "bcms_recall":{x:round(tp[x]/gold.count(x),5) for x in sorted(focus)},
    }

def run(args,out):
    with open(out,"w",encoding="utf8") as stream:
        subprocess.run(args,stdout=stream,check=True)

def timing(args):
    trials=[]
    for _ in range(3):
        with open("/dev/null","w") as sink:
            start=time.perf_counter()
            subprocess.run(args,stdout=sink,check=True)
            trials.append(time.perf_counter()-start)
    return round(statistics.median(trials),4)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workdir",default="/tmp/wili")
    a=p.parse_args()
    root=Path(a.workdir)
    model=root/"lex-byte.bin"
    exe="./bytefast-lexical"
    devtext=root/"lex-dev.txt"
    golddev=load_labels(root/"lex-dev.fasttext")
    testtext=root/"test.txt"
    goldtest=load_labels(root/"test.fasttext")
    configs=[("base",0,0)]
    base_dev=root/"lex-base-dev.pred"
    run(["./bytefast","predict",str(model),str(devtext)],base_dev)
    baseline=scores(golddev,load_labels(base_dev))
    print("DEV baseline",json.dumps(baseline,sort_keys=True),flush=True)
    chosen={"name":"base","cap":0,"alpha":0.0,"metrics":baseline}
    for cap in (50000,100000,200000):
        lex=root/f"lex-{cap}.bin"
        subprocess.run([exe,"build",str(model),str(root/"lex-train.fasttext"),
                        str(lex),str(cap),"3"],check=True)
        for weight in (0.60,1.0,1.5,2.0,3.0):
            candidate=f"words{cap}-a{weight}"
            devpred=root/f"lex-{candidate}-dev.pred"
            run([exe,"predict",str(model),str(lex),
                 str(devtext),"-weight",str(weight)],devpred)
            result=scores(golddev,load_labels(devpred))
            print("DEV lexical",candidate,json.dumps(result,sort_keys=True),flush=True)
            # BCMS priority, with a hard cap on the loss of overall quality.
            if result["macro_f1"]<baseline["macro_f1"]-0.01:
                continue
            def quality(x):
                v=x["metrics"]["bcms_f1"]
                return (v if v is not None else 0.0, x["metrics"]["macro_f1"])
            candidate_record={"name":candidate,"cap":cap,"alpha":weight,"metrics":result}
            if quality(candidate_record)>quality(chosen):
                chosen=candidate_record

    print("SELECTED using dev only",json.dumps(chosen,sort_keys=True),flush=True)
    reports=[]
    variants=[("base",["./bytefast","predict",str(model),str(testtext)],model)]
    if chosen["cap"]:
        lex=root/f"lex-{chosen['cap']}.bin"
        variants.append(("lexical",[exe,"predict",str(model),str(lex),
                    str(testtext),"-weight",str(chosen["alpha"])],lex))
    if (root/"lex-fasttext.bin").exists():
        variants.append(("fasttext",["./fasttext","predict",
                        str(root/"lex-fasttext.bin"),str(testtext)],root/"lex-fasttext.bin"))
    for name,command,modelpath in variants:
        prediction=root/f"lex-{name}-test.pred"
        run(command,prediction)
        result=scores(goldtest,load_labels(prediction))
        result["name"]=name
        result["wall_seconds_median_3"]=timing(command)
        result["bytes"]=modelpath.stat().st_size + (model.stat().st_size if name=="lexical" else 0)
        reports.append(result)
        print("TEST",json.dumps(result,sort_keys=True),flush=True)

    (root/"lexical-results.json").write_text(
        json.dumps({"selected":chosen,"test":reports},indent=2)+"\n")

if __name__=="__main__":main()
