#!/usr/bin/env python3
"""Train and compare ByteFast learned word embeddings.

Official WiLI train is split 400/100 per label. The entire choice between
TF-IDF, confusion-aware vocabulary, and word-table size happens on dev only.
Official WiLI test is evaluated once for the selected word model.
"""
from pathlib import Path
import argparse
import json
import subprocess
import statistics
import time
from run_lexical_experiment import load_labels, scores

def command(args,output=None):
    if output:
        with open(output,"w",encoding="utf8") as out:
            subprocess.run(args,stdout=out,check=True)
    else:
        subprocess.run(args,check=True)

def predict(cmd,filename):
    command(cmd,filename)
    return load_labels(filename)

def walltime(cmd):
    times=[]
    for _ in range(3):
        t=time.monotonic()
        with open("/dev/null","w") as devnull:
            subprocess.run(cmd,stdout=devnull,check=True)
        times.append(time.monotonic()-t)
    return round(statistics.median(times),4)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workdir",default="/tmp/wili")
    args=p.parse_args()
    root=Path(args.workdir)
    base=root/"word-byte.bin"
    train=root/"lex-train.fasttext"
    devtext=root/"lex-dev.txt"
    devgold=load_labels(root/"lex-dev.fasttext")
    testtext=root/"test.txt"
    testgold=load_labels(root/"test.fasttext")
    candidates=[
       {"id":"tfidf40k","selection":"tfidf","cap":40000,"dim":16},
       {"id":"confusion40k","selection":"confusion","cap":40000,"dim":16},
       {"id":"confusion100k","selection":"confusion","cap":100000,"dim":16},
    ]
    print("EXPERIMENT_BASE",str(base),str(train),flush=True)
    rankings=[]
    for candidate in candidates:
        model=root/("emb-"+candidate["id"]+".bin")
        cmd=["./bytefast-wordemb","train",str(base),str(train),str(model),
             "-vocab",str(candidate["cap"]),"-dim",str(candidate["dim"]),
             "-minCount","3","-epoch","6","-lr","0.025",
             "-selection",candidate["selection"]]
        print("TRAINING_WORD_EMBEDDINGS",candidate["id"],flush=True)
        command(cmd)
        out=root/("word-"+candidate["id"]+"-dev.pred")
        predcmd=["./bytefast-wordemb","predict",str(base),str(model),str(devtext)]
        quality=scores(devgold,predict(predcmd,out))
        rec={"name":candidate["id"],"test_selection":False,
             "model":str(model),"total_bytes":base.stat().st_size+model.stat().st_size,
             "quality":quality}
        # Combined metric rewards BCMS without destroying macro F1.
        rec["selection_score"]=round(quality["macro_f1"]+
                                    0.12*quality["bcms_f1"],8)
        rankings.append(rec)
        print("DEV_WORD_EMBEDDINGS",json.dumps(rec,sort_keys=True),flush=True)
    best=max(rankings,key=lambda x:(x["selection_score"],x["quality"]["macro_f1"]))
    print("SELECTED_ON_DEV",json.dumps(best,sort_keys=True),flush=True)

    # Fair test comparison: identical train/dev/test split; word-model choice
    # is fixed before any prediction on official test examples.
    comparisons=[
        ("byte_only",["./bytefast","predict",str(base),str(testtext)],
         base.stat().st_size),
        ("learned_word_embeddings",
         ["./bytefast-wordemb","predict",str(base),best["model"],str(testtext)],
         best["total_bytes"]),
    ]
    # The previous statistical sidecar is kept as a serious competing baseline.
    sparse=root/"word-sparse.lex"
    if sparse.exists():
        comparisons.append((
          "statistical_lexicon",
          ["./bytefast-lexical","predict",str(base),str(sparse),
           str(testtext),"-weight","3.0","-gateMargin","0.75"],
          base.stat().st_size+sparse.stat().st_size
        ))
    ft=root/"word-fasttext.bin"
    if ft.exists():
        comparisons.append((
          "ordinary_fasttext",
          ["./fasttext","predict",str(ft),str(testtext)],
          ft.stat().st_size
        ))
    outcome=[]
    for name,cmd,bytes_ in comparisons:
        predfile=root/("word-"+name+"-test.pred")
        result=scores(testgold,predict(cmd,predfile))
        result["name"]=name
        result["model_bytes"]=bytes_
        result["wall_seconds_median_three"]=walltime(cmd)
        outcome.append(result)
        print("TEST_WORD_EMBEDDINGS",json.dumps(result,sort_keys=True),flush=True)
    report={"dev_candidates":rankings,"selected":best,"test":outcome}
    (root/"wordemb-results.json").write_text(json.dumps(report,indent=2)+"\n",
                                              encoding="utf8")

if __name__=="__main__":
    main()
