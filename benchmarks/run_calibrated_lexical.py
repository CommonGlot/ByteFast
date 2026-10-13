#!/usr/bin/env python3
"""Compare fixed '3' lexical parameters against fitted fusion weights.

WiLI official TRAIN -> 400 train / 50 fusion calibration / 50 validation,
for each of 235 labels; untouched official TEST is scored for the chosen
settings. Prior exploratory WiLI tests have been viewed, so any gains still
need independent-domain confirmation.
"""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import statistics
import time
from run_lexical_experiment import load_labels,scores

ROOT=Path("/tmp/wili")
BASE=ROOT/"word-byte.bin"

def run(cmd, destination=None):
    if destination:
        with open(destination,"w",encoding="utf8") as f:
            subprocess.run(cmd,stdout=f,check=True)
    else:subprocess.run(cmd,check=True)

def time_it(cmd):
    out=[]
    for _ in range(3):
        start=time.perf_counter()
        with open("/dev/null","w") as sink:
            subprocess.run(cmd,stdout=sink,check=True)
        out.append(time.perf_counter()-start)
    return round(statistics.median(out),4)

def split_dev():
    grouped=defaultdict(list)
    for line in (ROOT/"lex-dev.fasttext").read_text(encoding="utf8").splitlines():
        if not line:continue
        label=line.split(" ",1)[0]
        grouped[label].append(line)
    calibration,selection=[],[]
    for label,items in sorted(grouped.items()):
        assert len(items)>=100,(label,len(items))
        calibration+=items[:50]
        selection+=items[50:100]
    def write(name,items):
        (ROOT/(name+".fasttext")).write_text(
            "\n".join(items)+"\n",encoding="utf8")
        (ROOT/(name+".txt")).write_text(
            "".join(line.split(" ",1)[1]+"\n" for line in items),
            encoding="utf8")
    write("fusion-calibration",calibration)
    write("fusion-validation",selection)
    print("FUSION_DATA split_count="+str(len(grouped))+
          " calibration="+str(len(calibration))+
          " validation="+str(len(selection)),flush=True)

def evaluate(name,cmd,goldpath,outprefix,modelsize):
    target=ROOT/(outprefix+".pred")
    run(cmd,target)
    quality=scores(load_labels(goldpath),load_labels(target))
    result={"name":name,"quality":quality,"bytes":modelsize,
            "wall_seconds_median_3":time_it(cmd)}
    return result

def main():
    split_dev()
    train=ROOT/"lex-train.fasttext"
    validation=ROOT/"fusion-validation.fasttext"
    validationTxt=ROOT/"fusion-validation.txt"
    goldtest=ROOT/"test.fasttext"
    testTxt=ROOT/"test.txt"
    records=[]
    for minimum in (1,3):
        lex=ROOT/f"words-min{minimum}.lex"
        trained=ROOT/f"words-min{minimum}.fuse"
        run(["./bytefast-lexical","build",str(BASE),str(train),
             str(lex),"200000",str(minimum)])
        run(["./bytefast-calibrated","fit",str(BASE),str(lex),
             str(ROOT/"fusion-calibration.fasttext"),
             str(validation),str(trained)])
        total=BASE.stat().st_size+lex.stat().st_size+trained.stat().st_size
        for gate in (0.5,0.75,1.0,2.0):
            pred=["./bytefast-calibrated","predict",str(BASE),str(lex),
                  str(trained),str(validationTxt),"-gate",str(gate)]
            record=evaluate(f"learned-min{minimum}-gate{gate}",pred,
                validation,f"learned-min{minimum}-gate{gate}-dev",total)
            record.update({"minimum_count":minimum,"gate":gate,
                           "lexicon_path":str(lex),"fusion_path":str(trained)})
            q=record["quality"]
            record["selection_score"]=round(
                q["macro_f1"]+.12*(q["bcms_f1"] or 0),8)
            records.append(record)
            print("LEARNED_FUSION_DEV",json.dumps(record,sort_keys=True),
                  flush=True)
    selected=max(records,key=lambda r:(r["selection_score"],
                        -r["wall_seconds_median_3"]))
    print("LEARNED_FUSION_SELECTED_ON_DEV",
          json.dumps(selected,sort_keys=True),flush=True)

    chosen=["./bytefast-calibrated","predict",str(BASE),
            selected["lexicon_path"],selected["fusion_path"],
            str(testTxt),"-gate",str(selected["gate"])]
    oldlex=ROOT/"words-min3.lex"
    old=["./bytefast-lexical","predict",str(BASE),
         str(oldlex),str(testTxt),"-weight","3.0","-gateMargin","0.75"]
    byte=["./bytefast","predict",str(BASE),str(testTxt)]
    fasttext=ROOT/"word-fasttext.bin"
    compare=[
        ("learned_fusion",chosen,selected["bytes"]),
        ("fixed_weight3_min3",old,BASE.stat().st_size+oldlex.stat().st_size),
        ("byte_only",byte,BASE.stat().st_size),
    ]
    if fasttext.exists():
        compare.append(("fasttext",["./fasttext","predict",
                       str(fasttext),str(testTxt)],fasttext.stat().st_size))
    results=[]
    for name,cmd,size in compare:
        result=evaluate(name,cmd,goldtest,f"fusion-{name}-test",size)
        results.append(result)
        print("LEARNED_FUSION_TEST",json.dumps(result,sort_keys=True),
              flush=True)
    report={"development_candidates":records,"selected":selected,
            "test":results,"external_validation_required":True,
            "note":"WiLI official-test metrics have been viewed in earlier experiments; "
                   "these new test comparisons are exploratory."}
    (ROOT/"calibrated-lexical-results.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf8")

if __name__=="__main__":main()
