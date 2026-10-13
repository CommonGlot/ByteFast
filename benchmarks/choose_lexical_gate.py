#!/usr/bin/env python3
"""Select a confidence gate for lexical rescoring using development data only."""
import json
from pathlib import Path
import subprocess
from run_lexical_experiment import load_labels, scores, run, timing

root=Path("/tmp/wili")
base=root/"lex-byte.bin"
lex=root/"lex-gate.bin"
dev=root/"lex-dev.txt"
test=root/"test.txt"
golddev=load_labels(root/"lex-dev.fasttext")
goldtest=load_labels(root/"test.fasttext")
cmd=["./bytefast-lexical","predict",str(base),str(lex)]

def predict(path,gate,out):
    run(cmd+[str(path),"-weight","3.0","-gateMargin",str(gate)],out)
    return load_labels(out)

rows=[]
for gate in (0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0,2.0):
    output=root/f"gate-{gate}-dev.pred"
    result=scores(golddev,predict(dev,gate,output))
    result["gate"]=gate
    result["wall_seconds_dev"]=timing(cmd+[str(dev),"-weight","3.0","-gateMargin",str(gate)])
    rows.append(result)
    print("GATE_DEV",json.dumps(result,sort_keys=True),flush=True)

full=next(r for r in rows if r["gate"]==2.0)
eligible=[r for r in rows if
    r["macro_f1"]>=full["macro_f1"]-0.003 and
    r["bcms_f1"]>=full["bcms_f1"]-0.015]
if not eligible:
    eligible=[full]
selected=min(eligible,key=lambda r:(r["wall_seconds_dev"],r["gate"]))
print("GATE_SELECTED using dev only",json.dumps(selected,sort_keys=True),flush=True)
for name,gate in (("base",0.0),("selected",selected["gate"]),("always",2.0)):
    out=root/f"gate-{name}-test.pred"
    result=scores(goldtest,predict(test,gate,out))
    result["name"]=name;result["gate"]=gate
    result["wall_seconds_median_3"]=timing(
        cmd+[str(test),"-weight","3.0","-gateMargin",str(gate)])
    print("GATE_TEST",json.dumps(result,sort_keys=True),flush=True)
