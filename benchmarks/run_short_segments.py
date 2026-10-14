#!/usr/bin/env python3
"""ByteFast and fastText on *identical* held-out Unicode character segments.

Training: only WiLI-2018 official train, split 400/50/50 per class.
Testing: 50 official WiLI test documents per class (235 labels).
Variants include baseline trained on long text and both architectures trained
on a common short/full augmentation. No official test texts are used to train,
fit fusion weights, select lexical entries or choose a gate.

This is exploratory: previous work inspected this WiLI test split. Follow up
on an independently held-out text domain before claiming an improvement.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

from run_lexical_experiment import load_labels, scores
from run_calibrated_lexical import split_dev

LENGTHS=(5,10,20,40,80)
MID_LENGTHS=(10,20,40)
GATE="0.75"
LABEL_PREFIX="__label__"

def call(args, output=None):
    if output is None:
        subprocess.run(args,check=True)
    else:
        with open(output,"w",encoding="utf8") as f:
            subprocess.run(args,stdout=f,check=True)

def records(path):
    output=[]
    for line in path.read_text(encoding="utf8").splitlines():
        if not line:continue
        label,text=line.split(" ",1)
        if not label.startswith(LABEL_PREFIX) or not text:
            raise ValueError("Bad labeled document")
        output.append((label,text))
    return output

def crop(text,chars,view):
    # Slice Python Unicode codepoints: never split a UTF-8 byte sequence.
    if len(text)<=chars:return text
    if view=="prefix":return text[:chars].strip() or text[:chars]
    if view=="middle":
        mid=(len(text)-chars)//2
        return text[mid:mid+chars].strip() or text[mid:mid+chars]
    if view=="seeded":
        h=int.from_bytes(hashlib.blake2b(text.encode("utf8"),digest_size=8).digest(),"little")
        start=random.Random(h).randrange(len(text)-chars+1)
        return text[start:start+chars].strip() or text[start:start+chars]
    raise ValueError(view)

def write_labeled(path,rows):
    path.write_text(
        "".join(label+" "+text+"\n" for label,text in rows),
        encoding="utf8")

def write_unlabeled(path,rows):
    path.write_text("".join(text+"\n" for _,text in rows),encoding="utf8")

def prepare_views(root):
    official=records(root/"test.fasttext")
    views={"full":official}
    for n in LENGTHS:
        views[f"prefix{n}"]=[(label,crop(text,n,"prefix"))
                                  for label,text in official]
    for n in MID_LENGTHS:
        views[f"middle{n}"]=[(label,crop(text,n,"middle"))
                                  for label,text in official]
    for name,rows in views.items():
        write_labeled(root/f"short-{name}.fasttext",rows)
        write_unlabeled(root/f"short-{name}.txt",rows)
    return views

def augment_short(root):
    original=records(root/"lex-train.fasttext")
    # Equal full and 20-character crops, same 94k training source documents.
    # Use a deterministic, document-dependent internal window, rather than
    # always extracting the Wikipedia paragraph's beginning.
    output=[]
    for label,text in original:
        output.append((label,text))
        output.append((label,crop(text,20,"seeded")))
    # Preserve label initialization order shared with the lexical sidecar.
    # ByteFast shuffles training examples internally each epoch.
    write_labeled(root/"short-mixed-train.fasttext",output)
    for part in ("calibration","validation"):
        original=records(root/f"fusion-{part}.fasttext")
        mixed=[]
        for label,text in original:
            mixed.append((label,text))
            mixed.append((label,crop(text,20,"seeded")))
        write_labeled(root/f"short-mixed-{part}.fasttext",mixed)
    return len(output)

def median_seconds(cmd):
    times=[]
    for _ in range(3):
        start=time.perf_counter()
        with open("/dev/null","w") as null:
            subprocess.run(cmd,stdout=null,check=True)
        times.append(time.perf_counter()-start)
    return round(statistics.median(times),4)

def bootstrap_diff(gold,model,control,seed=9241,n=300):
    # Stratified paired bootstrap: sample official test documents WITHIN
    # each gold label, preserving equal class count and model pairing.
    by_label=defaultdict(list)
    for i,label in enumerate(gold):
        by_label[label].append(i)
    ordered=[by_label[label] for label in sorted(by_label)]
    rng=random.Random(seed)
    differences=[]
    for _ in range(n):
        ids=[rng.choice(group) for group in ordered for _ in range(len(group))]
        samplegold=[gold[i] for i in ids]
        samplemodel=[model[i] for i in ids]
        samplecontrol=[control[i] for i in ids]
        differences.append(scores(samplegold,samplemodel)["macro_f1"]-
                           scores(samplegold,samplecontrol)["macro_f1"])
    differences.sort()
    return {
      "macro_f1_difference":round(scores(gold,model)["macro_f1"]-
                                  scores(gold,control)["macro_f1"],6),
      "stratified_paired_bootstrap_95_percentile_interval":[
          round(differences[int(.025*n)],6),
          round(differences[min(int(.975*n),n-1)],6)],
      "bootstrap_replicates":n,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workdir",default="/tmp/wili")
    args=parser.parse_args()
    root=Path(args.workdir)
    assert len(records(root/"lex-train.fasttext"))==235*400
    assert len(records(root/"test.fasttext"))==235*50

    split_dev()  # Train-derived 50 calibration / 50 validation per class
    augmented_count=augment_short(root)
    views=prepare_views(root)
    print("SHORT_DATA",json.dumps({
        "n_languages":235,"train_documents":235*400,
        "mixed_train_documents":augmented_count,
        "test_documents":235*50,
        "length_unit":"Unicode codepoints (not bytes)",
        "views":list(views)},sort_keys=True),flush=True)

    base=root/"word-byte.bin"
    shortbase=root/"short-byte.bin"
    lex=root/"short-words.lex"
    fulllayer=root/"short-original.fuse"
    shortlayer=root/"short-mixed.fuse"

    # Both byte models have the same architecture: 32d, 16384 buckets,
    # 10 epochs, ASCII folding. Only training examples change.
    baseargs=["-dim","32","-bucket","16384","-epoch","10",
              "-lr","0.3","-hashCount","1","-asciiFold","1"]
    call(["./bytefast","train",str(root/"lex-train.fasttext"),str(base)]+baseargs)
    call(["./bytefast","train",str(root/"short-mixed-train.fasttext"),
          str(shortbase)]+baseargs)
    call(["./bytefast-lexical","build",str(base),
          str(root/"lex-train.fasttext"),str(lex),"200000","1"])
    # Same vocabulary for full-trained and short-trained ByteFast. Base
    # model label orders are compared to avoid silent class misalignment.
    call(["./bytefast-calibrated","fit",str(base),str(lex),
          str(root/"fusion-calibration.fasttext"),
          str(root/"fusion-validation.fasttext"),str(fulllayer)])
    call(["./bytefast-calibrated","fit",str(shortbase),str(lex),
          str(root/"short-mixed-calibration.fasttext"),
          str(root/"short-mixed-validation.fasttext"),str(shortlayer)])

    fasttext_base=root/"short-fasttext.bin"
    fasttext_mixed=root/"short-mixed-fasttext.bin"
    # Same word/subword model architecture for normal and augmented runs.
    for train_file,out in (
        (root/"lex-train.fasttext",root/"short-fasttext"),
        (root/"short-mixed-train.fasttext",root/"short-mixed-fasttext")):
        call(["./fasttext","supervised","-input",str(train_file),
            "-output",str(out),"-dim","32","-bucket","16384",
            "-epoch","40","-lr","0.3","-minn","2","-maxn","5",
            "-thread","2","-verbose","0"])

    models={
        "byte_only":[["./bytefast","predict",str(base)],base.stat().st_size],
        "byte_lexical":[["./bytefast-calibrated","predict",str(base),
           str(lex),str(fulllayer)],base.stat().st_size+lex.stat().st_size+
                                          fulllayer.stat().st_size],
        "fasttext":[["./fasttext","predict",str(fasttext_base)],
                                             fasttext_base.stat().st_size],
        "byte_mixed":[["./bytefast","predict",str(shortbase)],
                                             shortbase.stat().st_size],
        "byte_lexical_mixed":[["./bytefast-calibrated","predict",str(shortbase),
           str(lex),str(shortlayer)],shortbase.stat().st_size+lex.stat().st_size+
                                         shortlayer.stat().st_size],
        "fasttext_mixed":[["./fasttext","predict",str(fasttext_mixed)],
                                             fasttext_mixed.stat().st_size],
    }

    results=[]
    predictions={}
    for view in views:
        gold=load_labels(root/f"short-{view}.fasttext")
        predictions[view]={}
        for name,(prefix,size) in models.items():
            cmd=prefix+[str(root/f"short-{view}.txt")]
            if name.startswith("byte_lexical"):
                cmd+=["-gate",GATE]
            predfile=root/f"short-{view}-{name}.pred"
            call(cmd,predfile)
            predicted=load_labels(predfile)
            predictions[view][name]=predicted
            quality=scores(gold,predicted)
            record={"view":view,"model":name,"quality":quality,
                    "model_bytes":size,"documents":len(gold)}
            if view in ("full","prefix20"):
                record["median_wall_seconds_including_model_load"]=median_seconds(cmd)
            results.append(record)
            print("SHORT_RESULT",json.dumps(record,sort_keys=True),flush=True)
        for name in ("byte_only","byte_lexical","byte_mixed","byte_lexical_mixed"):
            delta=round(scores(gold,predictions[view][name])["macro_f1"]-
                        scores(gold,predictions[view]["fasttext"])["macro_f1"],6)
            print("SHORT_DELTA",json.dumps(
                {"view":view,"model":name,"minus_long_trained_fasttext_macro_f1":delta},
                sort_keys=True),flush=True)
    intervals=[]
    for view in ("full","prefix10","prefix20","prefix40","middle20"):
        gold=load_labels(root/f"short-{view}.fasttext")
        for model,control in (
                ("byte_lexical","fasttext"),
                ("byte_lexical_mixed","fasttext_mixed"),
                ("byte_lexical_mixed","byte_lexical")):
            d=bootstrap_diff(gold,predictions[view][model],
                             predictions[view][control],n=250)
            d.update({"view":view,"model":model,"baseline":control})
            intervals.append(d)
            print("SHORT_BOOTSTRAP",json.dumps(d,sort_keys=True),flush=True)

    report={
        "setting":"WiLI official 235 language train/test, 400 train and 50+50 dev per label",
        "test_reused_in_prior_research":True,
        "external_validation_needed":True,
        "normal_training_vs_mixed_short_training":True,
        "mixed_train_crop":"full paragraph plus 20 Unicode-codepoint window",
        "fasttext_epochs":40,"bytefast_epochs":10,"lexical_gate":.75,
        "word_selection_count_minimum":1,"lexical_vocab_cap":200000,
        "score_summary":results,"paired_bootstrap_intervals":intervals,
        "important_note":"Length = Unicode codepoints; byte n-grams remain raw UTF-8. "
                         "Full and cropped test views derive from the same test paragraphs. "
                         "Different crop lengths are not independent test samples.",
    }
    (root/"short-segment-results.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf8")

if __name__=="__main__":main()
