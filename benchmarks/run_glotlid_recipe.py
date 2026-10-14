#!/usr/bin/env python3
"""Compare the *exact reported GlotLID fastText recipe* on WiLI to ByteFast.

Important: the original GlotLID model was trained on a much larger 1,975+
language/script dataset. This experiment deliberately applies its parameter
recipe to the same 94,000 WiLI-2018 training documents and official test
views as earlier ByteFast experiments. It does NOT reproduce GlotLID.

The 1M-bucket × 256D fastText table consumes ~1.024 GB before its vocabulary
and output head. Keep legacy file memory budgets distinct from ByteFast's.
Official WiLI test results were viewed in previous research: exploratory.
"""
import json
from pathlib import Path
from run_lexical_experiment import load_labels, scores
from run_calibrated_lexical import split_dev
from run_short_segments import prepare_views, call, median_seconds

ROOT=Path("/tmp/wili")
TRAIN=ROOT/"lex-train.fasttext"

def main():
    split_dev()
    views=prepare_views(ROOT)

    byte=ROOT/"recipe-byte.bin"
    words=ROOT/"recipe-words.lex"
    fusion=ROOT/"recipe-weights.fuse"
    call(["./bytefast","train",str(TRAIN),str(byte),
          "-dim","32","-bucket","16384","-epoch","10",
          "-lr","0.3","-hashCount","1","-asciiFold","1"])
    call(["./bytefast-lexical","build",str(byte),str(TRAIN),
          str(words),"200000","1"])
    call(["./bytefast-calibrated","fit",str(byte),str(words),
          str(ROOT/"fusion-calibration.fasttext"),
          str(ROOT/"fusion-validation.fasttext"),str(fusion)])

    def fasttext(output,config):
        command=["./fasttext","supervised",
                 "-input",str(TRAIN),"-output",str(ROOT/output)]
        for k,v in config.items():
            command+=["-"+k,str(v)]
        command+=["-thread","2","-verbose","0"]
        print("GLOTLID_RECIPE_TRAIN",json.dumps(
            {"name":output,"config":config},sort_keys=True),flush=True)
        call(command)
        return ROOT/(output+".bin")

    reported=dict(minCount=1000,minCountLabel=0,wordNgrams=1,
                  bucket=1000000,minn=2,maxn=5,loss="softmax",
                  dim=256,epoch=1,lr=0.8)
    glotlid_recipe=fasttext("glotlid-recipe-fasttext",reported)

    # Change only minCount to measure corpus-size dependence of the
    # original '1000 occurrences' setting on this much smaller corpus.
    relaxed=fasttext("glotlid-recipe-mincount1",
                     dict(reported,minCount=1))

    # Existing tuned fastText control for same 235 labels and 400
    # training documents each, on the same runner/hardware.
    tuned=fasttext("reference-32d-fasttext",
                  dict(dim=32,bucket=16384,minn=2,maxn=5,
                       epoch=40,lr=0.3,loss="softmax",
                       minCount=1,minCountLabel=0,wordNgrams=1))

    models={
        "bytefast_byte_only":(["./bytefast","predict",str(byte)],
                               byte.stat().st_size),
        "bytefast_learned_lexical":(
            ["./bytefast-calibrated","predict",str(byte),
             str(words),str(fusion)],byte.stat().st_size+
                                        words.stat().st_size+
                                        fusion.stat().st_size),
        "glotlid_reported_recipe":(
            ["./fasttext","predict",str(glotlid_recipe)],
            glotlid_recipe.stat().st_size),
        "glotlid_recipe_minCount1":(
            ["./fasttext","predict",str(relaxed)],
            relaxed.stat().st_size),
        "fasttext_32d_tuned":(
            ["./fasttext","predict",str(tuned)],tuned.stat().st_size),
    }
    results=[]
    for view in views:
        gold=load_labels(ROOT/f"short-{view}.fasttext")
        for name,(prefix,size) in models.items():
            cmd=prefix+[str(ROOT/f"short-{view}.txt")]
            if name=="bytefast_learned_lexical":
                cmd+=["-gate","0.75"]
            out=ROOT/f"recipe-{view}-{name}.pred"
            call(cmd,out)
            predicted=load_labels(out)
            quality=scores(gold,predicted)
            record={"view":view,"model":name,"quality":quality,
                    "model_bytes":size,"documents":len(gold)}
            if view in ("full","prefix20"):
                record["wall_seconds_median_three_including_model_load"] = (
                    median_seconds(cmd))
            results.append(record)
            print("GLOTLID_RECIPE_RESULT",json.dumps(record,sort_keys=True),
                  flush=True)
    report={
        "scope":"WiLI-2018 235-label evaluation, NOT original GlotLID's 1975-class corpus",
        "official_wili_test_previously_inspected":True,
        "official_test_examples":11750,
        "train_examples":94000,
        "training_per_label":400,
        "glotlid_reported_fasttext_params":reported,
        "minCount_ablation":dict(reported,minCount=1),
        "tuned_fasttext_reference":dict(dim=32,bucket=16384,minn=2,maxn=5,
                                       epoch=40,lr=0.3,loss="softmax"),
        "results":results,
    }
    (ROOT/"glotlid-fasttext-recipe-results.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf8")
    print("GLOTLID_RECIPE_COMPLETED",len(results),flush=True)

if __name__=="__main__":
    main()
