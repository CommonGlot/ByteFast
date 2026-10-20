#!/usr/bin/env python3
"""Per-label lexicons of correct words, built from the GlotLID corpus itself.

For the percent-threshold check (Caswell et al. 2020): a line keeps its label
only if enough of its words are known words of that label. The lists come
from our training lines only: each label's lines are split exactly like
prepare.py (same dedup, seed and hold-out), so dev and test lines
never enter a lexicon. A word is kept for a label when

1. it is in at least --min-lines training lines of the label (CommonLID
   normalization: lowercase, no digits or punctuation);
2. at least --min-script of its letters are in the label's script;
3. it is in at least two sources when the label has two or more sources with
   --min-source-lines lines (one source's boilerplate is not the language);
4. it is among the most frequent words that cover --coverage of the label's
   word occurrences (at most --max-words);
5. it is not a stray of a big language: for each of the --big labels with the
   most lines, the word is dropped if it is --stray-ratio times more frequent
   (per line) in that big label than in this one.

Writes OUT/lexicon.tsv (label, word, lines, sources), OUT/stats.tsv, and
OUT/lexicon.train.tsv for ByteFast's -lexicon: the same lines without the
labels written without spaces between words (a median of one word or fewer
per line), whose lines cannot be checked word by word.

  python recipes/lid/build_lexicon.py --corpus data/glotlid-corpus --out data/lexicon --workers 44
"""
import argparse
import random
import sys
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import regex

from prepare import label_files

NONWORD = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
LETTER = regex.compile(r"\p{L}")
# Scripts written with several Unicode scripts.
SCRIPTS = {"Jpan": ["Han", "Hiragana", "Katakana"], "Kore": ["Hangul", "Han"],
           "Hans": ["Han"], "Hant": ["Han"], "Hani": ["Han"]}
NO_SCRIPT = {"Zyyy", "Zzzz", "Zxxx", "Zmth", "Zsym", "Code", "Brai"}
ARGS = None


def init_worker(args):
    global ARGS
    ARGS = args


def script_pattern(code):
    if code in NO_SCRIPT:
        return None
    names = SCRIPTS.get(code, [code])
    try:
        return regex.compile("|".join(r"\p{Script=%s}" % n for n in names))
    except regex.error:
        return None


def train_lines(label, paths):
    """prepare.split's training lines, each with its source name."""
    seen, lines = set(), []
    prefix = label + "_"
    for path in paths:
        source = path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                text = " ".join(line.split())
                if text and text not in seen:
                    seen.add(text)
                    lines.append((text, source))
    rng = random.Random(ARGS.seed ^ zlib.crc32(label.encode()))
    rng.shuffle(lines)
    n = len(lines)
    hold = min(round(n * ARGS.holdout), n - 1) if n > 1 else 0
    return lines[hold:]


def count_label(item):
    label, paths = item
    lines = train_lines(label, paths)[:ARGS.max_lines]
    pattern = script_pattern(label.rsplit("_", 1)[-1])
    df, sources, per_source, tokens = Counter(), defaultdict(set), Counter(), []
    occurrences = Counter()
    for text, source in lines:
        words = NONWORD.sub("", text.lower()).split()
        tokens.append(len(words))
        per_source[source] += 1
        occurrences.update(words)
        for w in set(words):
            df[w] += 1
            if len(sources[w]) < 2:
                sources[w].add(source)
    n = max(len(lines), 1)
    multi = sum(1 for c in per_source.values() if c >= ARGS.min_source_lines) >= 2
    total = sum(occurrences.values()) or 1
    kept, covered = [], 0
    for w, c in occurrences.most_common():
        if covered / total >= ARGS.coverage or len(kept) >= ARGS.max_words:
            break
        covered += c
        if df[w] < ARGS.min_lines or (multi and len(sources[w]) < 2):
            continue
        if pattern is not None:
            letters = LETTER.findall(w)
            if not letters or sum(1 for ch in letters if pattern.match(ch)) < ARGS.min_script * len(letters):
                continue
        kept.append((w, df[w], len(sources[w])))
    tokens.sort()
    stats = dict(lines=len(lines), sources=len(per_source), multi=multi,
                 median_tokens=tokens[len(tokens) // 2] if tokens else 0)
    # Per-line rates of the big labels' words, for the stray check.
    rates = {w: c / n for w, c in df.most_common(ARGS.big_words)} if label in ARGS.big_labels else {}
    return label, kept, stats, n, rates


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--versions", nargs="+", default=["data"])
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-lines", type=int, default=300000)
    p.add_argument("--min-lines", type=int, default=3)
    p.add_argument("--min-script", type=float, default=0.5)
    p.add_argument("--min-source-lines", type=int, default=50)
    p.add_argument("--coverage", type=float, default=0.97)
    p.add_argument("--max-words", type=int, default=10000)
    p.add_argument("--big", type=int, default=40)
    p.add_argument("--big-words", type=int, default=200000)
    p.add_argument("--stray-ratio", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=24)
    ARGS = p.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)

    files = label_files(ARGS.corpus, ARGS.versions)
    print("labels found:", len(files), file=sys.stderr, flush=True)
    # Big labels by bytes on disk (close enough to lines for the top few).
    disk = {label: sum(p.stat().st_size for p in paths) for label, paths in files.items()}
    ARGS.big_labels = set(sorted(disk, key=disk.get, reverse=True)[:ARGS.big])
    lexicon, stats, size, rates = {}, {}, {}, {}
    with Pool(ARGS.workers, init_worker, (ARGS,)) as pool:
        for i, (label, kept, st, n, r) in enumerate(
                pool.imap_unordered(count_label, sorted(files.items()), chunksize=1)):
            lexicon[label], stats[label], size[label], rates[label] = kept, st, n, r
            if (i + 1) % 200 == 0:
                print("%d/%d labels" % (i + 1, len(files)), file=sys.stderr, flush=True)
    big = sorted(ARGS.big_labels)
    big_rates = {b: rates[b] for b in big}
    rates = None
    with open(ARGS.out / "lexicon.tsv", "w", encoding="utf-8") as out, \
            open(ARGS.out / "lexicon.train.tsv", "w", encoding="utf-8") as train_out, \
            open(ARGS.out / "stats.tsv", "w", encoding="utf-8") as st_out:
        st_out.write("label\tlines\tsources\tmulti_source\tmedian_tokens\twords\tstrays\n")
        for label in sorted(lexicon):
            n = max(size[label], 1)
            words, strays = [], 0
            for w, df, src in lexicon[label]:
                rate = df / n
                if any(b != label and big_rates[b].get(w, 0.0) > ARGS.stray_ratio * rate
                       for b in big):
                    strays += 1
                    continue
                words.append((w, df, src))
                row = "%s\t%s\t%d\t%d\n" % (label, w, df, src)
                out.write(row)
                if stats[label]["median_tokens"] > 1:
                    train_out.write(row)
            s = stats[label]
            st_out.write("%s\t%d\t%d\t%d\t%d\t%d\t%d\n" % (
                label, s["lines"], s["sources"], s["multi"], s["median_tokens"], len(words), strays))
    print("big labels:", " ".join(big), file=sys.stderr)


if __name__ == "__main__":
    main()
