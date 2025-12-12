#!/usr/bin/env python3
"""Standalone BEIR dataset validator (no package imports).

Parses corpus.jsonl, queries.jsonl and qrels/<split>.tsv and prints
basic counts, samples and simple consistency checks. Exits with
non-zero status for parsing/format errors.
"""

import argparse
import csv
import json
import os
import pathlib
import random
import sys


def load_jsonl(path, id_keys=("_id", "id"), text_keys=("text", "contents", "body", "passage", "question", "query")):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")

            doc_id = None
            for k in id_keys:
                if k in obj:
                    doc_id = obj[k]
                    break
            if not doc_id:
                raise ValueError(f"No id ('_id' or 'id') found in JSON object on line {i} of {path}: {obj}")

            # Build a small normalized dict but keep original keys too
            data = {k: obj.get(k) for k in text_keys if k in obj}
            data.update(obj)
            items[str(doc_id)] = data
    return items


def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, cols in enumerate(reader, 1):
            if not cols:
                continue
            if len(cols) < 3:
                raise ValueError(f"Invalid qrels line {i} in {path}: expected at least 3 columns, got {len(cols)}: {cols}")
            qid = cols[0].strip()
            docid = cols[1].strip()
            score = cols[2].strip()
            try:
                # allow floats that are integers like "1.0"
                score_int = int(float(score))
            except Exception:
                raise ValueError(f"Invalid relevance score on line {i} in {path}: {score}")
            qrels.setdefault(qid, {})[docid] = score_int
    return qrels


def sample_keys(d, n=1):
    keys = list(d.keys())
    if not keys:
        return []
    n = min(n, len(keys))
    return random.sample(keys, n)


def main(data_path: str, split: str):
    data_path = pathlib.Path(data_path)
    corpus_file = data_path / "corpus.jsonl"
    queries_file = data_path / "queries.jsonl"
    qrels_file = data_path / "qrels" / f"{split}.tsv"

    missing = []
    for p in (corpus_file, queries_file, qrels_file):
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: missing expected files:")
        for m in missing:
            print("  ", m)
        return 2

    try:
        corpus = load_jsonl(corpus_file)
        queries = load_jsonl(queries_file)
        qrels = load_qrels(qrels_file)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 1

    print(f"corpus size: {len(corpus)}")
    print(f"queries size: {len(queries)}")
    print(f"qrels size: {len(qrels)}")

    # show a sample doc
    for docid in sample_keys(corpus, 1):
        doc = corpus[docid]
        print("\nSample doc id:", docid)
        print("Fields:", list(doc.keys()))
        title = doc.get("title", "")
        text = doc.get("text") or doc.get("contents") or doc.get("body") or doc.get("passage") or ""
        print("Title:", title)
        print("Text (first 400 chars):", text[:400])

    # show a sample query
    for qid in sample_keys(queries, 1):
        q = queries[qid]
        qtext = q.get("text") or q.get("question") or q.get("query") or ""
        print("\nSample query id:", qid)
        print("Query text:", qtext)
        print("qrels for sample_q:", qrels.get(qid, {}))

    # Check qrels references
    missing_queries = [q for q in qrels.keys() if q not in queries]
    missing_docs = set()
    for q, docs in qrels.items():
        for d in docs.keys():
            if d not in corpus:
                missing_docs.add(d)

    if missing_queries:
        print("\nWARNING: qrels reference query ids not present in queries.jsonl:", missing_queries)
    if missing_docs:
        print("\nWARNING: qrels reference doc ids not present in corpus.jsonl:", list(missing_docs))

    print("\nValidation completed successfully.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate BEIR-style dataset folder (JSONL + qrels)')
    parser.add_argument('data_path', nargs='?', default=os.path.join('beir', 'datasets', 'toy_dataset'), help='Path to dataset folder')
    parser.add_argument('--split', '-s', default='test', help='qrels split to read (default: test)')
    args = parser.parse_args()
    sys.exit(main(args.data_path, args.split))
