"""
Corpus-wide yearly mean cosine similarity to each of the 4 function defintions
(care, hierarchy, transaction, mating), plus their per-sentence average.

"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ANCHORS_PATH = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/anchors_multi.npz"
OUT_DIR = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/FullCorpusScores"
DOMAIN_NAMES = ["care", "hierarchy", "transaction", "mating"]
VERSIONS = ["v1", "v2", "v3", "v4", "avg"]
BATCH_SIZE = 200_000

CORPORA = {
    "US": dict(raw_dir="/local/shangchengzhao/Congress_Corpus/Segun_sentence_embeddings",
               anchor_lang="en"),
    "Chinese": dict(raw_dir="/local/avamadesousa/PeopleDaily_v2",
                     anchor_lang="zh"),
}


def load_anchor_matrix(anchor_lang: str) -> np.ndarray:
    """(16, 3072): domain-major, version-minor (care_v1..v4, hier_v1..4, trans_v1..4, mating_v1..4)."""
    anchors = np.load(ANCHORS_PATH, allow_pickle=True)
    key = "en_embs" if anchor_lang == "en" else "zh_embs"
    return anchors[key].reshape(16, -1).astype(np.float32)


def process_year(raw_file: str, anchors: np.ndarray):
    sum20 = np.zeros(20, dtype=np.float64)
    sumsq20 = np.zeros(20, dtype=np.float64)
    n_total = 0

    pf = pq.ParquetFile(raw_file)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["embedding"]):
        n = batch.num_rows
        flat = batch.column("embedding").flatten()
        embs = np.asarray(flat, dtype=np.float32).reshape(n, -1)

        raw16 = embs @ anchors.T  # (n, 16)
        avg4 = raw16.reshape(n, 4, 4).mean(axis=2)  # (n, 4)
        scores20 = np.concatenate([raw16, avg4], axis=1)  # (n, 20)

        sum20 += scores20.sum(axis=0)
        sumsq20 += (scores20.astype(np.float64) ** 2).sum(axis=0)
        n_total += n

    mean20 = sum20 / n_total
    var20 = np.clip(sumsq20 / n_total - mean20 ** 2, 1e-12, None)
    se20 = np.sqrt(var20) / np.sqrt(n_total)
    ci_lo20 = mean20 - 1.96 * se20
    ci_hi20 = mean20 + 1.96 * se20
    return mean20, ci_lo20, ci_hi20, n_total


CHECKPOINT_PATH_TMPL = os.path.join(OUT_DIR, "{corpus}_checkpoint.pkl")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CORPORA:
        print("usage: 01a_recompute_full_corpus.py {US|Chinese}")
        sys.exit(1)
    corpus = sys.argv[1]
    cfg = CORPORA[corpus]
    anchors = load_anchor_matrix(cfg["anchor_lang"])

    out_dir = os.path.join(OUT_DIR, corpus)
    os.makedirs(out_dir, exist_ok=True)

    year_files = sorted(glob.glob(os.path.join(cfg["raw_dir"], "*.parquet")))
    print(f"[INIT] {corpus}: {len(year_files)} year files, anchor_lang={cfg['anchor_lang']}", flush=True)

    # rows[version][domain] = list of dict(year, mean, ci_lo, ci_hi, n)
    # Checkpointed after every year (whole `rows` dict pickled) and resumable -- this is a
    # many-minutes-to-hours full-corpus scan, and a checkpoint costs a fraction of a second
    # per year against that.
    checkpoint_path = CHECKPOINT_PATH_TMPL.format(corpus=corpus)
    done_years = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            rows = pickle.load(f)
        done_years = {d["year"] for d in rows[VERSIONS[0]][DOMAIN_NAMES[0]]}
        print(f"[RESUME] {corpus}: {len(done_years)} years already checkpointed, skipping those", flush=True)
    else:
        rows = {v: {d: [] for d in DOMAIN_NAMES} for v in VERSIONS}

    for yf in year_files:
        year = os.path.basename(yf).replace(".parquet", "")
        if int(year) in done_years:
            continue
        try:
            mean20, ci_lo20, ci_hi20, n_total = process_year(yf, anchors)
        except Exception as e:
            print(f"[FAIL] {corpus}/{year}: {e}", flush=True)
            continue

        # column layout: [care_v1..v4, hier_v1..4, trans_v1..4, mating_v1..4, care_avg,hier_avg,trans_avg,mating_avg]
        for vi, v in enumerate(VERSIONS):
            for di, d in enumerate(DOMAIN_NAMES):
                idx = 16 + di if v == "avg" else di * 4 + vi
                rows[v][d].append(dict(year=int(year), mean=float(mean20[idx]),
                                        ci_lo=float(ci_lo20[idx]), ci_hi=float(ci_hi20[idx]),
                                        n=n_total))
        with open(checkpoint_path, "wb") as f:
            pickle.dump(rows, f)
        print(f"[{corpus}] {year} done (n={n_total})", flush=True)

    for v in VERSIONS:
        for d in DOMAIN_NAMES:
            df = pd.DataFrame(rows[v][d]).sort_values("year")
            df.to_csv(os.path.join(out_dir, f"{v}__{d}.csv"), index=False)

    print(f"\n[DONE] {corpus}", flush=True)


if __name__ == "__main__":
    main()
 