"""

merge covariates together for US - relationship analogue

China's covariates come from Chen et al. 2024 (Moral attitudes towards effort and efficiency: a comparison between American and Chinese history)

Output: RelationshipFullCorpusCovariates/US/{relationship}/{version}__{domain}.csv
  columns: year, score, gdp, indi, coll, tight, loose
"""

import os
import glob
import pandas as pd

SCORES_DIR = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/RelationshipFullCorpus"
COVARIATE_FILES = {
    "US": "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/covariates_us.csv",
}
OUT_DIR = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/RelationshipFullCorpusCovariates"

VERSIONS = ["v1", "v2", "v3", "v4", "avg"]
DOMAINS = ["care", "hierarchy", "transaction", "mating"]
MIN_OVERLAP_YEARS = 15


def main():
    n_written, n_skipped = 0, 0
    for corpus in ["US"]:
        cov = pd.read_csv(COVARIATE_FILES[corpus])
        cov["year"] = cov["year"].round().astype(int)
        cov = cov.dropna()

        rel_dirs = sorted(glob.glob(os.path.join(SCORES_DIR, corpus, "*")))
        relationships = [os.path.basename(d) for d in rel_dirs if os.path.isdir(d)]

        for rel in relationships:
            out_dir = os.path.join(OUT_DIR, corpus, rel)
            os.makedirs(out_dir, exist_ok=True)

            for version in VERSIONS:
                for domain in DOMAINS:
                    src = os.path.join(SCORES_DIR, corpus, rel, f"{version}__{domain}.csv")
                    if not os.path.exists(src):
                        print(f"[MISSING] {corpus}/{rel}/{version}/{domain}")
                        n_skipped += 1
                        continue
                    df = pd.read_csv(src)[["year", "mean", "n"]].rename(columns={"mean": "score"})
                    df = df.dropna(subset=["score"])
                    df = df[df["n"] >= 5]  # drop years with too few matched sentences to trust the mean
                    merged = df.merge(cov, on="year", how="inner").sort_values("year")
                    if len(merged) < MIN_OVERLAP_YEARS:
                        print(f"[TOO SHORT] {corpus}/{rel}/{version}/{domain}: {len(merged)} overlapping years (n>=5 retained)")
                        n_skipped += 1
                        continue
                    out_path = os.path.join(out_dir, f"{version}__{domain}.csv")
                    merged[["year", "score", "gdp", "indi", "coll", "tight", "loose"]].to_csv(out_path, index=False)
                    n_written += 1
                    print(f"[OK] {corpus}/{rel}/{version}/{domain}: {len(merged)} years ({merged.year.min()}-{merged.year.max()})")

    print(f"\n[DONE] wrote {n_written} series, skipped {n_skipped}")


if __name__ == "__main__":
    main()
  