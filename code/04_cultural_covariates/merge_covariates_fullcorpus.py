"""
merge covariates together for US

China's covariates come from Chen et al. 2024 (Moral attitudes towards effort and efficiency: a comparison between American and Chinese history)

"""

import os
import pandas as pd

SCORES_DIR = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/FullCorpusScores"
COVARIATE_FILES = {
    "US": "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/covariates_us.csv",
}
OUT_DIR = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/FullCorpusTimeSeriesCovariates"

VERSIONS = ["v1", "v2", "v3", "v4", "avg"]
DOMAINS = ["care", "hierarchy", "transaction", "mating"]


def main():
    n_written, n_skipped = 0, 0
    for corpus in ["US"]:
        cov = pd.read_csv(COVARIATE_FILES[corpus])
        cov["year"] = cov["year"].round().astype(int)
        cov = cov.dropna()

        out_dir = os.path.join(OUT_DIR, corpus)
        os.makedirs(out_dir, exist_ok=True)

        for version in VERSIONS:
            for domain in DOMAINS:
                src = os.path.join(SCORES_DIR, corpus, f"{version}__{domain}.csv")
                if not os.path.exists(src):
                    print(f"[MISSING] {corpus}/{version}/{domain}")
                    n_skipped += 1
                    continue
                df = pd.read_csv(src)[["year", "mean"]].rename(columns={"mean": "score"})
                merged = df.merge(cov, on="year", how="inner").sort_values("year")
                if len(merged) < 15:
                    print(f"[TOO SHORT] {corpus}/{version}/{domain}: {len(merged)} overlapping years")
                    n_skipped += 1
                    continue
                out_path = os.path.join(out_dir, f"{version}__{domain}.csv")
                merged.to_csv(out_path, index=False)
                n_written += 1
                print(f"[OK] {corpus}/{version}/{domain}: {len(merged)} years ({merged.year.min()}-{merged.year.max()})")

    print(f"\n[DONE] wrote {n_written} series, skipped {n_skipped}")


if __name__ == "__main__":
    main()
 