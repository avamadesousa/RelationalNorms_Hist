# Relational Norms Across History

Data and analysis code for a study of how relational norms — care, hierarchy, transaction, and
mating — have shifted over the 20th and 21st centuries in American and Chinese political discourse.
Sentence embeddings are used to score U.S. Congressional speeches and China's People's Daily
newspaper against descriptions of each cooperative function, building annual trajectories that are
then related to historical measures of individualism, collectivism, tightness, and looseness via
Granger causality and ARIMAX models.

## Contents

- **`code/`** — the analysis pipeline: corpus preprocessing, anchor-sentence embedding,
  cosine-similarity scoring, cultural-value/GDP covariates, Bayesian change point detection, and the
  Granger causality/ARIMAX analysis. See `code/README.md` for the full breakdown and
  `code/MANIFEST.tsv` for each script's exact provenance.
- **`data/`** — the external source data the covariate scripts read: U.S. and Chinese Google Books
  Ngram cultural-value indices and Maddison GDP per capita. See `data/README.md`.

## Data sources

This project builds on two prior corpora and, for China, the measurement approach that produced it:

- China's People's Daily corpus, and the "Effort"/"Efficiency" cosine-similarity approach this
  project's own scoring is methodologically modeled on: A. X. Chen, S. Sun, H. Yu, Moral attitudes
  towards effort and efficiency: a comparison between American and Chinese history. *Humanit. Soc.
  Sci. Commun.* 11, 1–14 (2024). Their own repository has related materials worth checking too.
- The U.S. Congressional Record corpus: S. T. Aroyehun, et al., Computational analysis of US
  congressional speeches reveals a shift from evidence to intuition. *Nat. Hum. Behav.* 9, 1122–1133
  (2025).
