# Relational Norms Across History

Data and analysis code for a study of how discourse surrounding 4 coopeartive functions
(care, hierarchy, transaction, and mating) (see Earp et al., 2021, 2026; Bugental 2000) have shifted
overall, and in the context of specific relationships (i.e. relational norms),
over the 20th and 21st centuries in American and Chinese political discourse.
Sentence embeddings are used to score U.S. Congressional speeches and China's People's Daily
newspaper against descriptions of each cooperative function, building annual trajectories that are
then related to historical measures of individualism, collectivism, tightness, and looseness with
Granger causality and ARIMAX models.

## Contents

- **`code/`** — the analysis pipeline: corpus preprocessing, anchor-sentence embedding,
  cosine-similarity scoring, cultural-value/GDP covariates, Bayesian change point detection, and the
  Granger causality/ARIMAX analysis. 
- **`data/`** — the external source data the covariate scripts read: U.S. and Chinese Google Books
  Ngram cultural-value indices and Maddison GDP per capita. 

## Data sources

This project builds on two prior corpora and, for China, the measurement approach that produced it:

- China's People's Daily corpus, and the "Effort"/"Efficiency" cosine-similarity approach this
  project's own scoring is methodologically modeled on: A. X. Chen, S. Sun, H. Yu, Moral attitudes
  towards effort and efficiency: a comparison between American and Chinese history. *Humanit. Soc.
  Sci. Commun.* 11, 1–14 (2024). Their own repository has related materials worth checking too.
- The U.S. Congressional Record corpus: S. T. Aroyehun, et al., Computational analysis of US
  congressional speeches reveals a shift from evidence to intuition. *Nat. Hum. Behav.* 9, 1122–1133
  (2025).
