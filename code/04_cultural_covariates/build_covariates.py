# -*- coding: utf-8 -*-
"""
Builds covariate file for the US NGram cultural-value indices, plus Maddison GDP.

China's covariates come from a separately archived source instead
(prevalence_chi.csv; from Chen et al. 2024 (Moral attitudes towards effort and efficiency: a comparison between American and Chinese history)

Design decisions:
  * US corpus  -> en-US Ngram cultural indices (data/ngram_cultural_values_us.csv,
                  the 4 index columns, 1800-2022).
  * GDP        -> Maddison 2020 (data/gdp_maddison_2020.csv, 2011 int'l $) for <=2018,
                  CHAIN-EXTENDED 2019-2022 by applying Maddison-2023 (OWID,
                  data/gdp_maddison_2023_owid.csv) year-over-year growth factors to the
                  2018 Maddison-2020 level. This avoids Maddison-2023's China 2018 level
                  rebasing (+22%), which would inject a spurious 1-year jump into a
                  differenced GDP series.

Output schema:
  year,gdp,indi,coll,tight,loose
"""
import os, shutil, time
import pandas as pd

DATA = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/RelationalNormsNLP_clean/data"
HERE = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/pipeline_ngram2022"
EMB  = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings"
TS   = time.strftime("%Y%m%d_%H%M%S")

# ---------- cultural indices ----------
en = pd.read_csv(f"{DATA}/ngram_cultural_values_us.csv")                   # year, individualism, collectivism, tightness, looseness
en = en.rename(columns={"individualism": "indi", "collectivism": "coll",
                        "tightness": "tight", "looseness": "loose"})
en = en[["year", "indi", "coll", "tight", "loose"]]

# ---------- GDP: Maddison 2020 (<=2018) + Maddison-2023 growth (2019-2022) ----------
m20 = pd.read_csv(f"{DATA}/gdp_maddison_2020.csv")
m20 = m20.rename(columns={"Entity": "entity", "Code": "code", "Year": "year",
                          "GDP per capita": "gdppc"})[["code", "year", "gdppc"]]
m23 = pd.read_csv(f"{DATA}/gdp_maddison_2023_owid.csv")[["code", "year", "gdp_per_capita"]]
m23 = m23.rename(columns={"gdp_per_capita": "gdppc"})

def gdp_series(code):
    """Maddison-2020 through 2018, then chain-extend 2019-2022 with M2023 growth."""
    s20 = m20[m20.code == code].set_index("year")["gdppc"].sort_index()
    s20 = s20[s20.index <= 2018]
    s23 = m23[m23.code == code].set_index("year")["gdppc"].sort_index()
    level = float(s20.loc[2018])
    out = s20.to_dict()
    for y in (2019, 2020, 2021, 2022):
        factor = float(s23.loc[y]) / float(s23.loc[y - 1])
        level = level * factor
        out[y] = level
    return pd.Series(out, name="gdp").rename_axis("year").reset_index()

gdp_us = gdp_series("USA")

# ---------- merge + write ----------
def build(cultural, gdp, label):
    m = gdp.merge(cultural, on="year", how="inner").sort_values("year")
    m = m[["year", "gdp", "indi", "coll", "tight", "loose"]].reset_index(drop=True)
    assert m.notna().all().all(), f"{label}: NaNs in merged covariates"
    assert m["year"].is_monotonic_increasing and not m["year"].duplicated().any()
    return m

us = build(en, gdp_us, "US")

for name, df in [("covariates_us.csv", us)]:
    dst = f"{EMB}/{name}"
    if os.path.exists(dst):
        shutil.copy2(dst, f"{dst}.bak_pre_ngram2022_{TS}")
    df.to_csv(dst, index=False)
    print(f"[OK] {name}: {len(df)} yrs {df.year.min()}-{df.year.max()}  "
          f"(backup: {name}.bak_pre_ngram2022_{TS})")
    print(df.head(2).to_string(index=False))
    print("  ...")
    print(df.tail(3).to_string(index=False))
    print()

with open(f"{HERE}/COVARIATE_PROVENANCE.txt", "w") as f:
    f.write(f"covariates_us.csv rebuilt {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("cultural indices:\n")
    f.write(f"  US : {DATA}/ngram_cultural_values_us.csv  (en-US Google Books Ngram,\n")
    f.write(f"       years 1800-2022, smoothing 3)\n\n")
    f.write("GDP per capita (2011 int'l $):\n")
    f.write(f"  <=2018 : Maddison 2020 ({DATA}/gdp_maddison_2020.csv)\n")
    f.write(f"  2019-2022 : chain-extended from the 2018 level using Maddison Project 2023\n")
    f.write(f"       year-over-year growth (OWID mirror, {DATA}/gdp_maddison_2023_owid.csv);\n")
    f.write(f"       done this way to avoid M2023's China-2018 level rebasing (+22%).\n")
print("[wrote COVARIATE_PROVENANCE.txt]")
