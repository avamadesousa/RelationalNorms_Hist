"""
Relationship-filtered yearly mean cosine similarity to each of the 4 function defintions
(care, hierarchy, transaction, mating), plus their per-sentence average.


Instead of scoring every sentence in the corpus, this lexically filters sentences into 8 relationship categories: 
  parent_child, romantic_partners, close_friends, neighbors,
  acquaintances, boss_employee, teacher_student, doctor_patient

Output: RelationshipFullCorpus/{corpus}/{relationship}/{version}__{domain}.csv
  columns: year, mean, ci_lo, ci_hi, n   (same as full-corpus output)
"""

import os
import re
import sys
import glob
import zlib
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ANCHORS_PATH = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/anchors_multi.npz"
OUT_DIR = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings/RelationshipFullCorpus"
DOMAIN_NAMES = ["care", "hierarchy", "transaction", "mating"]
VERSIONS = ["v1", "v2", "v3", "v4", "avg"]
BATCH_SIZE = 200_000
MIN_YEARLY_N = 20
N_BOOT = 1000
BOOT_MAX_N = 5000
SEED = 42

CORPORA = {
    "US": dict(raw_dir="/local/shangchengzhao/Congress_Corpus/Segun_sentence_embeddings",
               anchor_lang="en", lang="en"),
    "Chinese": dict(raw_dir="/local/avamadesousa/PeopleDaily_v2",
                     anchor_lang="zh", lang="zh"),
}

# ----------------------------------------------------------------------
# Relationship term lists -- inclusive matching (role_a OR role_b suffices for paired-role
# relationships).
# ----------------------------------------------------------------------
RELATIONSHIP_TERMS_EN = {
    "parent_child": {
        "role_a": [r"\bparents?\b", r"\bmothers?\b", r"\bfathers?\b",
                   r"\bmoms?\b", r"\bdads?\b"],
        "role_b": [r"\bchild\b", r"\bchildren\b", r"\bsons?\b",
                   r"\bdaughters?\b", r"\bkids?\b"],
        "require_both": False,
    },
    "siblings": {
        "role_a": [r"\bbrothers?\b", r"\bsiblings?\b"],
        "role_b": [r"\bsisters?\b", r"\bsiblings?\b"],
        "require_both": False,
    },
    "romantic_partners": {
        "role_a": [r"\bhusbands?\b", r"\bwives\b", r"\bwife\b",
                   r"\bspouses?\b", r"\bboyfriends?\b", r"\bgirlfriends?\b",
                   r"\bcouples?\b", r"\bmarriages?\b", r"\bromantic\b"],
        "role_b": None, "require_both": False,
    },
    "close_friends": {
        "role_a": [r"\bfriends?\b", r"\bfriendships?\b"],
        "role_b": None, "require_both": False,
    },
    "neighbors": {
        "role_a": [r"\bneighbou?rs?\b"],
        "role_b": None, "require_both": False,
    },
    "acquaintances": {
        "role_a": [r"\bacquaintances?\b"],
        "role_b": None, "require_both": False,
    },
    "coworkers": {
        # symmetric/peer relationship, single-role pattern like
        # close_friends/neighbors/acquaintances. "colleague(s)" collides
        # with the existing "my colleagues?" congressional-courtesy
        # EXCLUDE_PATTERNS_EN entry below (applies globally), so most
        # real signal here is expected from co-worker/workmate/fellow
        # employee rather than colleague, in this corpus specifically.
        # teammate/partner deliberately left out -- sports/game-partner
        # contamination risk, not smoke-tested.
        "role_a": [r"\bco-?workers?\b", r"\bcolleagues?\b", r"\bworkmates?\b",
                   r"\bfellow (?:workers?|employees?)\b"],
        "role_b": None, "require_both": False,
    },
    "boss_employee": {
        "role_a": [r"\bbosses\b", r"\bboss\b", r"\bemployers?\b",
                   r"\bsupervisors?\b", r"\bmanagers?\b"],
        "role_b": [r"\bemployees?\b", r"\bworkers?\b", r"\bstaff\b",
                   r"\bsubordinates?\b"],
        "require_both": False,
    },
    "teacher_student": {
        "role_a": [r"\bteachers?\b", r"\bprofessors?\b", r"\binstructors?\b",
                   r"\beducators?\b"],
        "role_b": [r"\bstudents?\b", r"\bpupils?\b", r"\blearners?\b"],
        "require_both": False,
    },
    "doctor_patient": {
        "role_a": [r"\bdoctors?\b", r"\bphysicians?\b", r"\bsurgeons?\b"],
        "role_b": [r"\bpatients?\b"],
        "require_both": False,
    },
}

EXCLUDE_PATTERNS_EN = [
    r"\bmy (?:good |distinguished |dear |learned |able |old |close |very |honorable |hon\.? )?friends? (?:from|on|and|the)\b",
    r"\bthe gentle(?:man|woman|lady)\b",
    r"\bmy colleagues?\b",
    r"\bI yield\b", r"\byield (?:the floor|back|to)\b",
    r"\bour (?:democratic|republican|liberal|conservative|progressive) friends\b",
    r"\bfriends? (?:on|of) the (?:other side|committee|court|earth|bill|treaty)\b",
    r"\bfriends? (?:overseas|abroad)\b",
    r"\bfounding fathers?\b",
    r"\bfather of (?:the|our|this|his)\b",
    r"\bmother of all\b",
    r"\bcity fathers?\b",
    r"\bfatherland\b|\bmotherland\b|\bmother country\b",
    r"\bmother earth\b|\bmother nature\b",
    r"\bsister (?:city|cities|state|ship)\b",
    r"\bbrother(?:hood)?s? of\b",
    r"\bour (?:black |white |native american |american |fellow )?brothers and sisters\b",
    r"\bbrothers? (?:and|or) sisters? of (?:citizens?|aliens?|veterans?|residents?)\b",
    r"\bneighbou?ring (?:state|country|district|nation|government|people)s?\b",
    r"\bgood neighbou?r policy\b",
    r"\bhostile neighbou?rs?\b",
    r"\bneighbou?r (?:to the|on the) (?:north|south|east|west)\b",
    r"\bfriends? and neighbou?rs?\b",
    r"\ba couple of (?:weeks?|months?|years?|days?|hours?|minutes?|dozen|hundred|thousand|times)\b",
    r"\bcoupled? with\b",
    r"\bspin doctors?\b",
    r"\bdoctors? of (?:philosophy|divinity|law|science|letters)\b",
    r"\bwitch doctors?\b",
]

RELATIONSHIP_TERMS_ZH = {
    "parent_child": {"role_a": ["父母", "母亲", "妈妈", "父亲", "爸爸"],
                      "role_b": ["孩子", "子女", "儿子", "女儿"], "require_both": False},
    "siblings": {"role_a": ["兄弟", "手足", "兄弟姐妹"],
                 "role_b": ["姐妹", "手足", "兄弟姐妹"], "require_both": False},
    "romantic_partners": {"role_a": ["夫妻", "丈夫", "妻子", "太太", "情侣", "男朋友", "男友",
                                      "女朋友", "女友", "恋人", "配偶"],
                           "role_b": None, "require_both": False},
    "close_friends": {"role_a": ["朋友", "友谊", "好友", "挚友"], "role_b": None, "require_both": False},
    "neighbors": {"role_a": ["邻居", "邻里"], "role_b": None, "require_both": False},
    "acquaintances": {"role_a": ["熟人", "相识"], "role_b": None, "require_both": False},
    "coworkers": {"role_a": ["同事", "同僚"], "role_b": None, "require_both": False},  # 队友/搭档 left out, sports/game risk
    "boss_employee": {"role_a": ["老板", "上司", "雇主", "经理", "领导", "首长"],
                       "role_b": ["员工", "雇员", "下属", "工人"], "require_both": False},
    "teacher_student": {"role_a": ["老师", "教师", "教授", "导师"],
                         "role_b": ["学生", "弟子"], "require_both": False},
    "doctor_patient": {"role_a": ["医生", "大夫", "医师"], "role_b": ["病人", "患者"], "require_both": False},
}
EXCLUDE_TERMS_ZH = ["士大夫", "手足无措", "手足情"]

# Relationship-specific sentence-level exclusions: a sentence matching one of these
# is dropped from THAT relationship's matched set only (the global exclude patterns
# above apply to every relationship).
#   romantic_partners (en): the quantity phrase "a couple of" ("a couple of years"),
#     which matched \bcouples?\b without saying anything about a couple.
#   romantic_partners (zh): 老太太 ("old lady"), which matched the 太太 ("wife") term.
REL_EXCLUDE_EN = {"romantic_partners": [r"\bcouple of\b"]}
REL_EXCLUDE_ZH = {"romantic_partners": ["老太太"]}


def combined_pattern_en(pats):
    return "|".join(f"(?:{p})" for p in pats) if pats else None


def combined_pattern_zh(terms):
    return "|".join(re.escape(t) for t in terms) if terms else None


def build_patterns(lang):
    if lang == "en":
        terms, excl = RELATIONSHIP_TERMS_EN, EXCLUDE_PATTERNS_EN
        cp = combined_pattern_en
    else:
        terms, excl = RELATIONSHIP_TERMS_ZH, EXCLUDE_TERMS_ZH
        cp = combined_pattern_zh
    rel_excl = REL_EXCLUDE_EN if lang == "en" else REL_EXCLUDE_ZH
    rel_patterns = {rel: {"role_a": cp(d["role_a"]), "role_b": cp(d["role_b"]),
                           "require_both": d["require_both"],
                           "rel_exclude": cp(rel_excl.get(rel, []))}
                     for rel, d in terms.items()}
    exclude_pattern = cp(excl)
    return list(terms.keys()), rel_patterns, exclude_pattern


def lexical_mask(sent: pd.Series, rel: str, rel_patterns: dict, exclude_pattern) -> np.ndarray:
    p = rel_patterns[rel]
    a = sent.str.contains(p["role_a"], case=False, regex=True, na=False)
    if p["role_b"] is None:
        m = a
    else:
        b = sent.str.contains(p["role_b"], case=False, regex=True, na=False)
        m = (a & b) if p["require_both"] else (a | b)
    if exclude_pattern:
        m = m & ~sent.str.contains(exclude_pattern, case=False, regex=True, na=False)
    if p.get("rel_exclude"):
        m = m & ~sent.str.contains(p["rel_exclude"], case=False, regex=True, na=False)
    return m.to_numpy()


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2**31)


def mean_ci(v: np.ndarray, rng: np.random.Generator, n_boot=N_BOOT, alpha=0.05):
    n = len(v)
    if n < 2:
        return (np.nan, np.nan)
    if n > BOOT_MAX_N:
        se = v.std(ddof=1) / np.sqrt(n)
        m = v.mean()
        return (float(m - 1.96 * se), float(m + 1.96 * se))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def load_anchor_matrix(anchor_lang: str) -> np.ndarray:
    """(16, 3072): domain-major, version-minor (care_v1..v4, hier_v1..4, trans_v1..4, mating_v1..4)."""
    anchors = np.load(ANCHORS_PATH, allow_pickle=True)
    key = "en_embs" if anchor_lang == "en" else "zh_embs"
    return anchors[key].reshape(16, -1).astype(np.float32)


def fetch_embeddings_at(raw_file: str, positions: np.ndarray, expected_sentences: list) -> np.ndarray:
    """Streams the raw file once in bounded batches (avoids the int32 list-
    offset overflow on the embedding column for large year files),
    gathers only the wanted row positions via Arrow take(), and verifies
    sentence text at every fetched position against what Pass A saw."""
    positions = np.asarray(positions, dtype=np.int64)
    assert np.all(np.diff(positions) > 0), "positions must be sorted unique"

    pf = pq.ParquetFile(raw_file)
    chunks = []
    filled = 0
    offset = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["sentence", "embedding"]):
        n = batch.num_rows
        lo = int(np.searchsorted(positions, offset, side="left"))
        hi = int(np.searchsorted(positions, offset + n, side="left"))
        if hi > lo:
            local_idx = positions[lo:hi] - offset
            taken = batch.take(pa.array(local_idx, type=pa.int64()))

            sents = taken.column("sentence").to_pylist()
            for j, s in enumerate(sents):
                gi = lo + j
                if s != expected_sentences[gi]:
                    raise RuntimeError(
                        f"Alignment failure at row {int(positions[gi])} in {raw_file}: "
                        f"raw='{s[:80]}' vs pass-A='{expected_sentences[gi][:80]}'")

            flat = taken.column("embedding").flatten()
            embs = np.asarray(flat, dtype=np.float32).reshape(hi - lo, -1)
            chunks.append((lo, embs))
            filled += hi - lo
        offset += n

    if filled != len(positions):
        raise RuntimeError(f"{raw_file}: {len(positions) - filled} matched rows "
                           f"beyond raw file length.")

    out = np.empty((len(positions), chunks[0][1].shape[1]), dtype=np.float32)
    for lo, embs in chunks:
        out[lo:lo + len(embs)] = embs
    return out


CHECKPOINT_PATH_TMPL = os.path.join(OUT_DIR, "{corpus}_checkpoint.pkl")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CORPORA:
        print("usage: recompute_relationship_level.py {US|Chinese}")
        sys.exit(1)
    corpus = sys.argv[1]
    cfg = CORPORA[corpus]
    anchors = load_anchor_matrix(cfg["anchor_lang"])  # (16, 3072)
    rel_names, rel_patterns, exclude_pattern = build_patterns(cfg["lang"])

    year_files = sorted(glob.glob(os.path.join(cfg["raw_dir"], "*.parquet")))
    print(f"[INIT] {corpus}: {len(year_files)} year files, {len(rel_names)} relationships: {rel_names}", flush=True)

    # rows[rel][version][domain] = list of dict(year, mean, ci_lo, ci_hi, n)
    # Checkpointed after every year (whole `rows` dict pickled) and resumable -- this is a
    # multi-hour scan, and a checkpoint costs a fraction of a second per year against that.
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoint_path = CHECKPOINT_PATH_TMPL.format(corpus=corpus)
    done_years = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            rows = pickle.load(f)
        done_years = {d["year"] for d in rows[rel_names[0]][VERSIONS[0]][DOMAIN_NAMES[0]]}
        print(f"[RESUME] {corpus}: {len(done_years)} years already checkpointed, skipping those", flush=True)
    else:
        rows = {rel: {v: {d: [] for d in DOMAIN_NAMES} for v in VERSIONS} for rel in rel_names}

    for yf in year_files:
        year = int(os.path.basename(yf).replace(".parquet", ""))
        if year in done_years:
            continue

        # ---- Pass A: sentence-only, lexical filter every relationship ----
        pf = pq.ParquetFile(yf)
        sentences = []
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["sentence"]):
            sentences.extend(batch.column("sentence").to_pylist())
        sent_series = pd.Series(sentences)

        masks = {rel: lexical_mask(sent_series, rel, rel_patterns, exclude_pattern) for rel in rel_names}
        all_positions = np.unique(np.concatenate(
            [np.flatnonzero(masks[rel]) for rel in rel_names])) if rel_names else np.array([], dtype=np.int64)

        if len(all_positions) == 0:
            for rel in rel_names:
                for v in VERSIONS:
                    for d in DOMAIN_NAMES:
                        rows[rel][v][d].append(dict(year=year, mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0))
            with open(checkpoint_path, "wb") as f:
                pickle.dump(rows, f)
            print(f"[{corpus}] {year}: no relationship matches", flush=True)
            continue

        # ---- Pass B: fetch embeddings only at the union of matched positions ----
        expected = [sentences[i] for i in all_positions]
        all_embs = fetch_embeddings_at(yf, all_positions, expected)  # (n_union, 3072)
        raw16 = all_embs @ anchors.T                                 # (n_union, 16)
        avg4 = raw16.reshape(-1, 4, 4).mean(axis=2)                  # (n_union, 4)
        scores20 = np.concatenate([raw16, avg4], axis=1)             # (n_union, 20)
        pos_to_row = {int(p): i for i, p in enumerate(all_positions)}

        for rel in rel_names:
            rel_positions = np.flatnonzero(masks[rel])
            n = len(rel_positions)
            if n == 0:
                for v in VERSIONS:
                    for d in DOMAIN_NAMES:
                        rows[rel][v][d].append(dict(year=year, mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0))
                continue
            idx = np.array([pos_to_row[int(p)] for p in rel_positions])
            sub_scores = scores20[idx]  # (n, 20)
            rng = np.random.default_rng(stable_seed(SEED, corpus, rel, year))

            for vi, v in enumerate(VERSIONS):
                for di, d in enumerate(DOMAIN_NAMES):
                    col = 16 + di if v == "avg" else di * 4 + vi
                    vals = sub_scores[:, col].astype(np.float64)
                    lo, hi = mean_ci(vals, rng)
                    rows[rel][v][d].append(dict(year=year, mean=float(vals.mean()),
                                                 ci_lo=lo, ci_hi=hi, n=n))

        with open(checkpoint_path, "wb") as f:
            pickle.dump(rows, f)
        print(f"[{corpus}] {year} done (n_union={len(all_positions)})", flush=True)

    for rel in rel_names:
        rel_dir = os.path.join(OUT_DIR, corpus, rel)
        os.makedirs(rel_dir, exist_ok=True)
        for v in VERSIONS:
            for d in DOMAIN_NAMES:
                df = pd.DataFrame(rows[rel][v][d]).sort_values("year")
                df.to_csv(os.path.join(rel_dir, f"{v}__{d}.csv"), index=False)

    print(f"\n[DONE] {corpus}: wrote {len(rel_names)} relationships x {len(VERSIONS)} versions x {len(DOMAIN_NAMES)} domains", flush=True)


if __name__ == "__main__":
    main()
