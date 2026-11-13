# -*- coding: utf-8 -*-
"""Additional cleaning on People's Daily: e.g.,  content filters, depublication
(no args = all years)
"""
import os, re, sys, time
import pyarrow as pa, pyarrow.parquet as pq
from collections import Counter

SRC = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/GPTembedding_congress_people_daily/PeopleDaily_clean"
DST = "/local/avamadesousa/PeopleDaily_v2"
REPORT   = os.path.join(DST, "CLEANING_REPORT.txt")
SENTINEL = os.path.join(DST, "_BUILD_COMPLETE")
os.makedirs(DST, exist_ok=True)

# ---- article-run duplicate detector ----
MIN_LEN, MIN_RUN, EXTENSION_MIN_LEN = 15, 5, 4
def _normalize(s): return re.sub(r'\s+', ' ', s.strip())
def find_duplicate_regions(rows):
    all_norm = {rn: _normalize(s) for rn, s in rows}
    rn_set = set(rn for rn, _ in rows)
    content = [(rn, all_norm[rn]) for rn, _ in rows if len(all_norm[rn]) >= MIN_LEN]
    n = len(content); positions = {}; active = {}; finished_runs = []
    for i in range(n):
        ns = content[i][1]; occ = positions.get(ns); new_active = {}; matched = set()
        if occ:
            for j in occ:
                off = i - j; pl, si = active.get(off, (0, i))
                new_active[off] = (pl + 1, si); matched.add(off)
        for off, (ln, si) in active.items():
            if off not in matched and ln >= MIN_RUN: finished_runs.append((si, i - 1, off))
        active = new_active; positions.setdefault(ns, []).append(i)
    for off, (ln, si) in active.items():
        if ln >= MIN_RUN: finished_runs.append((si, n - 1, off))
    drop = set()
    for si, ei, off in finished_runs:
        ls, es = content[si][0], content[si - off][0]
        le, ee = content[ei][0], content[ei - off][0]
        lo_l, lo_e = ls, es
        while True:
            pl, pe = lo_l - 1, lo_e - 1
            if pl not in rn_set or pe not in rn_set: break
            if len(all_norm[pl]) < EXTENSION_MIN_LEN: break
            if all_norm[pl] != all_norm[pe]: break
            lo_l, lo_e = pl, pe
        hi_l, hi_e = le, ee
        while True:
            nl, ne = hi_l + 1, hi_e + 1
            if nl not in rn_set or ne not in rn_set: break
            if len(all_norm[nl]) < EXTENSION_MIN_LEN: break
            if all_norm[nl] != all_norm[ne]: break
            hi_l, hi_e = nl, ne
        drop.update(range(lo_l, hi_l + 1))
    return sorted(drop)

# ---- content filter ----
CJK  = re.compile(r"[一-鿿]")
TERM = "。！？!?…；;"
R_CSVHDR  = re.compile(r"标题[,，]标题链接|字段\s*\d|【字号")
R_PROV    = re.compile(r"^\s*本条数据为\s*[:：]")
R_METAHEAD= re.compile(r"^\s*［\s*版名")
R_PAGE    = re.compile(r"^\s*第\s*\d+\s*版\s*[（(]?[^）)]*[）)]?\s*$")
R_COL     = re.compile(r"^\s*专栏\s*[:：]")
R_HASH    = re.compile(r"^\s*#{2,}\s?")
R_DATE    = re.compile(r"^\s*\d{4}\s*[-./年]\s*\d{1,2}\s*[-./月]\s*\d{1,2}\s*日?\s*$")
R_PHOTO   = re.compile(r"[一-鿿]{1,4}\s*(摄|绘|摄影|制图)\s*$|(新华社|本报)\s*(记者|通讯员)?[\s　]*[一-鿿]{2,4}\s*(摄|绘)\s*$|(版式设计|制图|漫画)\s*[:：]")
R_QR      = re.compile(r"扫\s*描?.{0,3}二维码|二维码.{0,4}(扫描|关注|下载)|下载.{0,8}客户端|关注.{0,8}(微信|公众号|官方微博)")
R_XREF    = re.compile(r"^\s*（?\s*(上接|下转|转第|接第|详见|参见|相关报道见|另见|见本报|见第).{0,12}版?）?\s*[\"'”’]?\s*$")
R_CONTACT = re.compile(r"^\s*(电话|传真|地址|邮编|网址|投稿|来稿请寄|报名地址|邮寄地址|电子邮箱|E-?mail)\s*[:：]|来稿请寄")
R_CTRL    = re.compile(r"[\x00-\x08\x0b-\x1f]")
R_SHORTNM = re.compile(r"^[一-鿿]{2,4}([;；、·\s　]+[一-鿿]{2,4}){0,4}\s*[;；]?\s*$")

def frac_exotic(s):
    if not s: return 0.0
    ok = 0
    for ch in s:
        o = ord(ch)
        if 0x4e00<=o<=0x9fff or 0x3400<=o<=0x4dbf or 0x20<=o<=0x7e or 0x3000<=o<=0x303f or 0xff00<=o<=0xffef: ok += 1
        elif ch in "“”‘’…—、·《》「」『』〈〉％℃　\xa0﻿": ok += 1
    return 1.0 - ok/len(s)

def clean(s):
    if s is None: return ("DROP:null", None)
    if "\t" in s: return ("DROP:table_row", None)
    t = s.replace("﻿", "").replace("\xa0", " ").strip()
    if t == "": return ("DROP:empty", None)
    if R_CTRL.search(s) or frac_exotic(t) > 0.30: return ("DROP:garbage", None)
    if R_CSVHDR.search(t): return ("DROP:csv_header", None)
    if R_PROV.match(t): return ("DROP:provenance", None)
    if '","' in t: return ("DROP:csv_fused", None)
    if R_METAHEAD.match(t): return ("DROP:meta_head", None)
    if R_HASH.match(t): return ("DROP:hash_header", None)
    if R_PAGE.match(t): return ("DROP:page_marker", None)
    if R_COL.match(t): return ("DROP:col_label", None)
    if R_XREF.match(t): return ("DROP:xref", None)
    if R_QR.search(t): return ("DROP:qr_promo", None)
    if R_CONTACT.search(t): return ("DROP:contact", None)
    if R_PHOTO.search(t): return ("DROP:photo_credit", None)
    if R_DATE.match(t): return ("DROP:bare_date", None)
    t2 = t.strip('"').strip()
    if t2 == "": return ("DROP:bare_quote", None)
    if not CJK.search(t2): return ("DROP:no_cjk", None)
    if R_SHORTNM.match(t2) and len(t2) <= 20 and t2[-1] not in TERM: return ("DROP:byline_frag", None)
    if len(t2) < 5 and t2[-1] not in TERM: return ("DROP:too_short", None)
    return ("KEEP", t2)

def process_year(year):
    src = f"{SRC}/{year}.parquet"; dst = f"{DST}/{year}.parquet"
    pf = pq.ParquetFile(src)
    n_in = pf.metadata.num_rows
    # ---- pass 1: sentence-only, decide keep/drop ----
    reasons = Counter()
    kept = []          # (global_idx, cleaned_text)
    gi = 0
    for b in pf.iter_batches(batch_size=200_000, columns=["sentence"]):
        for s in b.column("sentence").to_pylist():
            r, kt = clean(s)
            reasons["KEEP" if r == "KEEP" else r] += 1
            if kt is not None:
                kept.append((gi, kt))
            gi += 1
    n_filter = len(kept)
    # article-run dedup on survivors (positional)
    dup_pos = set(find_duplicate_regions([(i, t) for i, (_, t) in enumerate(kept)]))
    final = [(g, t) for i, (g, t) in enumerate(kept) if i not in dup_pos]
    n_out = len(final)
    keep_global = set(g for g, _ in final)
    text_by_global = dict(final)
    # ---- pass 2: stream full file, write kept rows (embeddings untouched) ----
    writer = None
    gi = 0
    for b in pf.iter_batches(batch_size=100_000, columns=["sentence", "embedding"]):
        m = b.num_rows
        local = [j for j in range(m) if (gi + j) in keep_global]
        gi += m
        if not local:
            continue
        sub = b.take(pa.array(local, type=pa.int64()))
        new_sent = pa.array([text_by_global[gi - m + j] for j in local], type=pa.string())
        tbl = pa.table({"sentence": new_sent, "embedding": sub.column("embedding")})
        if writer is None:
            writer = pq.ParquetWriter(dst, tbl.schema, compression="zstd", compression_level=5)
        writer.write_table(tbl)
    if writer is None:   # nothing survived (shouldn't happen)
        pa.parquet.write_table(pa.table({"sentence": pa.array([], pa.string()),
                                         "embedding": pa.array([], pa.list_(pa.float32()))}), dst)
    else:
        writer.close()
    drop_top = ", ".join(f"{k.split(':',1)[1]} {v:,}" for k, v in reasons.most_common(10) if k.startswith("DROP:"))
    return dict(year=year, n_in=n_in, n_filter=n_filter, n_out=n_out,
                d_filter=n_in - n_filter, d_dedup=n_filter - n_out,
                pct=round(100 * n_out / n_in, 1), top=drop_top)

def main():
    args = sys.argv[1:]
    years = [int(a) for a in args] if args else sorted(
        int(f[:4]) for f in os.listdir(SRC) if f.endswith(".parquet"))
    if not args:
        with open(REPORT, "w") as fo:
            fo.write(f"PeopleDaily_v2 build  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fo.write(f"source: {SRC}  (unmodified)\n\n")
            fo.write(f"{'year':>5} {'in':>10} {'kept':>10} {'kept%':>6} {'drop:filter':>12} {'drop:dedup':>11}   top filter drop reasons\n")
    for y in years:
        t0 = time.time(); r = process_year(y)
        line = (f"{r['year']:>5} {r['n_in']:>10,} {r['n_out']:>10,} {r['pct']:>5.1f}% "
                f"{r['d_filter']:>12,} {r['d_dedup']:>11,}   {r['top']}")
        print(line + f"   ({time.time()-t0:.0f}s)", flush=True)
        if not args:
            with open(REPORT, "a") as fo: fo.write(line + "\n")
    if not args:
        with open(SENTINEL, "w") as fo: fo.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        print("\n[BUILD COMPLETE] ->", DST, flush=True)

if __name__ == "__main__":
    main()
 