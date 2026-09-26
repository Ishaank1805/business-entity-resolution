#!/usr/bin/env python3
"""Learned entity resolution (no hand-written normalization rules).

  prep     read TSVs; text = raw "name, address"; split train S1 50/50 into A (encoder) / B (decision model)
  train    fine-tune the encoder (default Qwen/Qwen3-Embedding-0.6B, LoRA) on A's (S1, copy) pairs,
           contrastive loss with negatives shared across the 4 GPUs (DDP); --epochs passes over the sample
           (default 3; each epoch reshuffles each GPU's own batch order, batches never move between GPUs)
  embed    encode every record of train and test (4 GPUs)
  search   for every S2/S3 record: top-k most similar S1 of the same country (exact GPU search)
  fit      recall report on B; LightGBM on B's candidates (similarity, rank, margins, competition, generic
           string similarities); threshold tuned for macro per-S1 F0.5 with one S1 per record
  predict  test -> output/matching_results.tsv + output/candidate_pairs.tsv
  all      everything above in order

Needs: pandas, numpy, scikit-learn, rapidfuzz, lightgbm, torch, transformers, peft, anyascii, tqdm.

Run on the GPU node, e.g.:
    python learned_er.py all --data dataset --work /scratch/ishaan.karan/er_work --out output
(--data = folder that contains train/ and test/)
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

T0 = time.time()
INSTR = "Given a business record (name and address), retrieve records of the same business at the same location"


def is_decoder(name: str) -> bool:
    """Qwen3-Embedding = decoder (last-token pooling, left padding, instruction); e5 = encoder (mean pooling)."""
    return "qwen" in name.lower()


def fmt(texts, dec):
    out = []
    for t in texts:
        t = t[7:] if t.startswith("query: ") else t
        out.append(f"Instruct: {INSTR}\nQuery:{t}" if dec else "query: " + t)
    return out


def load_tok(path, dec):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, padding_side="left" if dec else "right")


def load_backbone(path, dec, dtype):
    from transformers import AutoModel
    return AutoModel.from_pretrained(path, torch_dtype=dtype, **({} if dec else {"add_pooling_layer": False}))


def pool(out, mask, dec):
    if dec:  # left padding: the last position is every row's final real token
        return out[:, -1]
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1.0)


def check_fp16(a, path, dec) -> bool:
    """2080 Ti has no bf16. Compare fp16 vs fp32 embeddings on 64 real records; fall back to fp32 if they differ."""
    import torch
    import torch.nn.functional as F
    texts = fmt(pd.read_parquet(wp(a, "train_recs.parquet"), columns=["text"]).text.sample(64, random_state=0).tolist(), dec)
    tok = load_tok(path, dec)
    m = load_backbone(path, dec, torch.float32).to("cuda:0").eval()
    enc = tok(texts, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        e32 = F.normalize(pool(m(**enc).last_hidden_state.float(), enc["attention_mask"], dec), dim=-1)
        m = m.half()
        e16 = F.normalize(pool(m(**enc).last_hidden_state.float(), enc["attention_mask"], dec), dim=-1)
    finite = bool(torch.isfinite(e16).all().item())
    cos = float((e32 * e16).sum(-1).min().item()) if finite else float("nan")
    ok = finite and cos > 0.98
    log(f"fp16 check on {path}: finite={finite}, min cosine fp16 vs fp32={cos:.4f} -> using {'fp16' if ok else 'fp32 (slower)'}")
    del m
    torch.cuda.empty_cache()
    return ok


def model_info(a):
    """(path, is_decoder) of the model used by embed/search."""
    if a.model == "base":
        return a.base_model, is_decoder(a.base_model)
    with open(wp(a, os.path.join("model", "er_meta.json"))) as f:
        return wp(a, "model"), json.load(f)["decoder"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def wp(a, name):
    os.makedirs(a.work, exist_ok=True)
    return os.path.join(a.work, name)


def read_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, on_bad_lines="warn")
    df.columns = [c.strip() for c in df.columns]
    return df


# =====================================================================================================================
# prep
# =====================================================================================================================
def cmd_prep(a):
    for split in ("train", "test"):
        frames = []
        for s in tqdm((1, 2, 3), desc=f"prep {split}: reading sources"):
            df = read_tsv(f"{a.data}/{split}/{split}_source{s}.tsv")
            df = df.rename(columns={"business_name": "name", "business_address": "addr"})
            df["src"] = np.int8(s)
            frames.append(df[["entity_id", "name", "addr", "country", "src"]])
        df = pd.concat(frames, ignore_index=True)
        df["country"] = df.country.str.strip().str.lower()
        df["text"] = df.name.str.strip() + ", " + df.addr.str.strip()
        df.to_parquet(wp(a, f"{split}_recs.parquet"), index=False)
        log(f"prep {split}: {len(df):,} rows")
        if split == "train":
            gt = read_tsv(f"{a.data}/train/train_ground_truth.tsv")
            gt["m"] = gt.matched_entity_ids.str.split(",")
            ex = gt.explode("m")
            ex["m"] = ex.m.str.strip()
            ex = ex[ex.m.notna() & (ex.m != "")]
            idx = pd.Index(df.entity_id)
            s1r, rr = idx.get_indexer(ex.source1_entity_id), idx.get_indexer(ex.m)
            ok = (s1r >= 0) & (rr >= 0)
            n = len(df)
            owner = np.full(n, -1, np.int64)
            owner[rr[ok]] = s1r[ok]
            src = df.src.values
            rng = np.random.RandomState(0)
            half = (rng.rand(n) < 0.5).astype(np.int8)              # S1 half; unmatched records random
            rec = src != 1
            has = rec & (owner >= 0)
            half[has] = half[owner[has]]                            # a copy follows its business
            np.save(wp(a, "train_owner.npy"), owner)
            np.save(wp(a, "train_half.npy"), half)
            log(f"GT: {int(ok.sum()):,} pairs; S1 in half A (encoder): {int(((src == 1) & (half == 0)).sum()):,}, "
                f"half B (decision): {int(((src == 1) & (half == 1)).sum()):,}")


# =====================================================================================================================
# train (DDP, one process per GPU)
# =====================================================================================================================
def build_train_pairs(a):
    df = pd.read_parquet(wp(a, "train_recs.parquet"), columns=["name", "text", "country", "src"])
    owner, half = np.load(wp(a, "train_owner.npy")), np.load(wp(a, "train_half.npy"))
    r = np.flatnonzero((df.src.values != 1) & (owner >= 0) & (half == 0))
    rng = np.random.RandomState(1)
    if len(r) > a.max_pairs:
        r = rng.choice(r, a.max_pairs, replace=False)
    s = owner[r]
    P = pd.DataFrame({"s1_row": s, "s1_text": df.text.values[s], "rec_text": df.text.values[r],
                      "country": df.country.values[s],
                      "key": pd.Series(df.name.values[s]).str.lower().str.replace(r"[^0-9a-z]", "", regex=True).str[:3].values})
    # batches of similar S1 names within a country -> in-batch negatives are look-alikes (chains): hard negatives
    P = P.sort_values(["country", "key"], kind="stable").reset_index(drop=True)
    B = a.batch
    nb = len(P) // B
    order = rng.permutation(nb)
    rows = (order[:, None] * B + np.arange(B)[None, :]).ravel()
    P = P.iloc[rows].reset_index(drop=True)
    P.drop(columns=["key"]).to_parquet(wp(a, "train_pairs.parquet"), index=False)
    log(f"training pairs: {len(P):,} in {nb:,} batches of {B}")


def _train_worker(rank, world, a, fp16):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.nn.functional import all_gather as diff_all_gather
    from torch.nn.parallel import DistributedDataParallel as DDP
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    torch.manual_seed(1234 + rank)
    P = pd.read_parquet(wp(a, "train_pairs.parquet"))
    B = a.batch
    nb = len(P) // B
    nb -= nb % world                                              # same number of steps on every GPU
    epoch_batches = np.arange(rank, nb, world)
    epoch_rng = np.random.RandomState(1000 + rank)
    mine = np.concatenate([epoch_rng.permutation(epoch_batches) for _ in range(a.epochs)]).tolist()
    steps_per_epoch = len(epoch_batches)
    dec = is_decoder(a.base_model)
    tok = load_tok(a.base_model, dec)
    model = load_backbone(a.base_model, dec, torch.float16 if (fp16 and dec) else torch.float32).to(dev)
    if dec:  # 0.6B does not fit full fine-tuning on 11 GB: LoRA adapters (fp32) on a frozen base
        from peft import LoraConfig, get_peft_model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                 "gate_proj", "up_proj", "down_proj"]))
        if rank == 0:
            model.print_trainable_parameters()
    model.train()
    ddp = DDP(model, device_ids=[rank])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = max(1, len(mine))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda")
    s1t, rt, sid = P.s1_text.values, P.rec_text.values, P.s1_row.values
    lab = rank * B + torch.arange(B, device=dev)          # positives' positions in the gathered batch
    rows = torch.arange(B, device=dev)
    t0 = time.time()
    for step, j in enumerate(mine):
        if rank == 0 and step % steps_per_epoch == 0:
            log(f"epoch {step // steps_per_epoch + 1}/{a.epochs} starting")
        sl = slice(j * B, (j + 1) * B)
        enc = tok(fmt(list(s1t[sl]) + list(rt[sl]), dec), padding=True, truncation=True, max_length=a.max_len,
                  return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            out = ddp(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        emb = F.normalize(pool(out.float(), enc["attention_mask"], dec), dim=-1)
        ea, eb = emb[:B], emb[B:]
        # negatives from all GPUs: each S1 is compared with world*B copies and vice versa
        ea_all = torch.cat(diff_all_gather(ea), 0)
        eb_all = torch.cat(diff_all_gather(eb), 0)
        ids = torch.as_tensor(sid[sl], device=dev)
        ids_l = [torch.zeros_like(ids) for _ in range(world)]
        dist.all_gather(ids_l, ids)
        ids_all = torch.cat(ids_l, 0)
        same = ids[:, None] == ids_all[None, :]                   # other copies of the same S1: not negatives
        same[rows, lab] = False
        l_r = ((eb @ ea_all.T) / a.tau).masked_fill(same, -1e4)
        l_s = ((ea @ eb_all.T) / a.tau).masked_fill(same, -1e4)
        loss = 0.5 * (F.cross_entropy(l_r, lab) + F.cross_entropy(l_s, lab))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if rank == 0 and (step % 50 == 0 or step == steps - 1):
            el = time.time() - t0
            log(f"train step {step + 1}/{steps} loss {loss.item():.4f} ({(step + 1) / el:.2f} it/s, "
                f"eta {el / (step + 1) * (steps - step - 1) / 60:.1f} min)")
    if rank == 0:
        out_model = model.merge_and_unload() if dec else model   # fold LoRA into the weights for fast encoding
        out_model.save_pretrained(wp(a, "model"))
        tok.save_pretrained(wp(a, "model"))
        with open(wp(a, os.path.join("model", "er_meta.json")), "w") as f:
            json.dump({"decoder": dec, "base": a.base_model}, f)
        log(f"saved fine-tuned encoder to {wp(a, 'model')}")
    dist.barrier()
    dist.destroy_process_group()


def cmd_train(a):
    import torch
    import torch.multiprocessing as tmp
    build_train_pairs(a)
    fp16 = check_fp16(a, a.base_model, is_decoder(a.base_model))
    world = min(a.gpus, torch.cuda.device_count())
    tmp.spawn(_train_worker, args=(world, a, fp16), nprocs=world, join=True)


# =====================================================================================================================
# embed (one process per GPU, disjoint rows of one shared memmap)
# =====================================================================================================================
def _embed_worker(rank, world, a, split, path, n, fp16):
    import torch
    import torch.nn.functional as F
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    md, dec = model_info(a)
    tok = load_tok(md, dec)
    model = load_backbone(md, dec, torch.float16 if fp16 else torch.float32).to(dev).eval()
    texts = pd.read_parquet(wp(a, f"{split}_recs.parquet"), columns=["text"]).text.values
    idx = np.array_split(np.arange(n), world)[rank]
    lens = np.fromiter((len(t) for t in texts[idx]), dtype=np.int64, count=len(idx))
    order = idx[np.argsort(lens, kind="stable")]                    # similar lengths together: little padding
    E = np.load(path, mmap_mode="r+")
    t0 = time.time()
    bs = a.embed_bs
    oom_retries = 0
    with torch.inference_mode():
        k, st = 0, 0
        while st < len(order):
            ii = order[st:st + bs]
            try:
                enc = tok(fmt(list(texts[ii]), dec), padding=True, truncation=True, max_length=a.max_len,
                          return_tensors="pt").to(dev)
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
                e = F.normalize(pool(out, enc["attention_mask"], dec).float(), dim=-1)  # normalize in fp32, not the whole tensor
                E[ii] = e.half().cpu().numpy()
                del enc, out, e
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_retries += 1
                if bs <= 8:
                    raise
                bs = max(8, bs // 2)
                if rank == 0:
                    log(f"embed {split}: OOM at batch {bs * 2} -> halving to {bs} and retrying "
                        f"(retry #{oom_retries}; consider --embed-bs {bs} next time)")
                continue                                            # retry this same slice at the smaller size
            if rank == 0 and k % 200 == 0:
                done = st + len(ii)
                log(f"embed {split}: {done:,}/{len(order):,} on gpu0 "
                    f"(eta {(time.time() - t0) / max(1, done) * (len(order) - done) / 60:.1f} min)")
            if k % 500 == 0:
                torch.cuda.empty_cache()                             # periodic defrag: cheap, avoids slow fragmentation
            st += len(ii)
            k += 1
    E.flush()


def cmd_embed(a):
    import torch
    import torch.multiprocessing as tmp
    from transformers import AutoConfig
    world = min(a.gpus, torch.cuda.device_count())
    md, dec = model_info(a)
    dim = AutoConfig.from_pretrained(md).hidden_size
    fp16 = check_fp16(a, md, dec)
    for split in ("train", "test"):
        n = len(pd.read_parquet(wp(a, f"{split}_recs.parquet"), columns=["src"]))
        path = wp(a, f"{split}_emb_{a.model}.npy")
        E = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, dim))
        del E
        tmp.spawn(_embed_worker, args=(world, a, split, path, n, fp16), nprocs=world, join=True)
        log(f"embedded {split}: {n:,} x {dim}")


# =====================================================================================================================
# search (exact top-k per record among S1 of the same country; records sharded over GPUs)
# =====================================================================================================================
def _search_worker(rank, world, a, split):
    import torch
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    meta = pd.read_parquet(wp(a, f"{split}_recs.parquet"), columns=["country", "src"])
    cty, src = meta.country.values, meta.src.values
    E = np.load(wp(a, f"{split}_emb_{a.model}.npy"), mmap_mode="r")
    R, S, K, V = [], [], [], []
    for c in sorted(set(cty)):
        s1 = np.flatnonzero((cty == c) & (src == 1))
        rr = np.flatnonzero((cty == c) & (src != 1))
        if not len(s1) or not len(rr):
            continue
        mine = np.array_split(rr, world)[rank]
        Sm = torch.from_numpy(np.ascontiguousarray(E[s1])).to(dev)
        k = min(a.k, len(s1))
        for st in range(0, len(mine), a.search_chunk):
            q = mine[st:st + a.search_chunk]
            sc = torch.from_numpy(np.ascontiguousarray(E[q])).to(dev) @ Sm.T
            v, ix = torch.topk(sc, k, dim=1)
            R.append(np.repeat(q, k))
            S.append(s1[ix.cpu().numpy().ravel()])
            K.append(np.tile(np.arange(1, k + 1, dtype=np.int16), len(q)))
            V.append(v.float().cpu().numpy().ravel())
        if rank == 0:
            log(f"search {split} [{c}]: {len(s1):,} S1, {len(rr):,} records")
        del Sm
    cat = (lambda L, dt: np.concatenate(L) if L else np.zeros(0, dt))  # noqa: E731
    np.savez(wp(a, f"{split}_search_{a.model}_r{rank}.npz"), rec=cat(R, np.int64), s1=cat(S, np.int64),
             rank=cat(K, np.int16), score=cat(V, np.float32))


def cmd_search(a):
    import torch
    import torch.multiprocessing as tmp
    world = min(a.gpus, torch.cuda.device_count())
    for split in ("train", "test"):
        tmp.spawn(_search_worker, args=(world, a, split), nprocs=world, join=True)
        parts = [np.load(wp(a, f"{split}_search_{a.model}_r{r}.npz")) for r in range(world)]
        cand = pd.DataFrame({k: np.concatenate([p[k] for p in parts]) for k in ("rec", "s1", "rank", "score")})
        cand.to_parquet(wp(a, f"{split}_cands_{a.model}.parquet"), index=False)
        log(f"search {split}: {len(cand):,} candidates")


# =====================================================================================================================
# features + decision model
# =====================================================================================================================
try:
    from anyascii import anyascii as _anyascii
except ImportError:
    _anyascii = None
_DIG = re.compile(r"\d+")


def _lo(s):
    return (s if (s.isascii() or _anyascii is None) else _anyascii(s)).lower()


def _house(s):
    m = _DIG.search(s)
    return (m.group(0).lstrip("0") or "0") if m else ""


def _str_chunk(task):
    from rapidfuzz import fuzz
    n1, n2, a1, a2 = task
    out = np.zeros((len(n1), 6), np.float32)
    for k in range(len(n1)):
        x1, x2, y1, y2 = _lo(n1[k]), _lo(n2[k]), _lo(a1[k]), _lo(a2[k])
        out[k, 0] = fuzz.token_set_ratio(x1, x2)
        out[k, 1] = fuzz.ratio(x1, x2)
        if y1 and y2:
            out[k, 2] = fuzz.token_set_ratio(y1, y2)
            out[k, 3] = fuzz.partial_ratio(y1, y2)
            h1, h2 = _house(y1), _house(y2)
            out[k, 4] = float(h1 == h2) if (h1 and h2) else -1.0
        else:
            out[k, 2] = out[k, 3] = out[k, 4] = -1.0
        out[k, 5] = float(not y2)
    return out


STR_COLS = ["name_tset", "name_ratio", "addr_tset", "addr_partial", "house_eq", "addr_empty"]
FEATS = ["score", "rank", "gap_top1", "margin12", "s1_max", "gap_s1", "s1_rank", "s1_n", "s1_n_top1", "src"] + STR_COLS


def build_features(a, split, cand, recs, rows_mask=None):
    """cand: rec, s1, rank, score (all records, rank<=K). Competition features use every candidate;
    string features only for rows in rows_mask (to save time)."""
    import multiprocessing as mp
    c = cand
    top1 = c[c["rank"] == 1].set_index("rec").score
    top2 = c[c["rank"] == 2].set_index("rec").score
    c["gap_top1"] = c.rec.map(top1).values - c.score.values
    c["margin12"] = (c.rec.map(top1) - c.rec.map(top2).fillna(0)).values
    g = c.groupby("s1").score
    c["s1_max"] = g.transform("max").values
    c["gap_s1"] = c.s1_max - c.score
    c["s1_rank"] = g.rank(ascending=False, method="first").values
    c["s1_n"] = g.transform("size").values
    c["s1_n_top1"] = c.s1.map(c[c["rank"] == 1].groupby("s1").size()).fillna(0).values
    if rows_mask is not None:
        c = c[rows_mask].reset_index(drop=True)
    c["src"] = recs.src.values[c.rec.values]
    names, addrs = recs.name.values, recs.addr.values
    ch = 200000
    tasks = ((names[c.s1.values[i:i + ch]].tolist(), names[c.rec.values[i:i + ch]].tolist(),
              addrs[c.s1.values[i:i + ch]].tolist(), addrs[c.rec.values[i:i + ch]].tolist())
             for i in range(0, len(c), ch))
    n_tasks = -(-len(c) // 200000)
    with mp.get_context("fork").Pool(a.jobs) as pool:
        S = np.concatenate(list(tqdm(pool.imap(_str_chunk, tasks), total=n_tasks,
                                     desc=f"features {split}: string similarity")))
    for k, col in enumerate(STR_COLS):
        c[col] = S[:, k]
    log(f"features {split}: {len(c):,} pairs")
    return c


def assign(c, p, t):
    """one S1 per record: its best candidate if p >= t."""
    o = np.lexsort((-p, c.rec.values))
    first = np.r_[True, c.rec.values[o][1:] != c.rec.values[o][:-1]]
    best = o[first]
    return best[p[best] >= t]


def macro_f05(s1_rows, n_true, pred_s1, pred_ok, N):
    pred = np.bincount(pred_s1, minlength=N)[s1_rows]
    tp = np.bincount(pred_s1[pred_ok], minlength=N)[s1_rows]
    nt = n_true[s1_rows]
    f = np.where(nt == 0, (pred == 0).astype(float), np.where(tp > 0, 1.25 * tp / (0.25 * nt + pred), 0.0))
    return float(f.mean())


def cmd_fit(a):
    import lightgbm as lgb
    recs = pd.read_parquet(wp(a, "train_recs.parquet"), columns=["name", "addr", "country", "src"])
    owner, half = np.load(wp(a, "train_owner.npy")), np.load(wp(a, "train_half.npy"))
    N = len(recs)
    cand = pd.read_parquet(wp(a, f"train_cands_{a.model}.parquet"))
    src, cty = recs.src.values, recs.country.values
    # recall of the retrieval on half B (the encoder never trained on these businesses)
    tr = np.full(N, 10 ** 6, np.int64)
    hit = owner[cand.rec.values] == cand.s1.values
    np.minimum.at(tr, cand.rec.values[hit], cand["rank"].values[hit])
    print("\n=== retrieval recall on held-out half B (is the true S1 in the record's top-k?) ===")
    for cc in sorted(set(cty)):
        m = (cty == cc) & (src != 1) & (owner >= 0) & (half == 1)
        if m.any():
            print(f"{cc:8s} " + "  ".join(f"@{k}={(tr[m] <= k).mean():.4f}" for k in (1, 2, 3, 5, 10, 20) if k <= a.k))
    cand = cand[cand["rank"] <= a.kcls].reset_index(drop=True)
    inB = half[cand.rec.values] == 1
    F_ = build_features(a, "train", cand, recs, rows_mask=inB)
    y = (owner[F_.rec.values] == F_.s1.values).astype(np.int8)
    # holdout inside B: 10% of B's businesses (+ 10% of B's unmatched records) for early stopping + threshold
    hb = (np.arange(N) * 2654435761 % 1000) < 100
    key = np.where(owner >= 0, owner, np.arange(N))
    ho = hb[key[F_.rec.values]]
    fit_rows = np.flatnonzero(~ho)
    if len(fit_rows) > a.max_train_rows:
        fit_rows = np.random.RandomState(2).choice(fit_rows, a.max_train_rows, replace=False)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=a.jobs, seed=0)
    dtr = lgb.Dataset(F_[FEATS].values[fit_rows], y[fit_rows], feature_name=FEATS)
    dva = lgb.Dataset(F_[FEATS].values[ho], y[ho], reference=dtr)
    bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    Fh = F_[ho].reset_index(drop=True)
    p = bst.predict(Fh[FEATS].values, num_iteration=bst.best_iteration)
    n_true = np.bincount(owner[owner >= 0], minlength=N)
    s1_ho = np.flatnonzero((src == 1) & (half == 1) & hb)
    best = (-1.0, 0.5)
    for t in tqdm(np.arange(0.05, 0.96, 0.025), desc="fit: sweeping decision threshold"):
        sel = assign(Fh, p, t)
        ps1 = Fh.s1.values[sel]
        f = macro_f05(s1_ho, n_true, ps1, owner[Fh.rec.values[sel]] == ps1, N)
        if f > best[0]:
            best = (f, float(t))
    print(f"\n=== decision: holdout macro F0.5 = {best[0]:.4f} at threshold {best[1]:.3f} "
          f"({len(s1_ho):,} held-out S1; best iteration {bst.best_iteration}) ===")
    imp = sorted(zip(FEATS, bst.feature_importance("gain")), key=lambda x: -x[1])
    print("feature importance (gain): " + ", ".join(f"{k}={v:.0f}" for k, v in imp))
    bst.save_model(wp(a, "lgb.txt"), num_iteration=bst.best_iteration)
    with open(wp(a, "threshold.txt"), "w") as f:
        f.write(str(best[1]))


def cmd_predict(a):
    import lightgbm as lgb
    recs = pd.read_parquet(wp(a, "test_recs.parquet"), columns=["entity_id", "name", "addr", "country", "src"])
    cand = pd.read_parquet(wp(a, f"test_cands_{a.model}.parquet"))
    cand = cand[cand["rank"] <= a.kcls].reset_index(drop=True)
    F_ = build_features(a, "test", cand, recs)
    bst = lgb.Booster(model_file=wp(a, "lgb.txt"))
    t = float(open(wp(a, "threshold.txt")).read())
    p = bst.predict(F_[FEATS].values)
    sel = assign(F_, p, t)
    ids = recs.entity_id.values
    s1_rows = np.flatnonzero(recs.src.values == 1)
    match = pd.Series(ids[F_.rec.values[sel]]).groupby(F_.s1.values[sel]).agg(",".join)
    cands = pd.Series(ids[F_.rec.values]).groupby(F_.s1.values).agg(",".join)
    os.makedirs(a.out, exist_ok=True)
    for fn, col, ser in (("matching_results.tsv", "matched_entity_ids", match),
                         ("candidate_pairs.tsv", "candidate_entity_ids", cands)):
        out = pd.DataFrame({"source1_entity_id": ids[s1_rows], col: ser.reindex(s1_rows).fillna("").values})
        out.to_csv(os.path.join(a.out, fn), sep="\t", index=False)
    log(f"wrote {a.out}/matching_results.tsv: {len(s1_rows):,} S1, {len(sel):,} matched records "
        f"({(match.reindex(s1_rows).fillna('') != '').mean():.1%} of S1 non-empty), threshold {t:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "train", "embed", "search", "fit", "predict", "all"])
    ap.add_argument("--data", default="dataset", help="folder containing train/ and test/")
    ap.add_argument("--work", default="/scratch/ishaan.karan/er_work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="ft", choices=["ft", "base"], help="ft = fine-tuned, base = pretrained only")
    ap.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B",
                    help="e.g. Qwen/Qwen3-Embedding-0.6B (best that fits) or intfloat/multilingual-e5-small (fastest)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-pairs", type=int, default=None, help="training pairs (default 1.2M Qwen, 3M e5)")
    ap.add_argument("--batch", type=int, default=None, help="pairs per GPU per step (default 64 Qwen, 256 e5)")
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 (LoRA) / 3e-5 (full fine-tune)")
    ap.add_argument("--epochs", type=int, default=3, help="passes over the sampled training pairs")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--embed-bs", type=int, default=None, help="default 512 Qwen, 1024 e5")
    ap.add_argument("--k", type=int, default=20, help="candidates retrieved per record")
    ap.add_argument("--kcls", type=int, default=5, help="candidates per record given to the decision model")
    ap.add_argument("--search-chunk", type=int, default=1024)
    ap.add_argument("--max-train-rows", type=int, default=8000000)
    ap.add_argument("--port", type=int, default=29533)
    a = ap.parse_args()
    dec = is_decoder(a.base_model)
    a.lr = a.lr if a.lr is not None else (1e-4 if dec else 3e-5)
    a.batch = a.batch if a.batch is not None else (64 if dec else 256)
    a.max_pairs = a.max_pairs if a.max_pairs is not None else (1200000 if dec else 3000000)
    a.embed_bs = a.embed_bs if a.embed_bs is not None else (512 if dec else 1024)
    steps = ["prep", "train", "embed", "search", "fit", "predict"] if a.cmd == "all" else [a.cmd]
    for s in steps:
        log(f"===== {s} =====")
        globals()[f"cmd_{s}"](a)
    log("done")


if __name__ == "__main__":
    main()
