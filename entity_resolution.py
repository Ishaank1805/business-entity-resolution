#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Business Entity Resolution -- Amazon ML Challenge 2026 -- single-file solution
Sized for the real data (~2M S1 + ~10M S2/S3 records per split) on one machine with 4 x 12 GB GPUs.

RUN EVERYTHING FROM THE FOLDER THAT CONTAINS dataset/ (the student_resource root):
  python entity_resolution.py download        # Hugging Face dataset Ishaank18/student-resource -> ./dataset ./utils
  python entity_resolution.py cpu             # prep block feat l1 prune l2 decide  -> output/*.tsv
  python entity_resolution.py ce --name ce06  # [optional, GPU] cross-encoder, one CV fold per GPU
  python entity_resolution.py l2 && python entity_resolution.py decide
  python entity_resolution.py package --team YourTeam

STAGES (each caches to WORK_DIR and can be re-run alone)
  download  fetch dataset/, utils/, Documentation_template.md from the HF dataset repo
  eda       data audit (exclusivity, singletons, countries, leakage probes)
  prep      TSV -> normalized columnar store (parallel, chunked); ground truth as arrays; CV folds
  block     token-IDF blocking: name / address / mixed token views (hashed, df-capped, per country),
            sparse top-k in both directions (S1 -> records and records -> S1)
  feat      ~90 pair features computed in parallel over S1 chunks -> feature parts on disk
  l1        LightGBM trained on a SAMPLE of S1 entities (4-fold OOF) + a full-sample model that scores every pair
  prune     top candidates per S1 / per record  -> exactly the set written to candidate_pairs.tsv
  ce        [GPU] Qwen3-Reranker-0.6B + LoRA trained on the sample; OOF logits + sharded test scoring
  l2        LightGBM stacker: L1 features + p1 + CE logits + competition / neighbour context
  decide    exclusivity mode x policy (threshold | two-threshold | exact expected-F0.5), cross-fitted
  loco      leave-one-country-out report on the sample (proxy for unseen France)
  smoke     synthetic end-to-end test;  package: build <team>_submission.zip

RULES: no external data / lookups / geocoding. Models: Apache-2.0 / MIT, <= 8B params.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

try:
    import yaml
except ImportError:  # optional
    yaml = None

try:
    from tqdm import tqdm
except ImportError:  # optional; progress bars degrade to no-ops
    def tqdm(it=None, **kw):
        return it if it is not None else iter(())

__version__ = "2.0.0"

# =============================================================================================================
# CONFIG
# =============================================================================================================
DEFAULTS = {
    "data_dir": "dataset",
    "work_dir": os.environ.get("BER_WORK_DIR", "work"),      # relative: works on SageMaker / any box
    "output_dir": os.environ.get("BER_OUTPUT_DIR", "output"),
    "seed": 42,
    "n_jobs": -1,                      # CPU workers (-1 = all cores but one)
    "run": "main",                     # 'main' | 'loco' (set by the loco command)
    "hf_repo": "Ishaank18/student-resource",
    "cv": {"n_folds": 4, "scheme": "group"},
    "sample_s1": 150000,               # S1 entities used to TRAIN the models (every pair is still scored)
    "prep": {"chunk": 200000, "translit": True},   # Indic -> Latin romanization in basic()
    "blocking": {
        # Measured on the real data at the original settings: recall=0.869, i.e. 13% of true matches never
        # reached any model. The values below are the response; see `diagnose` for how to verify them.
        "within_country": True,        # False if eda shows many cross-country ground-truth pairs
        "hash_bits": 24,
        "max_df": 20000,               # tokens more frequent than this (within a country) are not used to
                                       # retrieve. At 5000 this dropped every token in >0.1% of a 5M-row
                                       # country group, which can zero out a row entirely -- an all-zero row
                                       # is unreachable in that view at ANY k.
        "max_products": 1.2e10,        # work budget per view; the most frequent tokens are dropped to meet it.
                                       # Blocking time scales roughly linearly with this.
        "k_s1": {"full": 15, "name": 6, "addr": 6},   # top-k records per S1, per view
        # k_rec is the efficient direction and was badly starved: recall_full_rec=0.839 at k_rec=3 beat
        # recall_full_s1=0.704 at k_s1=15, because a record has ~one correct S1 entity while an S1 entity has
        # ~4.7 correct records competing for its k slots. Both directions reuse the same sparse matmul -- k
        # only decides how many results per row are kept -- so raising k_rec costs almost no blocking time,
        # only more candidate pairs for feat/CE to chew through.
        "k_rec": {"full": 8, "name": 3, "addr": 3},   # top-k S1 per record, per view
        "diagnose": True,              # on train, split every missed true pair into 'unreachable' (no shared
                                       # surviving token in any view -- raising k cannot help, loosen max_df /
                                       # max_products) vs 'ranked_out' (reachable but outside top-k -- raise k)
    },
    "features": {"drop": [], "chunk_pairs": 1000000},
    "l1": {"prune_min_p": 0.002, "prune_rank_s1": 12, "prune_rank_rec": 6},
    "lgb": {"learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.8,
            "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0, "num_boost_round": 1200,
            "early_stopping": 60, "es_frac": 0.1, "max_bin": 127, "seeds": [42], "device": "cpu"},
    "l2": {"seeds": [42], "ce_names": "auto"},
    "decision": {"max_cands": 12, "policies": ["threshold", "two_threshold", "expected_f"],
                 "excl_modes": ["none", "softmax", "hard", "softmax_hard"], "max_eval_s1": 400000},
    # ce.gate: only pairs with lo < p1 < hi are scored by the cross-encoder (L2 fills the rest with p1).
    # The L1 model is already near-certain outside that band, so this is the single largest inference saving.
    # ce.n_models: how many CE models to train. 2 gives full OOF + test coverage at 1/3 the cost of 4.
    "ce": {"gpus": "0", "name": "ce-xlmr", "gate": [0.02, 0.98], "train_rank_s1": 10, "n_models": 2,
           "tok_parallel": True, "neg_per_rec": 1},
}

# One-shot speed profile (`--fast`): merged over the config before any stage runs.
FAST_PROFILE = {
    "sample_s1": 100000,
    "lgb": {"num_boost_round": 800, "learning_rate": 0.12, "num_leaves": 95, "early_stopping": 50},
    "ce": {"name": "ce-xlmr-fast", "gate": [0.03, 0.97], "n_models": 1},
}

# Cross-encoder presets.
#
#   arch="encoder" (RECOMMENDED): a real cross-encoder -- AutoModelForSequenceClassification, num_labels=1,
#     text-pair input, FULLY fine-tuned. ~12 layers x 768 hidden vs Qwen3-0.6B's 28 x 1024, and no ~100-token
#     instruction/chat boilerplate on every pair, so roughly an order of magnitude less compute per pair.
#     Full fine-tuning of 86M transformer params also adapts far more to this task than LoRA r=32 on a frozen
#     0.6B, so accuracy typically goes UP, not down.
#     xlm-roberta-base (MIT) is multilingual, which matters because France is unseen in train.
#
#   arch="causal" : the original Qwen3-Reranker yes/no head + LoRA. Kept so results stay reproducible.
CE_PRESETS = {
    "ce-xlmr": dict(arch="encoder", model="xlm-roberta-base", lr=2e-5, head_lr=1e-3, micro=64, accum=2,
                    epochs=2, neg_per_s1=6, max_len=128, quant=None, infer_bs=384, warmup=0.06, wd=0.01,
                    swap_p=0.5, seed=42, max_train_pairs=600000, grad_ckpt=False, compile=False,
                    pad_multiple=16, listwise_w=0.5),
    # Same model, one epoch and a tighter negative pool: for when the deadline is closer than the leaderboard.
    "ce-xlmr-fast": dict(arch="encoder", model="xlm-roberta-base", lr=3e-5, head_lr=1e-3, micro=64, accum=2,
                         epochs=1, neg_per_s1=4, max_len=112, quant=None, infer_bs=512, warmup=0.06, wd=0.01,
                         swap_p=0.5, seed=42, max_train_pairs=400000, grad_ckpt=False, compile=False,
                         pad_multiple=16, listwise_w=0.5),
    # Stronger multilingual backbone (Apache-2.0, 278M). ~2x slower than xlm-roberta-base; use if time allows.
    "ce-mdeberta": dict(arch="encoder", model="microsoft/mdeberta-v3-base", lr=1.5e-5, head_lr=1e-3, micro=32,
                        accum=4, epochs=2, neg_per_s1=6, max_len=128, quant=None, infer_bs=256, warmup=0.06,
                        wd=0.01, swap_p=0.5, seed=42, max_train_pairs=600000, grad_ckpt=False, compile=False,
                        pad_multiple=16, listwise_w=0.5),
    # RELEVANCE-PRETRAINED backbones. These already have a trained cross-encoder scoring head, so fine-tuning
    # starts from "knows how to compare a query to a document" instead of from masked-language-modelling.
    # Usually the single biggest accuracy jump available for the same compute budget.
    # gte-multilingual-reranker-base: Apache-2.0, 306M, 70+ languages. Needs trust_remote_code=True.
    "ce-gte": dict(arch="encoder", model="Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True,
                   lr=1e-5, head_lr=5e-4, micro=64, accum=2, epochs=2, neg_per_s1=6, max_len=128, quant=None,
                   infer_bs=320, warmup=0.06, wd=0.01, swap_p=0.5, seed=42, max_train_pairs=600000,
                   grad_ckpt=False, compile=False, pad_multiple=16, listwise_w=0.5),
    # bge-reranker-v2-m3: Apache-2.0, 568M (XLM-R large). Strongest of these, ~2.5x slower than xlm-roberta-base.
    "ce-bge": dict(arch="encoder", model="BAAI/bge-reranker-v2-m3", lr=8e-6, head_lr=3e-4, micro=32, accum=4,
                   epochs=1, neg_per_s1=6, max_len=128, quant=None, infer_bs=192, warmup=0.06, wd=0.01,
                   swap_p=0.5, seed=42, max_train_pairs=500000, grad_ckpt=False, compile=False,
                   pad_multiple=16, listwise_w=0.5),
    "ce06": dict(arch="causal", model="Qwen/Qwen3-Reranker-0.6B", lr=2e-4, micro=16, accum=4, epochs=1,
                 neg_per_s1=6, max_len=256, lora_r=32, lora_alpha=64, lora_dropout=0.05, quant=None,
                 infer_bs=128, warmup=0.05, wd=0.01, swap_p=0.5, seed=42, max_train_pairs=400000,
                 grad_ckpt=True, compile=False, pad_multiple=0),
    "ce4b": dict(arch="causal", model="Qwen/Qwen3-Reranker-4B", lr=1e-4, micro=4, accum=16, epochs=1,
                 neg_per_s1=3, max_len=256, lora_r=16, lora_alpha=32, lora_dropout=0.05, quant="4bit",
                 infer_bs=32, warmup=0.05, wd=0.01, swap_p=0.5, seed=42, max_train_pairs=150000,
                 grad_ckpt=True, compile=False, pad_multiple=0),
}
CE_INSTRUCTION = ("Judge whether the Document is the same real-world business at the same location as the Query. "
                  "Allow abbreviations, typos, transliterations, legal suffixes, trade names and address formats.")


# YAML 1.1 only recognises a float exponent written as 4.0e+9, so yaml.safe_load("4e9") hands back the STRING
# "4e9". That silently turned `--set blocking.max_products=4e9` into a string and blew up deep inside blocking
# with a numpy dtype error, so coerce anything that is numeric by shape after parsing.
# NB: _NUM_RE is already taken further down (digit extraction in normalisation) -- a second definition here
# would be silently clobbered at import time, so this one is named distinctly.
_SCALAR_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_num(x):
    if isinstance(x, str) and _SCALAR_NUM_RE.match(x.strip()):
        t = x.strip()
        return int(t) if re.match(r"^[+-]?\d+$", t) else float(t)
    return x


def _parse_val(v: str):
    if yaml is not None:
        try:
            return _coerce_num(yaml.safe_load(v))
        except Exception:
            return _coerce_num(v)
    try:
        return json.loads(v)
    except Exception:
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            return [_parse_val(x.strip().strip("'\"")) for x in s[1:-1].split(",") if x.strip()]
        return {"true": True, "false": False, "null": None, "none": None}.get(s.lower(), _coerce_num(v))


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | None, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) if (yaml is not None and path.endswith((".yaml", ".yml"))) else json.load(f)
        cfg = _merge(cfg, user or {})
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_val(val)
    if cfg["n_jobs"] in (-1, None):
        # os.cpu_count() reports the whole machine. On a shared cluster node that is 10x the allocation, and
        # forking that many workers thrashes the box. sched_getaffinity respects cgroup pinning; SLURM's own
        # variable is the authority when present.
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 2
        slurm = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm and slurm.isdigit():
            n = min(n, int(slurm))
        cfg["n_jobs"] = max(1, n - 1)
    global TRANSLIT
    TRANSLIT = bool(cfg.get("prep", {}).get("translit", True))
    cfg["train_dir"] = os.path.join(cfg["data_dir"], "train")
    cfg["test_dir"] = os.path.join(cfg["data_dir"], "test")
    return cfg


def cfg_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def wpath(cfg: dict, split: str, name: str, tagged: bool = False, ext: str = "parquet") -> str:
    """work/<split>/<name>[_<run>].<ext>; fold-dependent artefacts are tagged with the run name."""
    d = os.path.join(cfg["work_dir"], split)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}{run_tag(cfg) if tagged else ''}.{ext}")


def run_tag(cfg: dict) -> str:
    return f"_{cfg['run']}" if cfg.get("run", "main") != "main" else ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_exp(cfg: dict, stage: str, metrics: dict, t0: float | None = None, note: str = "") -> None:
    """Experiment tracker: one JSON line per stage run (config hash, metrics, wall time) + config snapshots."""
    os.makedirs(os.path.join(cfg["work_dir"], "configs"), exist_ok=True)
    h = cfg_hash(cfg)
    snap = os.path.join(cfg["work_dir"], "configs", f"{h}.json")
    if not os.path.exists(snap):
        with open(snap, "w") as f:
            json.dump(cfg, f, indent=1, default=str)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "run": cfg.get("run"), "cfg": h,
           "elapsed_s": round(time.time() - t0, 1) if t0 else None, "metrics": metrics, "note": note}
    with open(os.path.join(cfg["work_dir"], "experiments.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _need(path: str, hint: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run `{hint}` first")
    return path


def _pool_ctx():
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def parallel_map(fn, tasks: list, n_jobs: int, globals_: dict | None = None):
    """Ordered map over tasks. Uses fork so that large read-only numpy globals are shared, not copied."""
    if globals_ is not None:
        _G.clear()
        _G.update(globals_)
    ctx = _pool_ctx()
    try:
        if n_jobs > 1 and len(tasks) > 1 and ctx is not None:
            with ctx.Pool(min(n_jobs, len(tasks))) as pool:
                return list(pool.imap(fn, tasks))
        return [fn(t) for t in tasks]
    finally:
        if globals_ is not None:
            _G.clear()


_G: dict = {}  # globals for forked workers


# =============================================================================================================
# COLUMNAR STRINGS: one contiguous UTF-8 buffer + offsets in numpy. Forked workers share it without copying
# (no per-string Python objects, so copy-on-write is never triggered by reference counting).
# =============================================================================================================
class StrCol:
    __slots__ = ("off", "buf")

    def __init__(self, off: np.ndarray, buf: np.ndarray):
        self.off, self.buf = off, buf

    @classmethod
    def from_arrow(cls, arr) -> "StrCol":
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks() if arr.num_chunks != 1 else arr.chunk(0)
        if pa.types.is_large_string(arr.type):
            odt = np.int64
        else:
            if not pa.types.is_string(arr.type):
                arr = arr.cast(pa.string())
            odt = np.int32
        arr = arr.fill_null("") if arr.null_count else arr
        bufs = arr.buffers()
        off = np.frombuffer(bufs[1], dtype=odt)[arr.offset: arr.offset + len(arr) + 1]
        base = int(off[0]) if len(off) else 0
        end = int(off[-1]) if len(off) else 0
        data = np.frombuffer(bufs[2], dtype=np.uint8)[base:end].copy() if bufs[2] is not None and end > base \
            else np.zeros(0, np.uint8)
        off = (off.astype(np.int64) - base)
        return cls(off.astype(np.int32) if end - base < 2 ** 31 - 1 else off, data)

    @classmethod
    def from_list(cls, strings: list[str]) -> "StrCol":
        return cls.from_arrow(pa.array(strings, type=pa.string()))

    def __len__(self) -> int:
        return len(self.off) - 1

    def take(self, idx) -> list[str]:
        idx = np.asarray(idx, dtype=np.int64)
        mv = memoryview(self.buf)
        a, b = self.off[idx].tolist(), self.off[idx + 1].tolist()
        return [str(mv[x:y], "utf-8") for x, y in zip(a, b)]

    def lengths(self, idx=None) -> np.ndarray:
        """Byte lengths (cheap, vectorized)."""
        o = self.off.astype(np.int64)
        return (o[1:] - o[:-1]) if idx is None else (o[np.asarray(idx) + 1] - o[np.asarray(idx)])

    def to_arrow(self) -> pa.Array:
        return pa.StringArray.from_buffers(len(self), pa.py_buffer(self.off.astype(np.int32).tobytes()),
                                           pa.py_buffer(self.buf.tobytes()))


# =============================================================================================================
# IO (pyarrow: fast, multi-threaded, low memory). Quoting disabled: a stray '"' must not swallow lines.
# =============================================================================================================
SRC_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def _header(path: str) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        return [c.strip() for c in f.readline().rstrip("\r\n").split("\t")]


def read_tsv_arrow(path: str, expect: list[str]) -> pa.Table:
    import pyarrow.csv as pacsv
    names = _header(path)
    missing = [c for c in expect if c not in names]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {names}")
    bad: list[str] = []

    def handler(row):
        bad.append(row.text)
        return "skip"

    t = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=names, skip_rows=1, block_size=64 << 20, encoding="utf8"),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, invalid_row_handler=handler),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in names},
                                             strings_can_be_null=False, quoted_strings_can_be_null=False))
    if bad:  # repair rows with a wrong number of tab-separated fields instead of dropping entities
        n = len(names)
        rows = []
        for text in bad:
            f = text.rstrip("\r\n").split("\t")
            if len(f) > n:
                f = [f[0], ",".join(f[1:])] if n == 2 else f[:n - 2] + [" ".join(f[n - 2:-1]), f[-1]]
            f = (f + [""] * n)[:n]
            rows.append(f)
        log(f"WARNING {path}: repaired {len(bad)} malformed rows (wrong number of tabs)")
        fix = pa.table({c: pa.array([r[k] for r in rows], pa.string()) for k, c in enumerate(names)})
        t = pa.concat_tables([t, fix])
    t = t.select(expect)
    return pa.table({c: pc.utf8_trim_whitespace(t[c]) for c in expect})


def write_id_lists(path: str, s1_ids: list[str], lists: list[str], col2: str) -> None:
    """lists[k] = comma-joined ids for s1_ids[k]."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col2}\n")
        f.write("".join(f"{a}\t{b}\n" for a, b in zip(s1_ids, lists)))


def check_submission(match_path: str, cand_path: str, s1_ids: pa.Array, rec_ids: pa.Array) -> list[str]:
    """Replicates the stated rules (the official utils/validate_submission.py remains authoritative)."""
    problems: list[str] = []
    sets = {}
    for path, col in ((match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")):
        if _header(path) != ["source1_entity_id", col]:
            problems.append(f"{path}: bad header {_header(path)}")
            continue
        t = read_tsv_arrow(path, ["source1_entity_id", col])
        a = t["source1_entity_id"].combine_chunks()
        if len(pc.unique(a)) != len(a):
            problems.append(f"{path}: duplicate source1_entity_id rows")
        if len(a) != len(s1_ids) or not pc.all(pc.is_in(s1_ids, value_set=a)).as_py():
            problems.append(f"{path}: rows do not cover exactly the {len(s1_ids):,} test S1 ids")
        lst = pc.split_pattern(t[col].combine_chunks(), ",")
        flat = pc.list_flatten(lst)
        parent = pc.list_parent_indices(lst).to_numpy()
        ok = pc.not_equal(flat, "")
        flat = pc.filter(flat, ok)
        parent = parent[ok.to_numpy(zero_copy_only=False)]
        if len(flat):
            if not pc.all(pc.is_in(flat, value_set=rec_ids)).as_py():
                problems.append(f"{path}: ids that are not test S2/S3 records")
            key = pd.DataFrame({"p": parent, "x": flat.to_numpy(zero_copy_only=False)})
            if key.duplicated().any():
                problems.append(f"{path}: duplicate ids inside a list")
            sets[col] = key
    if len(sets) == 2:
        m, c = sets["matched_entity_ids"], sets["candidate_entity_ids"]
        # rows were written in the same S1 order in both files, so parent indices align
        n_bad = len(m.merge(c, on=["p", "x"], how="left", indicator=True).query("_merge == 'left_only'"))
        if n_bad:
            problems.append(f"{n_bad} matched ids are not in candidate_pairs.tsv")
    return problems[:50]


# =============================================================================================================
# METRIC
# =============================================================================================================
def f_beta_entity(pred: set, true: set, beta: float = 0.5) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * len(true) + len(pred))


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    ids = list(s1_ids)
    return float(np.mean([f_beta_entity(set(pred.get(s, ())), set(truth.get(s, ()))) for s in ids])) if ids else float("nan")


def breakdown_arrays(per_entity: np.ndarray, n_true: np.ndarray, n_pred: np.ndarray) -> dict:
    single, multi = n_true == 0, n_true > 0
    mean = lambda m: round(float(per_entity[m].mean()), 6) if m.any() else None  # noqa: E731
    return {"all": mean(np.ones(len(per_entity), bool)), "n": int(len(per_entity)), "singletons": mean(single),
            "n_singletons": int(single.sum()), "with_matches": mean(multi),
            "singleton_false_merge_rate": round(float((n_pred[single] > 0).mean()), 6) if single.any() else None,
            "matched_but_empty_rate": round(float((n_pred[multi] == 0).mean()), 6) if multi.any() else None}


# =============================================================================================================
# NORMALIZATION (country-agnostic: France is unseen in train, so nothing branches on the country label)
# =============================================================================================================
LEGAL_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lc", "ltd", "limited",
    "llp", "lp", "plc", "pllc", "pc", "pa", "chtd", "lllp",
    "pvt", "private", "opc", "huf",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "scp", "sca", "selarl", "selas", "cie",
    "gmbh", "ag", "bv", "nv", "srl", "spa", "sl", "sprl", "oy", "ab", "as",
    "pty", "pte", "bhd", "sdn", "kk",
}
NAME_MAP = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd", "private": "pvt",
    "brothers": "bros", "brother": "bro", "international": "intl", "internationale": "intl",
    "manufacturing": "mfg", "services": "svcs", "service": "svc", "technologies": "tech",
    "technology": "tech", "associates": "assoc", "association": "assoc", "department": "dept",
    "enterprises": "ent", "enterprise": "ent", "industries": "ind", "industry": "ind",
    "management": "mgmt", "government": "govt", "hospital": "hosp", "restaurant": "rest",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "laboratories": "labs", "laboratory": "lab",
    "solutions": "soln", "systems": "sys", "center": "ctr", "centre": "ctr", "national": "natl",
    "saint": "st", "sainte": "ste", "mount": "mt", "and": "&", "et": "&", "und": "&",
    "compagnie": "cie", "societe": "ste", "etablissements": "ets",
    "shree": "sri", "shri": "sri", "shre": "sri", "sree": "sri",
    "lakshmi": "laxmi", "laksmi": "laxmi", "ganesha": "ganesh", "krsna": "krishna",
}
NAME_STOP = {"the", "&", "of", "de", "la", "le", "les", "du", "des", "d", "l", "a", "an"}
ADDR_MAP = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave", "aven": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd", "lane": "ln", "drive": "dr",
    "court": "ct", "place": "pl", "plaza": "plz", "square": "sq", "highway": "hwy", "parkway": "pkwy",
    "expressway": "expy", "freeway": "fwy", "terrace": "ter", "circle": "cir", "trail": "trl",
    "suite": "ste", "apartment": "apt", "building": "bldg", "floor": "fl", "flr": "fl", "room": "rm",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "mount": "mt", "fort": "ft", "saint": "st", "sainte": "ste",
    "number": "no", "num": "no",
    "nagar": "ngr", "nager": "ngr", "marg": "mg", "sector": "sec", "sect": "sec", "market": "mkt",
    "opposite": "opp", "near": "nr", "behind": "bhd", "colony": "col", "cross": "crs", "main": "mn",
    "phase": "ph", "extension": "extn", "ext": "extn", "industrial": "indl", "estate": "est",
    "junction": "jn", "jct": "jn", "chowk": "chk", "bazaar": "bazar", "bajar": "bazar", "galli": "gali",
    "mohalla": "mohala", "post": "po", "district": "dist", "distt": "dist", "tehsil": "teh",
    "village": "vill", "vpo": "vill", "taluka": "tal", "taluk": "tal", "layout": "lyt", "stage": "stg",
    "block": "blk", "plot": "plt", "complex": "cplx", "tower": "twr", "house": "hse", "bhawan": "bhavan",
    "rue": "rue", "r": "rue", "impasse": "imp", "chemin": "chem", "ch": "chem", "route": "rte",
    "allee": "all", "faubourg": "fbg", "quai": "qu", "cours": "crs", "passage": "pass",
    "residence": "res", "batiment": "bat", "cedex": "",
}
ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
                 "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "premier": "1", "premiere": "1"}
# Source 1 writes the US state as a 2-letter code 97.2% of the time; the S2/S3 records spell it out 39.6% of
# the time. Same story in India (maharashtra/mh, delhi/dl, uttar pradesh/up). Full name -> code only, never the
# reverse: rewriting a bare "in"/"or"/"as" would clobber ordinary English words.
STATE_MAP = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    "maharashtra": "mh", "mharastr": "mh", "karnataka": "ka", "telangana": "ts", "gujarat": "gj",
    "rajasthan": "rj", "kerala": "kl", "punjab": "pb", "haryana": "hr", "bihar": "br", "odisha": "od",
    "orissa": "od", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "chattisgarh": "cg",
    "uttarakhand": "uk", "uttaranchal": "uk", "goa": "ga", "sikkim": "sk", "tripura": "tr", "manipur": "mnp",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "delhi": "dl", "dilli": "dl", "puducherry": "py",
    "pondicherry": "py", "chandigarh": "chd",
}
STATE_PHRASES = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "south carolina": "sc", "south dakota": "sd", "rhode island": "ri",
    "west virginia": "wv", "district of columbia": "dc", "puerto rico": "pr",
    "uttar pradesh": "up", "madhya pradesh": "mp", "andhra pradesh": "ap", "arunachal pradesh": "ar",
    "himachal pradesh": "hp", "tamil nadu": "tn", "tamilnadu": "tn", "west bengal": "wb", "new delhi": "dl",
    "nai dilli": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk", "andaman and nicobar": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
}
_STATE_PHRASE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_PHRASES), key=len, reverse=True))
                              + r")\b")
COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us", "united states of america": "us",
    "america": "us", "in": "india", "ind": "india", "india": "india", "bharat": "india", "republic of india": "india",
    "fr": "france", "fra": "france", "france": "france", "republique francaise": "france",
    "french republic": "france",
}
LANDMARK_RE = re.compile(
    r"\b(near|nr|opp|opposite|behind|bhd|beside|besides|next to|adjacent to|adj to|in front of|close to|"
    r"pres de|pres du|face a|face au|en face de|a cote de|derriere|a proximite de)\b\.?\s*([^,;]+)")
DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|trading as|doing business as|aka|a\s*/\s*k\s*/\s*a|"
                    r"formerly|formerly known as|fka)\b\.?", re.I)
_PUNCT_RE = re.compile(r"[^\w\s&#]")
_WS_RE = re.compile(r"\s+")
_ORD_RE = re.compile(r"\b(\d+)(st|nd|rd|th|er|ere|eme|e|me)\b")
_NUM_RE = re.compile(r"\d+")
_POSTAL6_RE = re.compile(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)")
_POSTAL5_RE = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")


# =============================================================================================================
# INDIC ROMANIZATION
# Source 1 names are 100% ASCII, but ~28% of India's S2/S3 names are in a native script (Devanagari, Tamil,
# Telugu, Bengali, Kannada, Gujarati, Malayalam, Oriya, Gurmukhi), and 23% of India's true S2 pairs differ from
# their S1 by script alone. Without romanization those pairs share no token at all: blocking cannot retrieve
# them, `skeleton()` cannot key them, and every fuzzy feature scores them at zero.
#
# All nine scripts inherit the ISCII layout, so a letter sits at the same offset from its Unicode block base in
# every one of them -- one Devanagari table romanizes all of them. Vowel accuracy barely matters downstream
# because `skeleton()` strips vowels anyway; what matters is getting the consonant skeleton right.
# =============================================================================================================
_INDIC_BASES = {0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00}
_IND_VOWEL = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "ri",
              0x0D: "li", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_IND_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
             0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
             0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
             0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
             0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
             0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
             0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h"}
_IND_MATRA = {0x3E: "aa", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
              0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
_IND_SIGN = {0x01: "n", 0x02: "n", 0x03: "h"}          # candrabindu, anusvara, visarga
_IND_VIRAMA, _IND_NUKTA = 0x4D, 0x3C
_INDIC_RE = re.compile(r"[ऀ-ൿ]")
_DBL_VOWEL_RE = re.compile(r"([aeiou])\1+")


def _indic_base(cp: int) -> int:
    return 0x0900 + ((cp - 0x0900) // 0x80) * 0x80


TRANSLIT = True   # set from cfg["prep"]["translit"]; read by forked workers after load_config


def translit_indic(s: str) -> str:
    """Romanize Indic text; ASCII and non-Indic input is returned untouched (fast path)."""
    if not TRANSLIT or s.isascii() or not _INDIC_RE.search(s):
        return s
    out, i, n = [], 0, len(s)
    while i < n:
        cp = ord(s[i])
        if not (0x0900 <= cp <= 0x0D7F) or _indic_base(cp) not in _INDIC_BASES:
            out.append(s[i])
            i += 1
            continue
        base = _indic_base(cp)
        off = cp - base
        if off in _IND_CONS:
            out.append(_IND_CONS[off])
            j = i + 1
            if j < n and ord(s[j]) == base + _IND_NUKTA:
                j += 1
            nxt = ord(s[j]) - base if (j < n and _indic_base(ord(s[j])) == base
                                       and 0x0900 <= ord(s[j]) <= 0x0D7F) else -1
            if nxt in _IND_MATRA:                      # explicit vowel sign
                out.append(_IND_MATRA[nxt])
                i = j + 1
            elif nxt == _IND_VIRAMA:                   # virama kills the inherent vowel
                i = j + 1
            else:                                      # bare consonant carries an inherent 'a'
                out.append("a")
                i = j
        elif off in _IND_VOWEL:
            out.append(_IND_VOWEL[off])
            i += 1
        elif off in _IND_SIGN:
            out.append(_IND_SIGN[off])
            i += 1
        elif 0x66 <= off <= 0x6F:                      # Indic digits -> ASCII (house numbers, PIN codes)
            out.append(chr(ord("0") + off - 0x66))
            i += 1
        else:                                          # nukta, avagraha, stray marks
            i += 1
    # Long vowels romanize to doubled letters (aa, ii, uu); Latin business names never spell them that way,
    # so collapse them or 'mahaaraashtra' never reaches the 'maharashtra' entry in STATE_MAP.
    return _DBL_VOWEL_RE.sub(r"\1", "".join(out))


def fold_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def basic(s: str) -> str:
    """NFKC + lowercase + Indic romanization + accent folding + punctuation -> space; '&' kept as a token."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = translit_indic(s)
    s = fold_accents(s)
    s = re.sub(r"['\u2019`]s\b", "s", s)
    s = re.sub(r"['\u2019`]", " ", s)
    s = s.replace("&", " & ").replace("+", " & ")
    s = _PUNCT_RE.sub(" ", s).replace("_", " ")
    return _WS_RE.sub(" ", s).strip()


def country_key(c: str) -> str:
    b = basic(c)
    return COUNTRY_ALIASES.get(b, b) if b else "unk"


def split_dba(name: str) -> list[str]:
    parts = [p.strip(" -,;:()") for p in DBA_RE.split(name or "")]
    return [p for p in parts if p]


def merge_initials(toks: list[str]) -> list[str]:
    """['m','g','rd'] -> ['mg','rd']: runs of >= 2 single letters become one token."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        out.extend(["".join(run)] if len(run) >= 2 else run)
        run = []
        out.append(t)
    out.extend(["".join(run)] if len(run) >= 2 else run)
    return out


def name_tokens(s: str) -> list[str]:
    toks = merge_initials(basic(s).split())
    return [NAME_MAP.get(t, t) for t in toks if t not in {"s", "l", "d"}]


def strip_legal(tokens: list[str]) -> tuple[list[str], list[str]]:
    toks, legal = list(tokens), []
    while toks and (toks[-1] in LEGAL_TOKENS or (toks[-1] in {"&", "and"} and legal)):
        legal.append(toks.pop())
    while toks and toks[0] == "the":
        toks.pop(0)
    return toks, [t for t in legal if t not in {"&", "and"}]


def legal_signature(legal: list[str]) -> str:
    s = set(legal)
    if "pvt" in s and "ltd" in s:
        return "pvtltd"
    for key in ("llc", "llp", "inc", "corp", "ltd", "plc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "gmbh",
                "pvt", "co", "cie", "lp", "pllc", "pc", "opc"):
        if key in s:
            return key
    return next(iter(sorted(s)), "")


_SKEL_FOLD = (("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
              ("ch", "c"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "j"), ("y", "i"),
              ("c", "k"))   # bare c -> k: 'medical' and a romanized 'medikal' must land on the same key
# Dravidian scripts (Tamil, Malayalam, and to a lesser extent Telugu/Kannada) do not mark voicing, so the same
# letter romanizes as k or g, t or d, p or b depending on who typed it. Folding those pairs recovers the match
# but loses real distinctions, so this second key is used ONLY as an extra blocking route -- never as a feature.
_SKEL2_FOLD = (("g", "k"), ("d", "t"), ("b", "p"), ("j", "s"))


def _skel(tok: str, folds) -> str:
    t = tok
    for a, b in folds:
        t = t.replace(a, b)
    if not t:
        return t
    out = t[0] + re.sub(r"[aeiouh]", "", t[1:])
    return re.sub(r"(.)\1+", r"\1", out)


def skeleton(tok: str) -> str:
    """Phonetic / transliteration-robust key: digraph folding, drop vowels after the first char, collapse repeats."""
    return _skel(tok, _SKEL_FOLD)


def skeleton2(tok: str) -> str:
    """Voicing-insensitive skeleton, for blocking recall on Dravidian-script records."""
    return _skel(tok, _SKEL_FOLD + _SKEL2_FOLD)


def is_abbrev(a: str, b: str) -> bool:
    """a abbreviates b: prefix, or ordered subsequence sharing first and last character."""
    if len(a) >= len(b) or len(a) < 2 or a[0] != b[0] or a.isdigit() or b.isdigit():
        return False
    if b.startswith(a):
        return True
    if a[-1] != b[-1]:
        return False
    it = iter(b)
    return all(ch in it for ch in a)


def acronym(tokens: list[str]) -> str:
    return "".join(t[0] for t in tokens if t and not t.isdigit())


def addr_tokens(s: str) -> list[str]:
    b = basic(s)
    # Multi-word state names collapse first, before ADDR_MAP turns "north" into "n" and splits them apart.
    b = _STATE_PHRASE_RE.sub(lambda m: STATE_PHRASES[m.group(1)], b)
    b = _POSTAL6_RE.sub(r"\1\2", b)
    b = _ORD_RE.sub(r"\1", b)
    out = []
    for t in merge_initials(b.split()):
        if t in {"l", "d"}:
            continue
        t = ORDINAL_WORDS.get(t, t)
        t = STATE_MAP.get(t) or ADDR_MAP.get(t, t)
        if t:
            out.append(t)
    return out


def extract_postal(raw: str) -> str:
    # Indic digits must be romanized here too, or PIN codes in native script never match.
    s = fold_accents(translit_indic(unicodedata.normalize("NFKC", (raw or "")).lower()))
    m6 = [a + b for a, b in _POSTAL6_RE.findall(s)]
    if m6:
        return m6[-1]
    m5 = _POSTAL5_RE.findall(s)
    return m5[-1] if m5 else ""


def extract_landmark(raw: str) -> str:
    for comp in re.split(r"[,;]", raw or ""):
        m = LANDMARK_RE.search(basic(comp))
        if m:
            return m.group(2).strip()
    return ""


def normalize_record(name: str, address: str) -> dict:
    parts = split_dba(name) or [name or ""]
    ntok = name_tokens(" ".join(parts))
    core, legal, dba_cores = [], [], []
    for part in parts:
        c, lg = strip_legal(name_tokens(part))
        legal += lg
        c2 = [t for t in c if t not in NAME_STOP] or c
        core += c2
        if c2:
            dba_cores.append(" ".join(c2))
    atok = addr_tokens(address)
    postal = extract_postal(address)
    nums = [t for t in atok if t.isdigit()]
    nums_np = [t for t in nums if t != postal]
    comps = [c.strip() for c in re.split(r"[,;]", fold_accents((address or "").lower())) if c.strip()]
    return {
        "n_full": " ".join(ntok),
        "n_core": " ".join(core),
        "n_core_sorted": " ".join(sorted(core)),
        "n_legal": legal_signature(legal),
        "n_acr": acronym(core),
        "n_skel": " ".join(skeleton(t) for t in core),
        "n_skel2": " ".join(skeleton2(t) for t in core),
        "n_dba": "|".join(dba_cores) if len(dba_cores) > 1 else "",
        "n_digits": " ".join(_NUM_RE.findall(" ".join(core))),
        "a_norm": " ".join(atok),
        "a_words": " ".join(t for t in atok if not t.isdigit()),
        "a_nums": " ".join(nums_np),
        "a_postal": postal,
        "a_house": nums_np[0] if nums_np else "",
        "a_tail": " ".join(addr_tokens(" ".join(comps[-2:]))) if comps else "",
        "a_landmark": extract_landmark(address),
    }


def mine_abbreviations(pairs: list[tuple[str, str]], min_count: int = 5, max_pairs: int = 200000) -> dict[str, str]:
    """Token rewrite rules (short -> long) mined from matched TRAIN pairs (for error analysis / dictionary growth)."""
    cnt: Counter = Counter()
    for a, b in pairs[:max_pairs]:
        ta, tb = set(basic(a).split()), set(basic(b).split())
        for x in ta - tb:
            for y in tb - ta:
                if is_abbrev(x, y):
                    cnt[(x, y)] += 1
                elif is_abbrev(y, x):
                    cnt[(y, x)] += 1
    rules: dict[str, str] = {}
    for (short, long_), c in cnt.most_common():
        if c >= min_count and short not in rules:
            rules[short] = long_
    return rules

# =============================================================================================================
# UNIVERSE = one self-contained ER problem (train or test), stored column-wise:
#   work/<split>/recs.parquet     raw + normalized string columns (row order = S1, S2, S3 as in the files)
#   work/<split>/recs_num.npz     src, country codes, chain / co-location frequencies
#   work/train/gt.npz             ground-truth pairs as sorted int64 keys (i_row * N + j_row), #matches per row
# =============================================================================================================
RAW_COLS = ["entity_id", "name", "address", "country"]
NORM_COLS = ["n_full", "n_core", "n_legal", "n_skel", "n_skel2", "n_dba", "n_digits", "a_norm", "a_words",
             "a_nums", "a_postal", "a_house", "a_tail", "a_landmark"]


def record_text(name: str, address: str, country: str) -> str:
    """Serialization used by the cross-encoder. Raw text, not normalized."""
    return f"name: {name} | address: {address} | country: {country}"


def _norm_chunk(task):
    names, addrs = task
    out = {c: [] for c in NORM_COLS}
    for n, a in zip(names, addrs):
        d = normalize_record(n, a)
        for c in NORM_COLS:
            out[c].append(d[c])
    # Convert to Arrow HERE rather than in the parent. Returning lists of Python strings meant the parent
    # pickled and rebuilt ~300M str objects across the run and did every string->Arrow conversion itself,
    # serially, while the workers idled. An IPC buffer pickles as one memcpy.
    batch = pa.record_batch([pa.array(out[c], pa.string()) for c in NORM_COLS], names=NORM_COLS)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as w:
        w.write_batch(batch)
    return sink.getvalue().to_pybytes()


def prep_split(cfg: dict, split: str) -> None:
    t0 = time.time()
    d = cfg["train_dir"] if split == "train" else cfg["test_dir"]
    parts, stats = [], {}
    for s in (1, 2, 3):
        t = read_tsv_arrow(os.path.join(d, f"{split}_source{s}.tsv"), SRC_COLS)
        stats[f"rows_src{s}"] = t.num_rows
        vc = pc.value_counts(t["country"].combine_chunks()).to_pylist()
        stats[f"raw_country_src{s}"] = {x["values"]: x["counts"] for x in sorted(vc, key=lambda x: -x["counts"])[:10]}
        stats[f"empty_name_rate_src{s}"] = round(float(pc.mean(pc.equal(t["business_name"], "").cast(pa.int8())).as_py() or 0), 5)
        stats[f"empty_address_rate_src{s}"] = round(float(pc.mean(pc.equal(t["business_address"], "").cast(pa.int8())).as_py() or 0), 5)
        parts.append(t.append_column("src", pa.array(np.full(t.num_rows, s, np.int8))))
        log(f"  [{split}] source{s}: {t.num_rows:,} rows")
    T = pa.concat_tables(parts).combine_chunks()
    del parts
    codes = pc.dictionary_encode(T["entity_id"].combine_chunks()).indices.to_numpy()
    _, first = np.unique(codes, return_index=True)
    if len(first) < T.num_rows:
        log(f"WARNING {split}: {T.num_rows - len(first):,} duplicated entity_ids; keeping first occurrence")
        T = T.take(pa.array(np.sort(first)))
    stats["duplicate_ids_dropped"] = int(len(codes) - len(first))
    n = T.num_rows
    cenc = pc.dictionary_encode(T["country"].combine_chunks())
    ckeys = [country_key(v) for v in cenc.dictionary.to_pylist()]
    cty_names = sorted(set(ckeys))
    cmap = {c: k for k, c in enumerate(cty_names)}
    cty_code = np.array([cmap[c] for c in ckeys], np.int16)[cenc.indices.to_numpy()] if len(ckeys) else np.zeros(n, np.int16)
    cty_obj = np.array(cty_names, dtype=object)
    schema = pa.schema([(c, pa.string()) for c in RAW_COLS + ["cty"] + NORM_COLS] + [("src", pa.int8())])
    path = wpath(cfg, split, "recs")
    tmp = path + ".tmp"
    ch = int(cfg["prep"]["chunk"])
    names, addrs = T["business_name"], T["business_address"]
    tasks = ((names.slice(st, ch).to_pylist(), addrs.slice(st, ch).to_pylist()) for st in range(0, n, ch))
    ctx = _pool_ctx()
    pool = ctx.Pool(cfg["n_jobs"]) if (cfg["n_jobs"] > 1 and n > ch and ctx is not None) else None
    writer = pq.ParquetWriter(tmp, schema)
    st = 0
    n_chunks = math.ceil(n / ch)
    results = pool.imap(_norm_chunk, tasks) if pool else map(_norm_chunk, tasks)
    try:
        for buf in tqdm(results, total=n_chunks, desc=f"[{split}] normalizing", unit="chunk"):
            res = pa.ipc.open_stream(pa.BufferReader(pa.py_buffer(buf))).read_all()
            m = res.num_rows
            cols = {"entity_id": T["entity_id"].slice(st, m), "name": T["business_name"].slice(st, m),
                    "address": T["business_address"].slice(st, m), "country": T["country"].slice(st, m),
                    "cty": pa.array(cty_obj[cty_code[st:st + m]], pa.string())}
            cols.update({c: res[c].combine_chunks() for c in NORM_COLS})
            cols["src"] = T["src"].slice(st, m)
            writer.write_table(pa.table(cols, schema=schema))
            st += m
            if (st // ch) % 10 == 0 or st == n:
                log(f"  [{split}] normalized {st:,}/{n:,} ({st / (time.time() - t0):,.0f}/s)")
    finally:
        writer.close()
        if pool:
            pool.close()
            pool.join()
    os.replace(tmp, path)
    src = T["src"].to_numpy()
    # chain / co-location frequencies: same (country, core name) or (country, address) among S1 / among records
    R = pq.read_table(path, columns=["n_core", "a_norm"])
    freq = {}
    for col, pref in (("n_core", "name"), ("a_norm", "addr")):
        k = pc.dictionary_encode(R[col].combine_chunks()).indices.to_numpy().astype(np.int64)
        key = cty_code.astype(np.int64) * (int(k.max()) + 1 if len(k) else 1) + k
        _, inv = np.unique(key, return_inverse=True)
        nk = int(inv.max()) + 1 if len(inv) else 1
        freq[f"f_{pref}_s1"] = np.bincount(inv[src == 1], minlength=nk)[inv].astype(np.float32)
        if pref == "name":
            freq["f_name_r"] = np.bincount(inv[src != 1], minlength=nk)[inv].astype(np.float32)
    del R
    np.savez(wpath(cfg, split, "recs_num", ext="npz"), src=src, cty_code=cty_code,
             cty_names=np.array(cty_names, dtype=str), **freq)
    gt_path = os.path.join(d, f"{split}_ground_truth.tsv")
    if split == "train" and os.path.exists(gt_path):
        stats.update(_prep_gt(cfg, gt_path, T["entity_id"].combine_chunks(), n))
    with open(wpath(cfg, split, "prep_stats", ext="json"), "w") as f:
        json.dump(stats, f, indent=1)
    log(f"prep {split}: {n:,} records in {time.time() - t0:.0f}s; countries={cty_names}")


def _prep_gt(cfg: dict, gt_path: str, all_ids: pa.Array, n: int) -> dict:
    g = read_tsv_arrow(gt_path, GT_COLS)
    s1 = pc.index_in(g["source1_entity_id"].combine_chunks(), value_set=all_ids).fill_null(-1).to_numpy()
    lst = pc.split_pattern(g["matched_entity_ids"].combine_chunks(), ",")
    flat = pc.utf8_trim_whitespace(pc.list_flatten(lst))
    parent = pc.list_parent_indices(lst).to_numpy()
    nonempty = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
    rec = pc.index_in(flat, value_set=all_ids).fill_null(-1).to_numpy()
    i_rows = s1[parent]
    ok = nonempty & (i_rows >= 0) & (rec >= 0)
    keys = np.unique(i_rows[ok].astype(np.int64) * n + rec[ok].astype(np.int64))
    n_true = np.bincount(keys // n, minlength=n).astype(np.float32)
    np.savez(wpath(cfg, "train", "gt", ext="npz"), keys=keys, n_true=n_true)
    return {"gt_rows": g.num_rows, "gt_s1_ids_not_in_source1": int((s1 < 0).sum()),
            "gt_match_ids_not_in_sources": int((nonempty & (rec < 0)).sum()), "gt_pairs": int(len(keys)),
            "records_in_multiple_gt_lists": int((np.bincount(keys % n, minlength=n) > 1).sum())}


class Universe:
    def __init__(self, cfg: dict, split: str):
        self.cfg, self.split = cfg, split
        self.path = _need(wpath(cfg, split, "recs"), "prep")
        z = np.load(_need(wpath(cfg, split, "recs_num", ext="npz"), "prep"))
        self.src, self.cty_code = z["src"], z["cty_code"]
        self.cty_names = [str(x) for x in z["cty_names"]]
        self.freq = {k: z[k] for k in ("f_name_s1", "f_name_r", "f_addr_s1")}
        self.n = len(self.src)
        self.s1_rows = np.flatnonzero(self.src == 1)
        self.r_rows = np.flatnonzero(self.src != 1)
        self._cols: dict[str, StrCol] = {}
        gp = wpath(cfg, split, "gt", ext="npz")
        self.has_gt = split == "train" and os.path.exists(gp)
        if self.has_gt:
            g = np.load(gp)
            self.gt_keys, self.n_true_all = g["keys"], g["n_true"].astype(np.float64)

    def col(self, name: str) -> StrCol:
        if name not in self._cols:
            self._cols[name] = StrCol.from_arrow(pq.read_table(self.path, columns=[name])[name])
        return self._cols[name]

    def load(self, names) -> None:
        for c in names:
            self.col(c)

    def s(self, name: str, ix, dedup: bool = False) -> np.ndarray:
        ix = np.asarray(ix, np.int64)
        if dedup and len(ix) > 1:
            uq, inv = np.unique(ix, return_inverse=True)
            return np.array(self.col(name).take(uq), dtype=object)[inv]
        return np.array(self.col(name).take(ix), dtype=object)

    def labels(self, I, J) -> np.ndarray:
        if not self.has_gt or len(self.gt_keys) == 0:
            return np.zeros(len(I), np.int8)
        keys = np.asarray(I, np.int64) * self.n + np.asarray(J, np.int64)
        pos = np.minimum(np.searchsorted(self.gt_keys, keys), len(self.gt_keys) - 1)
        return (self.gt_keys[pos] == keys).astype(np.int8)

    def n_true_rows(self) -> np.ndarray:
        return self.n_true_all[self.s1_rows]

    def cty_str(self, ix) -> np.ndarray:
        return np.array(self.cty_names, dtype=object)[self.cty_code[ix]]


def build_universe(cfg: dict, split: str, force: bool = False) -> Universe:
    have = os.path.exists(wpath(cfg, split, "recs")) and os.path.exists(wpath(cfg, split, "recs_num", ext="npz"))
    if force or not have:
        prep_split(cfg, split)
    return Universe(cfg, split)


def get_folds(cfg: dict, u: Universe, scheme: str | None = None) -> np.ndarray:
    """Fold per row (-1 for non-S1). group: random over S1 entities; loco: one fold per country.
    Persisted by entity_id so folds never change between runs (CE OOF depends on them)."""
    scheme = scheme or cfg["cv"]["scheme"]
    fold = np.full(u.n, -1, np.int64)
    if scheme == "loco":
        fold[u.s1_rows] = u.cty_code[u.s1_rows]
        return fold
    path = wpath(cfg, "train", f"folds_{scheme}")
    ids = u.col("entity_id").to_arrow()
    if os.path.exists(path):
        f = pq.read_table(path)
        pos = pc.index_in(f["entity_id"].combine_chunks(), value_set=ids).fill_null(-1).to_numpy()
        ok = pos >= 0
        fold[pos[ok]] = f["fold"].to_numpy()[ok]
        if (fold[u.s1_rows] >= 0).all():
            return fold
        log("folds file does not match the current S1 set; regenerating")
    perm = np.random.RandomState(cfg["seed"]).permutation(len(u.s1_rows))
    fs = np.empty(len(u.s1_rows), np.int64)
    fs[perm] = np.arange(len(u.s1_rows)) % cfg["cv"]["n_folds"]
    fold[u.s1_rows] = fs
    pq.write_table(pa.table({"entity_id": pc.take(ids, pa.array(u.s1_rows)), "fold": pa.array(fs)}), path)
    return fold


def sample_mask(cfg: dict, u: Universe) -> np.ndarray:
    """S1 entities used to TRAIN models (deterministic). Every pair is still scored."""
    k = min(len(u.s1_rows), int(cfg["sample_s1"]))
    perm = np.random.RandomState(cfg["seed"] + 7).permutation(len(u.s1_rows))[:k]
    m = np.zeros(u.n, bool)
    m[u.s1_rows[perm]] = True
    return m


# =============================================================================================================
# EDA / AUDIT (array-based: fast at millions of rows)
# =============================================================================================================
def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1])


def _num(eid: str) -> int:
    m = re.search(r"(\d+)$", eid)
    return int(m.group(1)) if m else -1


def stage_eda(cfg: dict) -> dict:
    t0 = time.time()
    tr, te = build_universe(cfg, "train"), build_universe(cfg, "test")
    rep: dict = {}
    for u in (tr, te):
        with open(wpath(cfg, u.split, "prep_stats", ext="json")) as f:
            rep[f"{u.split}_prep"] = json.load(f)
        rep[f"{u.split}_sizes_by_src_country"] = pd.crosstab(u.src, u.cty_str(np.arange(u.n))).to_dict()
    nt = tr.n_true_rows()
    rep["n_s1_train"] = int(len(nt))
    rep["singleton_rate"] = round(float((nt == 0).mean()), 5)
    rep["mean_matches_per_s1"] = round(float(nt.mean()), 3)
    v, c = np.unique(np.minimum(nt, 10).astype(int), return_counts=True)
    rep["match_count_hist"] = {int(a): int(b) for a, b in zip(v, c)}
    gi, gj = tr.gt_keys // tr.n, tr.gt_keys % tr.n
    rep["records_in_multiple_gt_lists"] = int((np.bincount(gj, minlength=tr.n) > 1).sum())
    matched = np.zeros(tr.n, bool)
    matched[gj] = True
    rep["unmatched_record_rate"] = round(float(1 - matched[tr.r_rows].mean()), 5)
    rep["matched_by_source"] = {f"S{s}": int((tr.src[gj] == s).sum()) for s in (2, 3)}
    rep["gt_pairs_cross_country_rate"] = round(float((tr.cty_code[gi] != tr.cty_code[gj]).mean()), 5) if len(gi) else 0.0
    rng = np.random.default_rng(0)
    sub = rng.choice(len(gi), size=min(200000, len(gi)), replace=False) if len(gi) else np.zeros(0, int)
    rep["row_order_spearman_s1_vs_match"] = spearman(gi[sub], gj[sub])
    ids_i, ids_j = tr.s("entity_id", gi[sub[:50000]]), tr.s("entity_id", gj[sub[:50000]])
    rep["id_number_spearman_s1_vs_match"] = spearman([_num(x) for x in ids_i], [_num(x) for x in ids_j])
    s5 = sub[:50000]
    a, b = tr.s("n_core", gi[s5]), tr.s("n_core", gj[s5])
    rep["pos_exact_core_name_rate"] = round(float((a == b).mean()), 5) if len(s5) else None
    pa_, pb_ = tr.s("a_postal", gi[s5]), tr.s("a_postal", gj[s5])
    both = (pa_ != "") & (pb_ != "")
    rep["pos_postal_both_present_rate"] = round(float(both.mean()), 5) if len(s5) else None
    rep["pos_postal_equal_given_present"] = round(float((pa_[both] == pb_[both]).mean()), 5) if both.any() else None
    rep["s1_share_with_duplicate_core_name"] = round(float((tr.freq["f_name_s1"][tr.s1_rows] > 1).mean()), 5)
    for u, key in ((tr, "train_country_share"), (te, "test_country_share")):
        cs = pd.Series(u.cty_str(u.s1_rows)).value_counts(normalize=True).round(4)
        rep[key] = cs.to_dict()
    rep["unseen_test_countries"] = sorted(set(te.cty_names) - set(tr.cty_names))
    rep["train_to_test_size_ratio"] = round(tr.n / max(1, te.n), 3)
    rep["advice"] = []
    if rep["records_in_multiple_gt_lists"] > 0:
        rep["advice"].append("exclusivity violated in GT: decision tuning will choose excl_mode=none if better")
    if rep["gt_pairs_cross_country_rate"] > 0.01:
        rep["advice"].append("many cross-country GT pairs: run with --set blocking.within_country=false")
    if abs(rep["row_order_spearman_s1_vs_match"]) > 0.3 or abs(rep["id_number_spearman_s1_vs_match"] or 0) > 0.3:
        rep["advice"].append("IDs/row order correlate with matches: keep features ID-free (they are)")
    with open(os.path.join(cfg["work_dir"], "eda.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    for k, v in rep.items():
        print(f"{k:40s} {v}")
    log_exp(cfg, "eda", {k: rep[k] for k in ("singleton_rate", "mean_matches_per_s1", "records_in_multiple_gt_lists",
                                            "gt_pairs_cross_country_rate")}, t0)
    return rep

# =============================================================================================================
# PAIR STORE: one row per candidate pair (sorted by S1 row, then record row); one .npy per column in
# work/<split>/pairs/ (memory-mappable, shared by forked workers).
# =============================================================================================================
class PairStore:
    def __init__(self, cfg: dict, split: str):
        self.dir = os.path.join(cfg["work_dir"], split, "pairs")
        os.makedirs(self.dir, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, f"{name}.npy")

    def save(self, name: str, arr: np.ndarray) -> None:
        np.save(self.path(name), arr)

    def has(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    def load(self, name: str, mmap: bool = False) -> np.ndarray:
        return np.load(_need(self.path(name), "block"), mmap_mode="r" if mmap else None)

    def n(self) -> int:
        return int(self.load("i", mmap=True).shape[0])

    def clear(self) -> None:
        for p in glob.glob(os.path.join(self.dir, "*.npy")):
            os.remove(p)


# =============================================================================================================
# BLOCKING: hashed token views (no vocabulary needed), IDF-weighted, cosine-normalized, per country.
#   name view : core-name words (n), 4-char prefixes (p), phonetic skeletons (k)
#   addr view : address words (a), postal code (z), house number + first street word (h)
#   full view : name + addr + mixed tokens (first name word + postal prefix / last address words)
# Tokens more frequent than max_df (and, if needed, the most frequent ones beyond a work budget) are not used
# for retrieval, which keeps the sparse products tractable at millions of records.
# Top-k in both directions: S1 -> records (k_s1) and records -> S1 (k_rec). Every pair is then scored by
# every view, plus rarity-aware overlap masses and two fuzzy scores, with rank/gap/margin context.
# =============================================================================================================
TOKEN_SRC = ["n_core", "n_skel", "n_skel2", "a_words", "a_postal", "a_house"]
VIEW_BITS = {("full", "s1"): 0, ("name", "s1"): 1, ("addr", "s1"): 2, ("full", "rec"): 3, ("name", "rec"): 4,
             ("addr", "rec"): 5}
SCORE_COLS = ["bs_full", "bs_name", "bs_addr", "tset_n_core", "tset_a_norm"]


def _hash_csr(rows: list, toks: list, n_rows: int, bits: int) -> sp.csr_matrix:
    m = 1 << bits
    if not toks:
        return sp.csr_matrix((n_rows, m), dtype=np.float32)
    h = pd.util.hash_array(np.array(toks, dtype=object), categorize=False)
    cols = (h % np.uint64(m)).astype(np.int32)
    X = sp.csr_matrix((np.ones(len(cols), np.float32), (np.asarray(rows, np.int32), cols)), shape=(n_rows, m))
    X.sum_duplicates()
    X.data[:] = 1.0
    return X


def _tok_chunk(task):
    st, en, bits = task
    cols = _G["cols"]
    idx = np.arange(st, en)
    nc, sk, sk2, aw, zc, hn = (cols[c].take(idx) for c in TOKEN_SRC)
    rn, tn, ra, ta, rx, tx = [], [], [], [], [], []
    for r in range(en - st):
        core = nc[r].split()
        for t in core:
            rn.append(r)
            tn.append("n" + t)
            if len(t) >= 5:
                rn.append(r)
                tn.append("p" + t[:4])
        for s_ in sk[r].split():
            if len(s_) >= 2:
                rn.append(r)
                tn.append("k" + s_)
        for s_ in sk2[r].split():          # voicing-folded key: Dravidian-script recall
            if len(s_) >= 2:
                rn.append(r)
                tn.append("v" + s_)
        words = aw[r].split()
        for t in words:
            ra.append(r)
            ta.append("a" + t)
        z, h = zc[r], hn[r]
        if z:
            ra.append(r)
            ta.append("z" + z)
        if h and words:
            ra.append(r)
            ta.append("h" + h + "|" + words[0])
        if core:
            f0 = core[0]
            if z:
                rx.append(r)
                tx.append("x" + f0 + "|" + z[:3])
            for w in words[-2:]:
                rx.append(r)
                tx.append("y" + f0 + "|" + w)
    n = en - st
    return _hash_csr(rn, tn, n, bits), _hash_csr(ra, ta, n, bits), _hash_csr(rx, tx, n, bits)


def token_matrices(u: Universe, cfg: dict) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    bits = int(cfg["blocking"]["hash_bits"])
    u.load(TOKEN_SRC)
    ch = 100000
    tasks = [(st, min(u.n, st + ch), bits) for st in range(0, u.n, ch)]
    res = parallel_map(_tok_chunk, tasks, cfg["n_jobs"], {"cols": {c: u.col(c) for c in TOKEN_SRC}})
    return tuple(sp.vstack([r[k] for r in res], format="csr") for k in range(3))


def _row_sums(M: sp.csr_matrix, square: bool = False) -> np.ndarray:
    d = M.data.astype(np.float64)
    if square:
        d = d * d
    if len(d) == 0:
        return np.zeros(M.shape[0])
    s = np.add.reduceat(d, np.minimum(M.indptr[:-1], len(d) - 1))
    s[np.diff(M.indptr) == 0] = 0.0
    return s


def _normalize(W: sp.csr_matrix) -> sp.csr_matrix:
    nr = np.sqrt(_row_sums(W, square=True))
    nr[nr == 0] = 1.0
    return (sp.diags((1.0 / nr).astype(np.float32)) @ W).tocsr()


def _weigh(X: sp.csr_matrix, w: np.ndarray, normalize: bool) -> sp.csr_matrix:
    W = X.copy()
    W.data = w[W.indices].astype(np.float32)
    W.eliminate_zeros()
    return _normalize(W) if normalize else W


def _retrieval_idf(df_a: np.ndarray, df_b: np.ndarray, n: int, max_df: float, budget: float) -> tuple[np.ndarray, np.ndarray]:
    df = df_a + df_b
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    w = idf.copy()
    w[(df > max_df) | (df == 0)] = 0.0
    cost = df_a.astype(np.float64) * df_b
    alive = np.flatnonzero(w > 0)
    tot = cost[alive].sum()
    if tot > budget:  # drop the most frequent surviving tokens until the product budget is met
        order = alive[np.argsort(-df[alive], kind="stable")]
        cum = np.cumsum(cost[order])
        cut = int(np.searchsorted(cum, tot - budget)) + 1
        w[order[:cut]] = 0.0
    return idf, w


def topn(A: sp.csr_matrix, B: sp.csr_matrix, k: int, n_jobs: int):
    """Top-k cosine per row of A against rows of B -> (row_a, row_b, score)."""
    if A.shape[0] == 0 or B.shape[0] == 0 or k <= 0 or A.nnz == 0 or B.nnz == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    k = min(k, B.shape[0])
    try:
        from sparse_dot_topn import sp_matmul_topn
        C = sp_matmul_topn(A.astype(np.float32), B.T.tocsr().astype(np.float32), top_n=k, n_threads=max(1, n_jobs))
        C = C.tocsr()
        C.eliminate_zeros()
        rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        return rows.astype(np.int64), C.indices.astype(np.int64), C.data.astype(np.float32)
    except ImportError:
        return _topn_scipy(A, B, k)


def _topn_scipy(A, B, k, chunk: int = 2000):
    """Fallback without sparse_dot_topn (much slower at scale: pip install sparse_dot_topn)."""
    BT = B.T.tocsr()
    R, Cc, D = [], [], []
    for st in range(0, A.shape[0], chunk):
        M = (A[st:st + chunk] @ BT).tocsr()
        M.eliminate_zeros()
        rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        order = np.lexsort((-M.data, rows))
        r = rows[order]
        first = np.r_[True, r[1:] != r[:-1]] if len(r) else np.zeros(0, bool)
        starts = np.flatnonzero(first)
        rank = np.arange(len(r)) - starts[np.cumsum(first) - 1] if len(r) else np.zeros(0, np.int64)
        keep = order[rank < k]
        R.append(rows[keep] + st)
        Cc.append(M.indices[keep])
        D.append(M.data[keep])
    if not R:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(R).astype(np.int64), np.concatenate(Cc).astype(np.int64), np.concatenate(D).astype(np.float32)


def _rowdot_task(task):
    st, en = task
    X, Y, a, b = _G["X"], _G["Y"], _G["a"], _G["b"]
    return st, np.asarray(X[a[st:en]].multiply(Y[b[st:en]]).sum(1)).ravel().astype(np.float32)


def rowwise_dot(X: sp.csr_matrix, Y: sp.csr_matrix, a: np.ndarray, b: np.ndarray, chunk: int = 400000,
                n_jobs: int = 1) -> np.ndarray:
    """Per-pair cosine between selected rows of X and Y.

    Called once per blocking view per country group, so at full scale it is a few hundred million sparse row
    products -- enough to matter, and previously all on one core. Forked workers share the CSR matrices and the
    index arrays copy-on-write, so each chunk costs only its own output array to ship back.
    """
    out = np.zeros(len(a), np.float32)
    tasks = [(st, min(len(a), st + chunk)) for st in range(0, len(a), chunk)]
    if n_jobs > 1 and len(tasks) > 1:
        for st, v in parallel_map(_rowdot_task, tasks, n_jobs, {"X": X, "Y": Y, "a": a, "b": b}):
            out[st:st + len(v)] = v
        return out
    for st, en in tasks:
        out[st:en] = np.asarray(X[a[st:en]].multiply(Y[b[st:en]]).sum(1)).ravel()
    return out


def country_groups(u: Universe, within_country: bool) -> list[tuple[str, np.ndarray]]:
    """Row sets per blocking group. Records with an empty country join every group."""
    if not within_country:
        return [("all", np.arange(u.n))]
    unk = u.cty_names.index("unk") if "unk" in u.cty_names else -1
    named = [k for k in range(len(u.cty_names)) if k != unk]
    if not named:
        return [("unk", np.arange(u.n))]
    is_unk = u.cty_code == unk
    return [(u.cty_names[k], np.flatnonzero((u.cty_code == k) | is_unk)) for k in named]


def group_context(keys: np.ndarray, vals: np.ndarray, fill: float = -1.0):
    """Per row: rank within its key group (1 = best), gap to group max, margin to the group's 2nd best, group size."""
    n = len(keys)
    if n == 0:
        z = np.zeros(0, np.float32)
        return z, z, z, z
    order = np.lexsort((-vals, keys))
    k, v = keys[order], vals[order].astype(np.float64)
    first = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    size = np.diff(np.r_[starts, n])
    sec = np.where(size >= 2, v[np.minimum(starts + 1, n - 1)], fill)
    out = [np.empty(n, np.float32) for _ in range(4)]
    out[0][order] = np.arange(n) - starts[gid] + 1
    out[1][order] = v - v[starts][gid]
    out[2][order] = v - sec[gid]
    out[3][order] = size[gid]
    return tuple(out)


def _cpdist_task(task):
    """Decode this chunk's strings AND score it, inside the worker."""
    from rapidfuzz import process
    st, en = task
    u, cols, scorer, I, J = _G["u"], _G["cols"], _G["scorer"], _G["I"], _G["J"]
    out = []
    for col in cols:
        c = u.col(col)
        out.append(process.cpdist(c.take(I[st:en]), c.take(J[st:en]), scorer=scorer, workers=1,
                                  dtype=np.float32))
    return st, out


def _cpdist_chunked(u: Universe, I: np.ndarray, J: np.ndarray, cols, scorer, n_jobs: int,
                    chunk: int = 2000000) -> dict[str, np.ndarray]:
    """Token-set similarity for every pair, on one or more columns.

    `cpdist` threads internally, but materializing the Python strings for each chunk did not -- at ~100M pairs
    that decode is hundreds of millions of str objects on a single core. Doing the decode inside the same
    worker that scores the chunk keeps every core busy and ships back only the float32 results.
    """
    cols = [cols] if isinstance(cols, str) else list(cols)
    u.load(cols)
    out = {c: np.zeros(len(I), np.float32) for c in cols}
    tasks = [(st, min(len(I), st + chunk)) for st in range(0, len(I), chunk)]
    if n_jobs > 1 and len(tasks) > 1:
        res = parallel_map(_cpdist_task, tasks, n_jobs,
                           {"u": u, "cols": cols, "scorer": scorer, "I": I, "J": J})
        for st, vals in res:
            for c, v in zip(cols, vals):
                out[c][st:st + len(v)] = v
        return out
    from rapidfuzz import process
    for st, en in tasks:
        for c in cols:
            sc = u.col(c)
            out[c][st:en] = process.cpdist(sc.take(I[st:en]), sc.take(J[st:en]), scorer=scorer,
                                           workers=n_jobs, dtype=np.float32)
    return out


def generate_candidates(u: Universe, cfg: dict) -> dict[str, np.ndarray]:
    cb = cfg["blocking"]
    t0 = time.time()
    Xn, Xa, Xm = token_matrices(u, cfg)
    log(f"  [{u.split}] token matrices: name nnz={Xn.nnz:,} addr nnz={Xa.nnz:,} mix nnz={Xm.nnz:,} "
        f"({time.time() - t0:.0f}s)")
    out = {k: [] for k in ("i", "j", "found", "bs_full", "bs_name", "bs_addr", "name_shared", "name_unm_a",
                           "name_unm_b", "addr_shared", "addr_unm_a", "addr_unm_b")}
    groups = country_groups(u, cb.get("within_country", True))
    # Miss diagnosis. A true pair can be absent from the candidate set for two very different reasons, and the
    # fixes are opposite: if it shares no surviving token in any view its cosine is 0 everywhere and no k will
    # ever retrieve it (vocabulary/budget problem), whereas if its cosine is positive it was merely ranked out
    # (k problem). Recall alone cannot tell these apart, so measure it directly on the ground-truth pairs --
    # exact, not sampled, and cheap next to the work already being done over every candidate pair.
    want_diag = bool(cb.get("diagnose", False)) and getattr(u, "has_gt", False) and len(getattr(u, "gt_keys", ()))
    dg_key, dg_reach, dg_got = [], [], []
    if want_diag:
        gt_i, gt_j = u.gt_keys // u.n, u.gt_keys % u.n
    for g, rows in tqdm(groups, desc=f"[{u.split}] blocking by country", unit="group"):
        tg = time.time()
        src = u.src[rows]
        s1l, rl = rows[src == 1], rows[src != 1]
        if len(s1l) == 0 or len(rl) == 0:
            continue
        n_g = len(rows)
        W, U = {}, {}
        for view, X in (("name", Xn), ("addr", Xa), ("mix", Xm)):
            A, B = X[s1l], X[rl]
            df_a = np.bincount(A.indices, minlength=X.shape[1]).astype(np.float64)
            df_b = np.bincount(B.indices, minlength=X.shape[1]).astype(np.float64)
            idf, w = _retrieval_idf(df_a, df_b, n_g, cb["max_df"], cb["max_products"])
            W[view] = (_weigh(A, w, True), _weigh(B, w, True))
            if view != "mix":  # rarity-aware overlap uses ALL tokens (uncapped IDF)
                U[view] = (A, B, _weigh(A, idf, False), _weigh(B, idf, False))
        full = tuple(_normalize(sp.hstack([W["name"][s], W["addr"][s], W["mix"][s]], format="csr")) for s in (0, 1))
        W["full"] = full
        keys, bitv = [], []
        for view in ("full", "name", "addr"):
            A, B = W[view]
            ra, rb, _ = topn(A, B, cb["k_s1"][view], cfg["n_jobs"])
            keys.append(s1l[ra].astype(np.int64) * u.n + rl[rb])
            bitv.append(np.full(len(ra), VIEW_BITS[(view, "s1")], np.int8))
            rb2, ra2, _ = topn(B, A, cb["k_rec"][view], cfg["n_jobs"])
            keys.append(s1l[ra2].astype(np.int64) * u.n + rl[rb2])
            bitv.append(np.full(len(ra2), VIEW_BITS[(view, "rec")], np.int8))
        kb = np.unique(np.concatenate(keys) * 8 + np.concatenate(bitv))
        key, bit = kb // 8, (kb % 8).astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        found = np.bincount(inv, weights=(1 << bit).astype(np.float64), minlength=len(uk)).astype(np.int16)
        I, J = uk // u.n, uk % u.n
        ia, jb = np.searchsorted(s1l, I), np.searchsorted(rl, J)
        out["i"].append(I)
        out["j"].append(J)
        out["found"].append(found)
        nj = cfg["n_jobs"]
        for view in ("full", "name", "addr"):
            out[f"bs_{view}"].append(rowwise_dot(W[view][0], W[view][1], ia, jb, n_jobs=nj))
        for view in ("name", "addr"):
            A, B, Ai, Bi = U[view]
            shared = rowwise_dot(A, Bi, ia, jb, n_jobs=nj)
            out[f"{view}_shared"].append(shared)
            out[f"{view}_unm_a"].append(np.maximum(_row_sums(Ai)[ia] - shared, 0).astype(np.float32))
            out[f"{view}_unm_b"].append(np.maximum(_row_sums(Bi)[jb] - shared, 0).astype(np.float32))
        if want_diag and len(uk):
            m = np.isin(gt_i, s1l) & np.isin(gt_j, rl)
            if m.any():
                ti, tj = gt_i[m], gt_j[m]
                ia_t, jb_t = np.searchsorted(s1l, ti), np.searchsorted(rl, tj)
                reach = np.zeros(len(ti), bool)
                for view in ("full", "name", "addr"):
                    reach |= rowwise_dot(W[view][0], W[view][1], ia_t, jb_t, n_jobs=nj) > 0
                tk = ti.astype(np.int64) * u.n + tj
                sp_ = np.searchsorted(uk, tk)
                got = uk[np.minimum(sp_, len(uk) - 1)] == tk
                dg_key.append(tk)
                dg_reach.append(reach)
                dg_got.append(got)
                nm = int((~got).sum())
                log(f"  [{u.split}] group {g}: {len(ti):,} true pairs, {int(got.sum()):,} retrieved, "
                    f"{nm:,} missed ({int(((~got) & ~reach).sum()):,} unreachable, "
                    f"{int(((~got) & reach).sum()):,} ranked out)")
        log(f"  [{u.split}] group {g}: {len(s1l):,} S1 x {len(rl):,} records -> {len(uk):,} pairs "
            f"({len(uk) / len(s1l):.1f}/S1, {time.time() - tg:.0f}s)")
        del W, U, full
    if not out["i"]:
        return {k: np.zeros(0, np.float32 if k.startswith(("bs_", "name_", "addr_")) else np.int64) for k in out}
    if dg_key:
        # A record with an unknown country joins every group, so the same true pair can be measured more than
        # once; dedupe by key, counting a pair as reachable/retrieved if it was so in ANY group.
        k_all = np.concatenate(dg_key)
        uq, inv = np.unique(k_all, return_inverse=True)
        agg = lambda v: np.bincount(inv, weights=np.concatenate(v).astype(np.float64), minlength=len(uq)) > 0
        reach, got = agg(dg_reach), agg(dg_got)
        n_t = max(1, len(uq))
        u.block_diag = {
            "diag_true_pairs": int(n_t),
            "diag_retrieved": round(float(got.sum()) / n_t, 5),
            "diag_miss_unreachable": round(float((~got & ~reach).sum()) / n_t, 5),
            "diag_miss_ranked_out": round(float((~got & reach).sum()) / n_t, 5),
            "diag_reach_ceiling": round(float(reach.sum()) / n_t, 5),  # best recall any k could reach
        }
    res = {k: np.concatenate(v) for k, v in out.items()}
    key = res["i"] * u.n + res["j"]
    _, first = np.unique(key, return_index=True)  # unknown-country rows can appear in several groups
    order = first[np.lexsort((res["j"][first], res["i"][first]))]
    return {k: v[order] for k, v in res.items()}


def stage_block(cfg: dict) -> None:
    from rapidfuzz import fuzz
    t0 = time.time()
    for split in ("train", "test"):
        u = build_universe(cfg, split)
        C = generate_candidates(u, cfg)
        ps = PairStore(cfg, split)
        ps.clear()
        I, J = C.pop("i"), C.pop("j")
        n_pairs = len(I)
        ps.save("i", I.astype(np.int32 if u.n < 2 ** 31 else np.int64))
        ps.save("j", J.astype(np.int32 if u.n < 2 ** 31 else np.int64))
        ps.save("found", C.pop("found").astype(np.int16))
        for k, v in C.items():
            ps.save(k, v.astype(np.float32))
        del C
        tsets = _cpdist_chunked(u, I, J, ("n_core", "a_norm"), fuzz.token_set_ratio, cfg["n_jobs"])
        for col, name in (("n_core", "tset_n_core"), ("a_norm", "tset_a_norm")):
            ps.save(name, tsets[col])
        del tsets
        for c in SCORE_COLS:  # rank / gap / margin within the S1 group and within the record group
            v = ps.load(c)
            for side, keys in (("i", I), ("j", J)):
                rk, gap, m2, size = group_context(keys, v)
                ps.save(f"rk_{side}_{c}", rk)
                ps.save(f"gap_{side}_{c}", gap)
                ps.save(f"m2_{side}_{c}", m2)
                if c == SCORE_COLS[0]:
                    ps.save(f"n_cand_{side}", size)
        if u.has_gt:
            L = u.labels(I, J)
            ps.save("label", L)
            rep = recall_report(u, I, J, L, ps.load("found"))
            rep.update(getattr(u, "block_diag", {}))
            log(f"block train: {json.dumps(rep)}")
            if "diag_reach_ceiling" in rep:
                log(f"  recall ceiling with this vocabulary: {rep['diag_reach_ceiling']:.4f} "
                    f"({rep['diag_miss_unreachable']:.4f} of true pairs share no surviving token in any view "
                    f"-> raise max_df/max_products, not k; "
                    f"{rep['diag_miss_ranked_out']:.4f} are reachable but ranked out -> raise k_rec/k_s1)")
            log_exp(cfg, "block", rep, t0, note=json.dumps(cfg["blocking"], default=str))
        else:
            log(f"block {split}: {n_pairs:,} pairs, {n_pairs / max(1, len(u.s1_rows)):.1f} per S1")


def recall_report(u: Universe, I, J, L, found) -> dict:
    n_true = max(1.0, float(u.n_true_rows().sum()))
    pos = L == 1
    rep = {"n_pairs": int(len(I)), "pairs_per_s1": round(len(I) / max(1, len(u.s1_rows)), 2),
           "recall": round(float(pos.sum()) / n_true, 5)}
    for (view, direction), b in VIEW_BITS.items():
        rep[f"recall_{view}_{direction}"] = round(float((((found[pos] >> b) & 1) > 0).sum()) / n_true, 5)
    for k, name in enumerate(u.cty_names):
        s1 = u.s1_rows[u.cty_code[u.s1_rows] == k]
        if len(s1):
            rep[f"recall_country_{name}"] = round(float((pos & (u.cty_code[I] == k)).sum()) / max(1.0, u.n_true_all[s1].sum()), 5)
    return rep

# =============================================================================================================
# PAIR FEATURES: computed by forked workers over S1-aligned chunks of the pair store. Each worker reads its
# rows from the memory-mapped pair columns, decodes only the strings it needs from the shared StrCol buffers,
# and writes work/<split>/feats/part_XXXXX.parquet (row id r = position in the pair store).
# =============================================================================================================
STR_FEATURE_COLS = ["name", "n_full", "n_core", "n_skel", "n_legal", "n_dba", "n_digits", "a_norm", "a_words",
                    "a_nums", "a_postal", "a_house", "a_tail", "a_landmark"]


def _tri(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """+1 agree, -1 conflict, 0 at least one side missing."""
    both = (a != "") & (b != "")
    return np.where(both, np.where(a == b, 1, -1), 0).astype(np.int8)


def _lens(arr) -> np.ndarray:
    return np.fromiter((len(s) for s in arr), dtype=np.int32, count=len(arr))


def _acronym(s: str) -> str:
    return "".join(t[0] for t in s.split() if not t.isdigit())


def string_features(u: Universe, I: np.ndarray, J: np.ndarray, workers: int = 1) -> dict[str, np.ndarray]:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    n = len(I)
    cache: dict = {}

    def col(c, side):
        if (c, side) not in cache:
            cache[(c, side)] = u.s(c, I, dedup=True) if side == "i" else u.s(c, J)
        return cache[(c, side)]

    cp = lambda a, b, sc: process.cpdist(list(a), list(b), scorer=sc, workers=workers, dtype=np.float32)  # noqa: E731
    F: dict[str, np.ndarray] = {}
    for v in ("n_full", "n_core", "n_skel", "a_norm", "a_words", "a_tail"):
        a, b = col(v, "i"), col(v, "j")
        F[f"ratio_{v}"] = cp(a, b, fuzz.ratio)
        if v not in ("n_core", "a_norm"):  # token_set on these two is computed globally at blocking time
            F[f"tset_{v}"] = cp(a, b, fuzz.token_set_ratio)
        if v in ("n_core", "a_words"):
            F[f"tsort_{v}"] = cp(a, b, fuzz.token_sort_ratio)
            F[f"partial_{v}"] = cp(a, b, fuzz.partial_ratio)
            F[f"jw_{v}"] = cp(a, b, JaroWinkler.normalized_similarity)
            F[f"lev_{v}"] = cp(a, b, Levenshtein.normalized_similarity)
    ca, cb_ = col("n_core", "i"), col("n_core", "j")
    F["wratio_n_core"] = cp(ca, cb_, fuzz.WRatio)
    F["ratio_raw_name"] = cp([s.lower() for s in col("name", "i")], [s.lower() for s in col("name", "j")], fuzz.ratio)
    la, lb = col("a_landmark", "i"), col("a_landmark", "j")
    lm = cp(la, lb, fuzz.token_set_ratio)
    F["tset_landmark"] = np.where((la != "") & (lb != ""), lm, np.nan).astype(np.float32)
    F["landmark_present"] = ((la != "").astype(np.int8) + (lb != "").astype(np.int8)).astype(np.int8)
    da, db = col("n_dba", "i"), col("n_dba", "j")
    dba = np.full(n, np.nan, np.float32)
    for r in np.flatnonzero((da != "") | (db != "")):
        xa = da[r].split("|") if da[r] else [ca[r]]
        xb = db[r].split("|") if db[r] else [cb_[r]]
        dba[r] = max(fuzz.token_set_ratio(x, y) for x in xa for y in xb)
    F["dba_best"] = dba
    pa_, pb_ = col("a_postal", "i"), col("a_postal", "j")
    F["postal_tri"] = _tri(pa_, pb_)
    both = (pa_ != "") & (pb_ != "")
    F["postal_prefix3"] = np.where(both, np.array([x[:3] == y[:3] for x, y in zip(pa_, pb_)], dtype=bool), -1).astype(np.int8)
    F["postal_lev"] = np.where(both, cp(pa_, pb_, Levenshtein.distance), np.nan).astype(np.float32)
    ha, hb = col("a_house", "i"), col("a_house", "j")
    F["house_tri"] = _tri(ha, hb)
    # House numbers agree outright in only 76% (India) / 87% (US) of true pairs, but a further ~6% differ by a
    # single digit -- a typo, not a different building. A hard equal/not-equal flag throws that signal away.
    bh = (ha != "") & (hb != "")
    F["house_lev"] = np.where(bh, cp(ha, hb, Levenshtein.distance), np.nan).astype(np.float32)
    F["house_lendiff"] = np.where(bh, np.abs(_lens(ha) - _lens(hb)), -1).astype(np.int8)
    F["legal_tri"] = _tri(col("n_legal", "i"), col("n_legal", "j"))
    F["eq_n_core"] = (ca == cb_).astype(np.int8)
    sa = np.array([" ".join(sorted(s.split())) for s in ca], dtype=object)
    sb = np.array([" ".join(sorted(s.split())) for s in cb_], dtype=object)
    F["eq_n_core_sorted"] = (sa == sb).astype(np.int8)
    F["eq_n_skel"] = (col("n_skel", "i") == col("n_skel", "j")).astype(np.int8)
    F["eq_a_norm"] = (col("a_norm", "i") == col("a_norm", "j")).astype(np.int8)
    aa = np.array([_acronym(s) for s in ca], dtype=object)
    ab = np.array([_acronym(s) for s in cb_], dtype=object)
    nsa = np.array([s.replace(" ", "") for s in ca], dtype=object)
    nsb = np.array([s.replace(" ", "") for s in cb_], dtype=object)
    F["acronym_hit"] = (((aa == nsb) & (_lens(nsb) >= 2)) | ((ab == nsa) & (_lens(nsa) >= 2))).astype(np.int8)
    F["name_digits_tri"] = _tri(col("n_digits", "i"), col("n_digits", "j"))
    na_, nb_ = col("a_nums", "i"), col("a_nums", "j")
    F["addr_nums_jacc"] = np.array([len(set(x.split()) & set(y.split())) / max(1, len(set(x.split()) | set(y.split())))
                                    if (x or y) else np.nan for x, y in zip(na_, nb_)], dtype=np.float32)
    for v in ("n_core", "a_norm"):
        l1, l2 = _lens(col(v, "i")), _lens(col(v, "j"))
        F[f"len_{v}_a"], F[f"len_{v}_b"] = l1.astype(np.int16), l2.astype(np.int16)
        F[f"lendiff_{v}"] = np.abs(l1 - l2).astype(np.int16)
    F["src_b"] = u.src[J].astype(np.int8)
    F["freq_name_s1"] = u.freq["f_name_s1"][I]
    F["freq_name_r"] = u.freq["f_name_r"][J]
    F["freq_addr_s1"] = u.freq["f_addr_s1"][I]
    return F


def _feat_chunk(task):
    k, a, b, out_path = task
    u, pdir, cols = _G["u"], _G["pdir"], _G["cols"]
    P = {c: np.array(np.load(os.path.join(pdir, f"{c}.npy"), mmap_mode="r")[a:b]) for c in cols}
    I, J = P["i"].astype(np.int64), P["j"].astype(np.int64)
    F = string_features(u, I, J, workers=_G.get("workers", 1))
    found = P.pop("found").astype(np.int16)
    for (view, direction), bit in VIEW_BITS.items():
        F[f"found_{view}_{direction}"] = ((found >> bit) & 1).astype(np.int8)
    df = pd.DataFrame({"r": np.arange(a, b, dtype=np.int64), **P, **F})
    df.to_parquet(out_path, index=False)
    return k, b - a


def avail_ram_gb() -> float | None:
    """Memory this process may actually use, in GB, or None if it cannot be determined.

    /proc/meminfo reports the whole node. Under SLURM or any cgroup the job is capped far below that, and
    trusting MemAvailable there means the feat guard never fires and the job is OOM-killed instead. Take the
    smallest of every limit that applies.
    """
    caps: list[float] = []
    v = os.environ.get("SLURM_MEM_PER_NODE")
    if v and v.isdigit():
        caps.append(int(v) / 1024)                      # SLURM reports MB
    v, c = os.environ.get("SLURM_MEM_PER_CPU"), os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit() and c and c.isdigit():
        caps.append(int(v) * int(c) / 1024)
    for p in ("/sys/fs/cgroup/memory.max",                       # cgroup v2
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):    # cgroup v1
        try:
            with open(p) as f:
                t = f.read().strip()
            if t.isdigit() and int(t) < 2 ** 60:       # "max"/huge sentinel means no limit
                caps.append(int(t) / 1024 ** 3)
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    caps.append(int(line.split()[1]) / (1024 ** 2))
                    break
    except Exception:
        pass
    return min(caps) if caps else None


# Each feat worker holds its chunk's decoded strings (14 columns x 2 sides of Python str objects), the ~100
# output feature columns and a parquet write buffer. Measured at roughly this many bytes per pair per worker.
FEAT_BYTES_PER_PAIR = 2200


def fit_feat_chunk(chunk: int, n_workers: int, budget_frac: float = 0.6) -> int:
    """Shrink features.chunk_pairs until n_workers can run concurrently inside available RAM.

    chunk_pairs=1,000,000 on a 48-core box projects to >100 GB in this stage alone, which OOMs hours into a
    run. Fewer, smaller chunks keep every core busy at a fraction of the peak.
    """
    ram = avail_ram_gb()
    if not ram or n_workers < 1:
        return chunk
    budget = ram * budget_frac * (1024 ** 3)
    safe = int(budget / (n_workers * FEAT_BYTES_PER_PAIR))
    safe = max(50_000, safe - safe % 10_000)
    if safe >= chunk:
        return chunk
    log(f"  feat: {n_workers} workers x chunk_pairs={chunk:,} projects to "
        f"~{n_workers * chunk * FEAT_BYTES_PER_PAIR / 1024 ** 3:.0f} GB but only {ram:.0f} GB is available; "
        f"reducing chunk_pairs to {safe:,} (~{n_workers * safe * FEAT_BYTES_PER_PAIR / 1024 ** 3:.0f} GB). "
        f"Override with --set features.chunk_pairs=N")
    return safe


def _chunk_bounds(I: np.ndarray, chunk: int) -> list[int]:
    """Row boundaries of ~chunk pairs that never split an S1 group (pairs are sorted by i)."""
    n = len(I)
    b = [0]
    while b[-1] < n:
        t = min(n, b[-1] + chunk)
        if t < n:
            t2 = int(np.searchsorted(I, I[t], side="left"))
            t = t2 if t2 > b[-1] else int(np.searchsorted(I, I[t], side="right"))
        b.append(t)
    return b


def parts_meta(cfg: dict, split: str) -> list[tuple[str, int, int]]:
    with open(_need(os.path.join(cfg["work_dir"], split, "feats", "parts.json"), "feat")) as f:
        return [tuple(x) for x in json.load(f)["parts"]]


def read_parts(cfg: dict, split: str, columns: list[str] | None = None, prefetch: int = 2):
    """Stream the feature parts, reading ahead on a background thread.

    Every caller does real work between parts (filtering, LightGBM prediction), and parquet reads release the
    GIL, so overlapping the next read with the current part's compute removes the disk wait almost entirely.
    """
    meta = parts_meta(cfg, split)
    if prefetch < 1 or len(meta) < 2:
        for path, a, b in meta:
            yield pd.read_parquet(path, columns=columns), a, b
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=prefetch) as ex:
        futures = [None] * len(meta)
        for k in range(min(prefetch, len(meta))):
            futures[k] = ex.submit(pd.read_parquet, meta[k][0], columns=columns)
        for k, (path, a, b) in enumerate(meta):
            df = futures[k].result()
            futures[k] = None
            nxt = k + prefetch
            if nxt < len(meta):
                futures[nxt] = ex.submit(pd.read_parquet, meta[nxt][0], columns=columns)
            yield df, a, b
            del df


def stage_feat(cfg: dict) -> None:
    t0 = time.time()
    for split in ("train", "test"):
        ts = time.time()
        u = build_universe(cfg, split)
        ps = PairStore(cfg, split)
        I = np.array(ps.load("i", mmap=True))
        cols = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ps.dir, "*.npy")))
        cols = [c for c in cols if not c.startswith(("p1", "p2", "pruned"))]
        u.load(STR_FEATURE_COLS)
        out_dir = os.path.join(cfg["work_dir"], split, "feats")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        n_jobs = cfg["n_jobs"]
        want = int(cfg["features"]["chunk_pairs"])
        bounds = _chunk_bounds(I, fit_feat_chunk(want, min(n_jobs, max(1, len(I) // max(1, want)))))
        tasks = [(k, bounds[k], bounds[k + 1], os.path.join(out_dir, f"part_{k:05d}.parquet"))
                 for k in range(len(bounds) - 1)]
        workers = 1 if (n_jobs > 1 and len(tasks) > 1) else max(1, n_jobs)
        log(f"feat {split}: {len(I):,} pairs in {len(tasks)} parts on {min(n_jobs, len(tasks))} workers")
        done = 0
        g = {"u": u, "pdir": ps.dir, "cols": cols, "workers": workers}
        _G.clear()
        _G.update(g)
        ctx = _pool_ctx()
        try:
            if n_jobs > 1 and len(tasks) > 1 and ctx is not None:
                with ctx.Pool(min(n_jobs, len(tasks))) as pool:
                    for k, m in pool.imap_unordered(_feat_chunk, tasks):
                        done += m
                        el = time.time() - ts
                        if k % 10 == 0:
                            log(f"  {split}: {done:,}/{len(I):,} pairs ({done / el:,.0f}/s, "
                                f"eta {(len(I) - done) / max(1, done / el) / 60:.1f} min)")
            else:
                for t in tasks:
                    _feat_chunk(t)
        finally:
            _G.clear()
        with open(os.path.join(out_dir, "parts.json"), "w") as f:
            json.dump({"parts": [(t[3], t[1], t[2]) for t in tasks], "n": int(len(I))}, f)
        log(f"feat {split}: done in {time.time() - ts:.0f}s")
    log_exp(cfg, "feat", {}, t0)

# =============================================================================================================
# LIGHTGBM. Models are trained on a SAMPLE of S1 entities (all their candidate pairs):
#   * 4-fold OOF predictions for the sample (early stopping on an INNER holdout of training S1 groups);
#   * one full-sample model that scores every other train pair and every test pair.
# Pairs of non-sampled train S1 entities are never trained on, so their scores are honest too (like test).
# =============================================================================================================
def lgb_params(cl: dict, seed: int, n_jobs: int) -> dict:
    return {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": seed,
            "learning_rate": cl["learning_rate"], "num_leaves": cl["num_leaves"],
            "min_data_in_leaf": cl["min_data_in_leaf"], "feature_fraction": cl["feature_fraction"],
            "bagging_fraction": cl["bagging_fraction"], "bagging_freq": cl["bagging_freq"],
            "lambda_l2": cl["lambda_l2"], "num_threads": n_jobs, "max_bin": cl.get("max_bin", 255),
            "device_type": cl.get("device", "cpu")}


def _hash01(x: np.ndarray) -> np.ndarray:
    return ((x.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32


def _fit(X, y, tr, es, cl, seed, n_jobs):
    import lightgbm as lgb
    use_es = es.sum() > 0 and y[es].min() != y[es].max() and y[tr].min() != y[tr].max()
    if not use_es:  # tiny / degenerate data: no inner split, fixed number of rounds
        tr = tr | es
        return lgb.train(lgb_params(cl, seed, n_jobs), lgb.Dataset(X[tr], y[tr]),
                         num_boost_round=min(cl["num_boost_round"], 300))
    dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
    des = lgb.Dataset(X[es], y[es], reference=dtr)
    return lgb.train(lgb_params(cl, seed, n_jobs), dtr, num_boost_round=cl["num_boost_round"], valid_sets=[des],
                     callbacks=[lgb.early_stopping(cl["early_stopping"], verbose=False)])


def train_cv(df: pd.DataFrame, feats: list[str], cl: dict, n_jobs: int, seeds=None):
    from sklearn.metrics import roc_auc_score
    y = df["label"].values.astype(np.float32)
    folds = df["fold"].values
    es_mask = _hash01(df["i"].values) < cl.get("es_frac", 0.1)
    oof = np.zeros(len(df), np.float64)
    imp = pd.Series(0.0, index=feats)
    seeds = seeds or cl.get("seeds", [42])
    X = df[feats]
    for f in sorted(np.unique(folds)):
        va = folds == f
        if y[~va].min() == y[~va].max():
            log(f"  fold {f}: constant training labels -> constant prediction")
            oof[va] = y[~va][0]
            continue
        pred = np.zeros(int(va.sum()))
        for sd in seeds:
            m = _fit(X, y, ~va & ~es_mask, ~va & es_mask, cl, sd, n_jobs)
            pred += m.predict(X[va], num_iteration=m.best_iteration or None) / len(seeds)
            imp += pd.Series(m.feature_importance("gain"), index=feats)
        oof[va] = pred
        auc = roc_auc_score(y[va], pred) if 0 < y[va].sum() < va.sum() else float("nan")
        log(f"  fold {f}: n={int(va.sum()):,} pos={int(y[va].sum()):,} auc={auc:.5f} iters={m.best_iteration or m.current_iteration()}")
    return oof, imp.sort_values(ascending=False)


def train_full(df: pd.DataFrame, feats: list[str], cl: dict, n_jobs: int, seeds=None) -> list:
    y = df["label"].values.astype(np.float32)
    es_mask = _hash01(df["i"].values) < cl.get("es_frac", 0.1)
    X = df[feats]
    if y.min() == y.max():
        raise RuntimeError("training labels are constant -- check the ground truth / blocking recall")
    models = [_fit(X, y, ~es_mask, es_mask, cl, sd, n_jobs) for sd in (seeds or cl.get("seeds", [42]))]
    log(f"  full-sample model: iters={[m.best_iteration or m.current_iteration() for m in models]}")
    return models


def lgb_predict(models, X: pd.DataFrame, chunk: int = 2000000) -> np.ndarray:
    if len(X) == 0:
        return np.zeros(0)
    out = np.zeros(len(X))
    for st in range(0, len(X), chunk):
        xs = X.iloc[st:st + chunk]
        out[st:st + chunk] = np.mean([m.predict(xs, num_iteration=m.best_iteration or None) for m in models], axis=0)
    return out


def feature_list(df: pd.DataFrame, drop: list[str] | None = None) -> list[str]:
    bad = {"r", "i", "j", "label", "fold", "found", "p1", "p2", "i_id", "j_id"} | set(drop or [])
    return [c for c in df.columns if c not in bad and df[c].dtype.kind in "fiub"]


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score
    ok = 0 < y.sum() < len(y)
    return {"auc": round(float(roc_auc_score(y, p)), 6) if ok else None,
            "logloss": round(float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])), 6) if len(y) else None}


# =============================================================================================================
# L2 CONTEXT FEATURES: competition, exclusivity and neighbour support from stage-1 scores (p1 logit, CE logits).
# =============================================================================================================
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def within_group_pairs(keys: np.ndarray, max_group: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """All ordered row pairs (a, b), a != b, with keys[a] == keys[b]; vectorized per group size."""
    if len(keys) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sizes = np.diff(np.r_[starts, len(k)])
    A, B = [], []
    for sz in np.unique(sizes):
        if sz < 2 or sz > max_group:
            continue
        st = starts[sizes == sz]
        aa, bb = np.meshgrid(np.arange(sz), np.arange(sz), indexing="ij")
        m = aa != bb
        A.append((st[:, None] + aa[m][None, :]).ravel())
        B.append((st[:, None] + bb[m][None, :]).ravel())
    if not A:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return order[np.concatenate(A)], order[np.concatenate(B)]


def _rec_sim(u: Universe, ra: np.ndarray, rb: np.ndarray, n_jobs: int, chunk: int = 2000000) -> np.ndarray:
    """Cheap record-record similarity in [0,1]: mean of name and address token-set ratios.

    At full scale this runs over a few hundred million within-group pairs, so it goes through the parallel
    chunked path rather than decoding every string on the main process.
    """
    from rapidfuzz import fuzz
    if len(ra) == 0:
        return np.zeros(0, np.float32)
    d = _cpdist_chunked(u, ra, rb, ("n_core", "a_norm"), fuzz.token_set_ratio, n_jobs, chunk)
    return ((d["n_core"] + d["a_norm"]) / 200.0).astype(np.float32)


def _group_max(keys: np.ndarray, vals: np.ndarray, n: int, fill: float) -> np.ndarray:
    out = np.full(n, fill, np.float64)
    if len(keys):
        o = np.argsort(keys, kind="stable")
        k, v = keys[o], vals[o]
        st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        out[k[st]] = np.maximum(np.maximum.reduceat(v, st), fill)
    return out


def context_features(P: pd.DataFrame, u: Universe, score_cols: list[str], n_jobs: int) -> pd.DataFrame:
    new = {}
    I, J = P["i"].values.astype(np.int64), P["j"].values.astype(np.int64)
    n = len(P)
    inv = {"i": np.unique(I, return_inverse=True)[1], "j": np.unique(J, return_inverse=True)[1]}
    for c in score_cols:
        z = np.clip(P[c].values.astype(np.float64), -30, 30)
        z = np.where(np.isfinite(z), z, 0.0)
        p = sigmoid(z)
        for side, g in (("i", I), ("j", J)):
            rk, gap, m2, _ = group_context(g, z, fill=-30.0)
            new[f"cx_rk_{side}_{c}"] = rk
            new[f"cx_gap_{side}_{c}"] = gap
            new[f"cx_m2_{side}_{c}"] = m2
            new[f"cx_other_{side}_{c}"] = np.where(gap >= -1e-9, z - m2, z - gap).astype(np.float32)  # best competitor
            e = np.exp(z)
            new[f"cx_sm_{side}_{c}"] = (e / (1.0 + np.bincount(inv[side], weights=e)[inv[side]])).astype(np.float32)
        new[f"cx_sump_i_{c}"] = np.bincount(inv["i"], weights=p)[inv["i"]].astype(np.float32)
        new[f"cx_n05_i_{c}"] = np.bincount(inv["i"], weights=(p > 0.5).astype(np.float64))[inv["i"]].astype(np.float32)
    # neighbour support: other candidates of the same S1 that look like the SAME business as this record
    # (e.g. its S2 and S3 copies) and are scored high reinforce this pair.
    main = score_cols[-1]
    pz = sigmoid(np.clip(P[main].values.astype(np.float64), -30, 30))
    a, b = within_group_pairs(I)
    sim = _rec_sim(u, J[a], J[b], n_jobs)
    cross = u.src[J[a]] != u.src[J[b]]
    new["nb_max_sim"] = _group_max(a, sim, n, 0.0).astype(np.float32)
    new["nb_support"] = _group_max(a, sim * pz[b], n, 0.0).astype(np.float32)
    new["nb_support_x"] = _group_max(a[cross], (sim * pz[b])[cross], n, 0.0).astype(np.float32)
    new["nb_support_sum"] = np.bincount(a, weights=sim * pz[b], minlength=n).astype(np.float32)
    # competitor S1 similarity: is the record's best other S1 a near-duplicate of this S1 (chain branches)?
    a2, b2 = within_group_pairs(J)
    sim2 = _rec_sim(u, I[a2], I[b2], n_jobs)
    new["cmp_max_sim"] = _group_max(a2, sim2, n, 0.0).astype(np.float32)
    best = np.zeros(n, np.float32)
    if len(a2):
        o = np.lexsort((-pz[b2], a2))
        first = np.r_[True, a2[o][1:] != a2[o][:-1]]
        best[a2[o][first]] = sim2[o][first]
    new["cmp_best_sim"] = best
    return pd.DataFrame(new, index=P.index)


# =============================================================================================================
# DECISION LAYER: pair probabilities -> per-S1 match sets, optimizing macro per-entity F0.5 directly.
#  * S1 is deduplicated => each record belongs to <= 1 S1. Under independent calibrated scores the posterior over a
#    record's candidate S1s (plus 'none') is P(i|j) = o_ij / (1 + sum_i' o_i'j), o = p/(1-p)  ("softmax" mode).
#  * Per-entity F0.5 with singletons implies non-uniform optimal thresholds. Policies: threshold | two_threshold |
#    expected_f (exact expected-F0.5 over top-k sets, Poisson-binomial, vectorized) with a tunable calibration.
#  Tuned on held-out scores; the honest estimate is CROSS-FITTED (tune on other folds, score the held-out fold),
#  computed in ONE pass over the grid because per-entity scores are additive over folds.
# =============================================================================================================
def exclusive(j: np.ndarray, p: np.ndarray, mode: str) -> np.ndarray:
    p = np.asarray(p, np.float64)
    if mode == "none":
        return p
    out = p.copy()
    inv = np.unique(j, return_inverse=True)[1]
    if mode in ("softmax", "softmax_hard"):
        o = np.exp(logit(p))
        out = o / (1.0 + np.bincount(inv, weights=o)[inv])
    if mode in ("hard", "softmax_hard"):
        rk = group_context(inv, out)[0]
        out = np.where(rk == 1, out, 0.0)
    return out


class Evaluator:
    """Fast macro-F0.5 over many candidate selections. n_true counts ALL ground-truth matches (incl. the ones
    blocking/pruning missed), so recall losses upstream are charged correctly."""

    def __init__(self, i_rows: np.ndarray, label: np.ndarray, s1_rows: np.ndarray, n_true: np.ndarray, n_recs: int):
        pos = np.full(n_recs, -1, np.int64)
        pos[s1_rows] = np.arange(len(s1_rows))
        self.gi = pos[i_rows]
        if (self.gi < 0).any():
            raise ValueError("pair rows reference S1 rows outside the evaluation set")
        self.label = label.astype(np.float64)
        self.n_true = n_true.astype(np.float64)
        self.n = len(s1_rows)

    def per_entity(self, sel: np.ndarray) -> np.ndarray:
        k = np.bincount(self.gi, weights=sel.astype(np.float64), minlength=self.n)
        tp = np.bincount(self.gi, weights=sel * self.label, minlength=self.n)
        return np.where(k == 0, (self.n_true == 0).astype(np.float64),
                        np.where((self.n_true > 0) & (tp > 0), 1.25 * tp / np.maximum(0.25 * self.n_true + k, EPS), 0.0))

    def score(self, sel: np.ndarray) -> float:
        return float(self.per_entity(sel).mean()) if self.n else float("nan")


def group_layout(i: np.ndarray, p: np.ndarray, K: int):
    """Sort rows by (group, -p). Returns order, group id and rank of each sorted row, padded matrix Q (G x K)."""
    order = np.lexsort((-p, i))
    i_s = i[order]
    first = np.r_[True, i_s[1:] != i_s[:-1]] if len(i_s) else np.zeros(0, bool)
    gid = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    rank = np.arange(len(i_s)) - starts[gid] if len(i_s) else np.zeros(0, np.int64)
    G = len(starts)
    keep = rank < K
    Q = np.zeros((G, K))
    Q[gid[keep], rank[keep]] = p[order][keep]
    n_real = np.minimum(np.bincount(gid, minlength=G), K) if G else np.zeros(0, np.int64)
    return order, gid, rank, Q, n_real


def expected_f_table(Q: np.ndarray, beta2: float = 0.25) -> np.ndarray:
    """E[F_beta] of predicting the top-k set, k = 0..K, for every group (independent Bernoulli q's)."""
    G, K = Q.shape
    pre = [np.ones((G, 1))]
    for k in range(K):
        prev, q = pre[-1], Q[:, k:k + 1]
        nxt = np.zeros((G, k + 2))
        nxt[:, :k + 1] += prev * (1 - q)
        nxt[:, 1:] += prev * q
        pre.append(nxt)
    suf = [None] * (K + 1)
    suf[K] = np.ones((G, 1))
    for k in range(K - 1, -1, -1):
        prev, q = suf[k + 1], Q[:, k:k + 1]
        nxt = np.zeros((G, prev.shape[1] + 1))
        nxt[:, :-1] += prev * (1 - q)
        nxt[:, 1:] += prev * q
        suf[k] = nxt
    E = np.zeros((G, K + 1))
    E[:, 0] = suf[0][:, 0]
    for k in range(1, K + 1):
        a = np.arange(k + 1)[:, None]
        b = np.arange(K - k + 1)[None, :]
        Fm = np.where(a > 0, (1 + beta2) * a / (beta2 * (a + b) + k), 0.0)
        E[:, k] = np.einsum("ga,ab,gb->g", pre[k], Fm, suf[k])
    return E


def select_expected_f(layout, a: float = 1.0, b: float = 0.0, chunk: int = 100000) -> np.ndarray:
    order, gid, rank, Q, n_real = layout
    G, K = Q.shape
    best_k = np.zeros(G, np.int64)
    for st in range(0, G, chunk):
        q, nr = Q[st:st + chunk], n_real[st:st + chunk]
        real = np.arange(K)[None, :] < nr[:, None]
        Qc = np.where(real, sigmoid(a * logit(np.clip(q, EPS, 1 - EPS)) + b), 0.0)
        E = expected_f_table(Qc)
        E[np.arange(K + 1)[None, :] > nr[:, None]] = -1.0
        best_k[st:st + chunk] = E.argmax(1)
    sel = np.zeros(len(order))
    sel[order] = (rank < best_k[gid]).astype(np.float64)
    return sel


GRIDS = {
    "threshold": [{"t": float(t)} for t in np.round(np.arange(0.20, 0.96, 0.025), 3)],
    "two_threshold": [{"t1": float(t1), "t2": float(t2)} for t1 in np.round(np.arange(0.25, 0.81, 0.05), 3)
                      for t2 in np.round(np.arange(0.40, 0.96, 0.05), 3)],
    "expected_f": [{"a": a, "b": b} for a in (0.7, 0.85, 1.0, 1.2, 1.5) for b in (-1.0, -0.5, 0.0, 0.5, 1.0)],
}


class PolicyCache:
    """Per exclusivity mode: transformed p (computed on the FULL pair set: a record competes with S1s from every
    fold), then restricted to the evaluation rows; rank-in-S1, group max and expected-F layout computed once."""

    def __init__(self, i, j, p, mode, K, rows=None):
        pe = exclusive(j, p, mode)
        if rows is not None:
            i, pe = i[rows], pe[rows]
        self.p = pe
        rk, gap, _, _ = group_context(i, pe)
        self.rk, self.gmax = rk, pe - gap
        self.layout = group_layout(i, pe, K)

    def select(self, policy: str, prm: dict) -> np.ndarray:
        p = self.p
        if policy == "threshold":
            return (p >= prm["t"]).astype(np.float64)
        if policy == "two_threshold":
            top_ok = self.gmax >= prm["t1"]
            return (((self.rk == 1) & (p >= prm["t1"])) | ((self.rk > 1) & (p >= prm["t2"]) & top_ok)).astype(np.float64)
        if policy == "expected_f":
            return select_expected_f(self.layout, prm["a"], prm["b"])
        raise ValueError(policy)


def tune_all(caches: dict, ev: Evaluator, s1_fold: np.ndarray, policies) -> dict:
    """Score every (mode, policy, params) once; derive the best overall AND the cross-fitted estimate."""
    fv, fidx = np.unique(s1_fold, return_inverse=True)
    nf = np.bincount(fidx, minlength=len(fv)).astype(np.float64)
    cands = []
    for mode, pc_ in caches.items():
        for pol in policies:
            for prm in GRIDS[pol]:
                pe = ev.per_entity(pc_.select(pol, prm))
                cands.append({"excl": mode, "policy": pol, "params": prm, "sum": float(pe.sum()),
                              "fold_sum": np.bincount(fidx, weights=pe, minlength=len(fv))})
    tot = np.array([c["sum"] for c in cands])
    FS = np.vstack([c["fold_sum"] for c in cands])
    best = cands[int(tot.argmax())]
    per, cf = [], 0.0
    for k, f in enumerate(fv):
        if len(fv) > 1:
            b = int(((tot - FS[:, k]) / max(1.0, ev.n - nf[k])).argmax())
        else:
            b = int(tot.argmax())
        sc = FS[b, k] / max(1.0, nf[k])
        cf += FS[b, k]
        per.append({"fold": int(f), "score": round(float(sc), 6), "excl": cands[b]["excl"],
                    "policy": cands[b]["policy"], "params": cands[b]["params"], "n_s1": int(nf[k])})
    ranked = sorted(cands, key=lambda c: -c["sum"])
    top = [{"excl": c["excl"], "policy": c["policy"], "params": c["params"], "score": round(c["sum"] / ev.n, 6)}
           for c in ranked[:10]]
    return {"best": {"excl": best["excl"], "policy": best["policy"], "params": best["params"],
                     "score": best["sum"] / ev.n}, "cross_fitted": cf / max(1.0, ev.n), "per_fold": per, "top": top}


# =============================================================================================================
# CPU STAGES
# =============================================================================================================
def stage_prep(cfg: dict) -> None:
    t0 = time.time()
    for split in ("train", "test"):
        build_universe(cfg, split, force=True)
    get_folds(cfg, build_universe(cfg, "train"))
    log_exp(cfg, "prep", {}, t0)


def stage_l1(cfg: dict) -> None:
    t0 = time.time()
    main = cfg["run"] == "main"
    tr = build_universe(cfg, "train")
    folds, samp = get_folds(cfg, tr), sample_mask(cfg, tr)
    frames = []
    for df, _, _ in read_parts(cfg, "train"):
        m = samp[df["i"].values]
        if m.any():
            frames.append(df[m])
    F = pd.concat(frames, ignore_index=True)
    del frames
    F["fold"] = folds[F["i"].values]
    feats = feature_list(F, cfg["features"]["drop"])
    log(f"l1: training on {F['i'].nunique():,} sampled S1 ({len(F):,} pairs, {int(F['label'].sum()):,} positives), "
        f"{len(feats)} features")
    oof, imp = train_cv(F, feats, cfg["lgb"], cfg["n_jobs"])
    ps = PairStore(cfg, "train")
    p1 = np.full(ps.n(), np.nan, np.float32)
    p1[F["r"].values] = oof
    met = binary_metrics(F["label"].values, oof)
    met.update(n_train_pairs=int(len(F)), n_train_s1=int(F["i"].nunique()), n_features=len(feats))
    full = train_full(F, feats, cfg["lgb"], cfg["n_jobs"]) if main else None
    del F
    if main:
        for df, _, _ in read_parts(cfg, "train"):
            m = ~samp[df["i"].values]
            if m.any():
                p1[df["r"].values[m]] = lgb_predict(full, df.loc[m, feats])
    ps.save(f"p1{run_tag(cfg)}", p1)
    if main:
        pt = PairStore(cfg, "test")
        p1t = np.zeros(pt.n(), np.float32)
        for df, _, _ in read_parts(cfg, "test"):
            p1t[df["r"].values] = lgb_predict(full, df[feats])
        pt.save("p1", p1t)
    imp.to_csv(wpath(cfg, "train", "l1_importance", tagged=True, ext="csv"))
    with open(wpath(cfg, "train", "l1_meta", tagged=True, ext="json"), "w") as f:
        json.dump({"features": feats, **met}, f)
    log(f"l1 {cfg['run']}: sample OOF {met}; top features: {list(imp.index[:10])}")
    log_exp(cfg, "l1", met, t0)


def prune_keep(I, J, p, cl: dict) -> np.ndarray:
    ok = np.isfinite(p)
    pz = np.where(ok, p, -1.0)
    rk_i, rk_j = group_context(I, pz)[0], group_context(J, pz)[0]
    return ok & (pz >= cl["prune_min_p"]) & (rk_i <= cl["prune_rank_s1"]) & (rk_j <= cl["prune_rank_rec"])


def oracle_f05(n_true: np.ndarray, tp: np.ndarray) -> float:
    """Best achievable macro-F0.5 given the candidate set (predict exactly the true candidates)."""
    with np.errstate(invalid="ignore"):  # 0/0 in the n_true==0 branch is computed but discarded by np.where
        f = np.where(n_true == 0, 1.0, np.where(tp > 0, 1.25 * tp / (0.25 * n_true + tp), 0.0))
    return float(f.mean()) if len(f) else float("nan")


def stage_prune(cfg: dict) -> None:
    t0 = time.time()
    tag, main = run_tag(cfg), cfg["run"] == "main"
    for split in (("train", "test") if main else ("train",)):
        ps = PairStore(cfg, split)
        I, J, p1 = ps.load("i"), ps.load("j"), ps.load(f"p1{tag}")
        keep = prune_keep(I, J, p1, cfg["l1"])
        rows = np.flatnonzero(keep).astype(np.int64)
        ps.save(f"pruned{tag}", rows)
        if split == "train":
            u = build_universe(cfg, "train")
            L = ps.load("label")
            scope = sample_mask(cfg, u) if not main else np.isin(np.arange(u.n), u.s1_rows)
            s1 = u.s1_rows[scope[u.s1_rows]]
            nt = u.n_true_all[s1]
            in_scope = scope[I]
            pos = np.full(u.n, -1, np.int64)
            pos[s1] = np.arange(len(s1))
            tp = np.bincount(pos[I[keep & in_scope]], weights=L[keep & in_scope].astype(np.float64), minlength=len(s1))
            rep = {"n_s1": int(len(s1)), "recall_blocking": round(float(L[in_scope].sum() / max(1, nt.sum())), 5),
                   "recall_pruned": round(float(tp.sum() / max(1, nt.sum())), 5),
                   "pairs_per_s1": round(float((keep & in_scope).sum() / max(1, len(s1))), 3),
                   "oracle_f05": round(oracle_f05(nt, tp), 5)}
            log(f"prune train ({cfg['run']}): {rep}")
            log_exp(cfg, "prune", rep, t0)
        else:
            log(f"prune test: {len(rows):,} pairs kept ({len(rows) / max(1, len(np.unique(I))):.2f} per S1 with candidates)")


def ce_dir(cfg: dict, name: str) -> str:
    d = os.path.join(cfg["work_dir"], "ce", name)
    os.makedirs(d, exist_ok=True)
    return d


def ce_names_available(cfg: dict) -> list[str]:
    if cfg["run"] != "main":
        return []
    names = cfg["l2"]["ce_names"]
    if names in ("auto", None):
        names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(cfg["work_dir"], "ce", "*", "train.parquet")))
    elif isinstance(names, str):
        names = [x for x in names.split(",") if x]
    return [n for n in names if os.path.exists(os.path.join(cfg["work_dir"], "ce", n, "test.parquet"))]


def build_l2_context(cfg: dict, split: str, u: Universe, ce_names: list[str]):
    """Ids, stage-1 scores and context features over the FULL pruned set -- but NOT the ~130 pair features.

    Materializing every pruned row with its full feature set was this pipeline's memory ceiling: 24M rows x
    ~190 columns is ~17 GB for train alone, before the LightGBM Dataset copy. Context features only ever read
    i, j and the score columns, so they are computed on a narrow frame; the wide features are attached later,
    for the sampled rows only (training) or one feature part at a time (prediction).
    """
    tag = run_tag(cfg)
    ps = PairStore(cfg, split)
    rows = ps.load(f"pruned{tag}")
    S = pd.DataFrame({"r": rows,
                      "i": ps.load("i")[rows].astype(np.int64),
                      "j": ps.load("j")[rows].astype(np.int64)})
    S["z_p1"] = logit(ps.load(f"p1{tag}")[rows]).astype(np.float32)
    score_cols = ["z_p1"]
    Ia, Ja = ps.load("i", mmap=True), ps.load("j", mmap=True)
    for nme in ce_names:
        Z = pd.read_parquet(os.path.join(ce_dir(cfg, nme), f"{split}.parquet"))
        r = Z["r"].values.astype(np.int64)
        if len(r) and (r.max() >= len(Ia) or not ((np.asarray(Ia[r]) == Z["i"].values).all()
                                                  and (np.asarray(Ja[r]) == Z["j"].values).all())):
            raise RuntimeError(f"CE '{nme}' outputs belong to an older candidate set -- re-run `ce --name {nme}`")
        pos = np.searchsorted(rows, r)
        ok = pos < len(rows)
        ok[ok] = rows[pos[ok]] == r[ok]
        z = np.full(len(S), np.nan, np.float32)
        z[pos[ok]] = Z["z"].values[ok]
        has = np.isfinite(z)
        S[f"ce_has_{nme}"] = has.astype(np.int8)
        S[f"z_{nme}"] = np.where(has, z, S["z_p1"].values).astype(np.float32)
        score_cols.append(f"z_{nme}")
        log(f"  {split}: CE '{nme}' covers {has.mean():.1%} of pruned pairs")
        del Z
    if len(ce_names) > 1:
        S["z_ce_mean"] = S[[f"z_{n}" for n in ce_names]].mean(1).astype(np.float32)
        score_cols.append("z_ce_mean")
    C = context_features(S, u, score_cols, cfg["n_jobs"])
    log(f"  {split}: l2 context built for {len(S):,} pruned pairs "
        f"({(S.memory_usage(deep=False).sum() + C.memory_usage(deep=False).sum()) / 2 ** 30:.1f} GB)")
    return S, C, score_cols, rows


def iter_l2_parts(cfg: dict, split: str, rows: np.ndarray, S: pd.DataFrame, C: pd.DataFrame,
                  mask: np.ndarray | None = None):
    """Yield (positions into `rows`, full L2 frame) one feature part at a time.

    Keeps peak memory at one part rather than the whole split, which is what lets `l2` run inside a modest
    SLURM allocation.
    """
    extra = [c for c in S.columns if c not in ("r", "i", "j")]
    seen = 0
    for df, a, b in read_parts(cfg, split):
        lo, hi = int(np.searchsorted(rows, a)), int(np.searchsorted(rows, b))
        if hi <= lo:
            continue
        sel = np.arange(lo, hi)
        if mask is not None:
            sel = sel[mask[sel]]
            if len(sel) == 0:
                continue
        X = df.iloc[rows[sel] - a].reset_index(drop=True)
        if not (X["r"].values == rows[sel]).all():
            raise RuntimeError(f"{split}: feature parts do not match the pair store -- re-run `feat`")
        seen += len(sel)
        yield sel, pd.concat([X, S.iloc[sel][extra].reset_index(drop=True),
                              C.iloc[sel].reset_index(drop=True)], axis=1)
        del X
    if mask is None and seen != len(rows):
        raise RuntimeError(f"{split}: feature parts cover {seen:,} of {len(rows):,} pruned rows -- re-run `feat`")


def stage_l2(cfg: dict) -> None:
    t0 = time.time()
    tag, main = run_tag(cfg), cfg["run"] == "main"
    tr = build_universe(cfg, "train")
    ce_names = ce_names_available(cfg)
    S, C, score_cols, rows = build_l2_context(cfg, "train", tr, ce_names)
    folds, samp = get_folds(cfg, tr), sample_mask(cfg, tr)
    ms = samp[S["i"].values]
    # Full features for the sampled rows only -- that is all the model ever trains on.
    Xs = pd.concat([f for _, f in iter_l2_parts(cfg, "train", rows, S, C, mask=ms)], ignore_index=True)
    Xs["fold"] = folds[Xs["i"].values]
    feats = feature_list(Xs, cfg["features"]["drop"])
    log(f"l2 {cfg['run']}: {len(Xs):,} sample pairs of {len(rows):,} pruned, {len(feats)} features, "
        f"scores={score_cols}")
    oof, imp = train_cv(Xs, feats, cfg["lgb"], cfg["n_jobs"], seeds=cfg["l2"]["seeds"])
    p2 = np.full(len(rows), np.nan, np.float32)
    p2[ms] = oof
    y = Xs["label"].values
    met = {"l2": binary_metrics(y, oof), "inputs": {c: binary_metrics(y, sigmoid(Xs[c].values)) for c in score_cols},
           "ce_names": ce_names, "n_features": len(feats)}
    full = None
    if main:
        full = train_full(Xs, feats, cfg["lgb"], cfg["n_jobs"], seeds=cfg["l2"]["seeds"])
    del Xs
    if main and (~ms).any():                       # score the rest of train one part at a time
        for sel, Xc in iter_l2_parts(cfg, "train", rows, S, C, mask=~ms):
            p2[sel] = lgb_predict(full, Xc[feats])
            del Xc
    PairStore(cfg, "train").save(f"p2{tag}", p2)
    imp.to_csv(wpath(cfg, "train", "l2_importance", tagged=True, ext="csv"))
    with open(wpath(cfg, "train", "l2_meta", tagged=True, ext="json"), "w") as f:
        json.dump({"features": feats, **met}, f, default=str)
    log(f"l2 {cfg['run']}: {json.dumps(met, default=str)}; top: {list(imp.index[:12])}")
    del S, C, p2
    if main:
        te = build_universe(cfg, "test")
        St, Ct, _, rows_t = build_l2_context(cfg, "test", te, ce_names)
        p2t = np.zeros(len(rows_t), np.float32)
        for sel, Xc in iter_l2_parts(cfg, "test", rows_t, St, Ct):
            p2t[sel] = lgb_predict(full, Xc[feats])
            del Xc
        PairStore(cfg, "test").save("p2", p2t)
    log_exp(cfg, "l2", met, t0)


def _decision_scores(cfg: dict, split: str, ps: PairStore, rows: np.ndarray) -> tuple[np.ndarray, str]:
    tag = run_tag(cfg) if split == "train" else ""
    if ps.has(f"p2{tag}"):
        p = ps.load(f"p2{tag}").astype(np.float64)
        if len(p) == len(rows) and os.path.getmtime(ps.path(f"p2{tag}")) >= os.path.getmtime(ps.path(f"pruned{tag}")):
            return p, "p2"
        log(f"{split}: p2 is stale (prune re-run after l2) -- falling back to p1; re-run l2")
    return ps.load(f"p1{tag}")[rows].astype(np.float64), "p1"


def stage_decide(cfg: dict) -> dict:
    t0 = time.time()
    tag, main, dc = run_tag(cfg), cfg["run"] == "main", cfg["decision"]
    K = int(dc["max_cands"])
    tr = build_universe(cfg, "train")
    ps = PairStore(cfg, "train")
    rows = ps.load(f"pruned{tag}")
    I, J = ps.load("i")[rows].astype(np.int64), ps.load("j")[rows].astype(np.int64)
    L = ps.load("label")[rows]
    p, pcol = _decision_scores(cfg, "train", ps, rows)
    p = np.where(np.isfinite(p), p, 0.0)
    ce_used = False
    if pcol == "p2" and os.path.exists(wpath(cfg, "train", "l2_meta", tagged=True, ext="json")):
        with open(wpath(cfg, "train", "l2_meta", tagged=True, ext="json")) as f:
            ce_used = bool(json.load(f).get("ce_names"))
    samp = sample_mask(cfg, tr)
    if not main or ce_used:  # CE logits exist only for the sample -> evaluate there
        eval_s1 = tr.s1_rows[samp[tr.s1_rows]]
    else:
        eval_s1 = tr.s1_rows
        if len(eval_s1) > dc["max_eval_s1"]:
            eval_s1 = np.sort(np.random.RandomState(cfg["seed"] + 11).choice(eval_s1, int(dc["max_eval_s1"]), replace=False))
    in_eval = np.zeros(tr.n, bool)
    in_eval[eval_s1] = True
    sub = np.flatnonzero(in_eval[I])
    caches = {m: PolicyCache(I, J, p, m, K, rows=sub) for m in dc["excl_modes"]}
    ev = Evaluator(I[sub], L[sub], eval_s1, tr.n_true_all[eval_s1], tr.n)
    res = tune_all(caches, ev, get_folds(cfg, tr)[eval_s1], dc["policies"])
    best = res["best"]
    sel = caches[best["excl"]].select(best["policy"], best["params"])
    pe = ev.per_entity(sel)
    n_pred = np.bincount(ev.gi, weights=sel, minlength=ev.n)
    bd = breakdown_arrays(pe, ev.n_true, n_pred)
    by_cty = {}
    for k, name in enumerate(tr.cty_names):
        m = tr.cty_code[eval_s1] == k
        if m.any():
            by_cty[name] = breakdown_arrays(pe[m], ev.n_true[m], n_pred[m])["all"]
    rep = {"run": cfg["run"], "scores": pcol, "n_eval_s1": int(ev.n), "best": best,
           "cross_fitted_f05": round(res["cross_fitted"], 6), "in_sample_best_f05": round(best["score"], 6),
           "per_fold": res["per_fold"], "breakdown": bd, "by_country": by_cty, "top_configs": res["top"]}
    with open(wpath(cfg, "train", "decision", tagged=True, ext="json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    log(f"decide {cfg['run']} [{pcol}]: cross-fitted F0.5={rep['cross_fitted_f05']:.5f} "
        f"(in-sample {best['score']:.5f}) best={best['excl']}/{best['policy']} {best['params']}")
    log(f"  breakdown: {bd}; by country: {by_cty}")
    if main:
        rep["submission_problems"] = write_submission(cfg, best, K)
    log_exp(cfg, "decide", {"cross_fitted_f05": rep["cross_fitted_f05"], "in_sample": round(best["score"], 6),
                            "scores": pcol, "best": best}, t0)
    return rep


def write_submission(cfg: dict, best: dict, K: int) -> list[str]:
    te = build_universe(cfg, "test")
    ps = PairStore(cfg, "test")
    rows = ps.load("pruned")
    I, J = ps.load("i")[rows].astype(np.int64), ps.load("j")[rows].astype(np.int64)
    p, pcol = _decision_scores(cfg, "test", ps, rows)
    sel = PolicyCache(I, J, p, best["excl"], K).select(best["policy"], best["params"])
    order = np.lexsort((-p, I))
    I, J, sel = I[order], J[order], sel[order] > 0
    ids = te.col("entity_id")
    s1 = te.s1_rows
    st, en = np.searchsorted(I, s1, "left").tolist(), np.searchsorted(I, s1, "right").tolist()
    rid, sl = ids.take(J), sel.tolist()
    cand = [",".join(rid[a:b]) for a, b in zip(st, en)]
    match = [",".join([x for x, s_ in zip(rid[a:b], sl[a:b]) if s_]) for a, b in zip(st, en)]
    s1_ids = ids.take(s1)
    od = cfg["output_dir"]
    mp_, cp_ = os.path.join(od, "matching_results.tsv"), os.path.join(od, "candidate_pairs.tsv")
    write_id_lists(mp_, s1_ids, match, "matched_entity_ids")
    write_id_lists(cp_, s1_ids, cand, "candidate_entity_ids")
    all_ids = ids.to_arrow()
    problems = check_submission(mp_, cp_, pc.filter(all_ids, pa.array(te.src == 1)),
                                pc.filter(all_ids, pa.array(te.src != 1)))
    n_m = np.array([len(x) > 0 for x in match])
    log(f"wrote {mp_} [{pcol}]: {len(s1):,} S1 rows, {int(sel.sum()):,} matches, {n_m.mean():.1%} non-empty, "
        f"{len(J):,} candidates; checks: {'PASS' if not problems else problems}")
    return problems


def stage_cpu(cfg: dict) -> None:
    for split in ("train", "test"):
        build_universe(cfg, split)
    stage_block(cfg)
    stage_feat(cfg)
    stage_l1(cfg)
    stage_prune(cfg)
    stage_l2(cfg)
    stage_decide(cfg)


def stage_loco(cfg: dict) -> dict:
    """Leave-one-country-out on the sample: train on one country, evaluate on the other (proxy for France)."""
    c2 = copy.deepcopy(cfg)
    c2["run"], c2["cv"]["scheme"] = "loco", "loco"
    stage_l1(c2)
    stage_prune(c2)
    stage_l2(c2)
    return stage_decide(c2)

# =============================================================================================================
# GPU COMMON  (torch / transformers / peft are imported lazily: CPU stages never need them)
# =============================================================================================================
CFG_ARGV: list[str] = []  # --config/--set arguments, forwarded to GPU worker subprocesses


def gpu_list(s) -> list[int]:
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def ce_device() -> str:
    """Device for the cross-encoder. BER_DEVICE=cpu lets you dry-run the CE stage before paying for a GPU."""
    import torch
    return os.environ.get("BER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _oom_error(torch):
    return getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


def _empty_cache(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_mem_gb(torch) -> float:
    try:
        return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def pick_dtype(torch, override: str = "auto"):
    if override == "fp32":
        return torch.float32
    if override == "fp16":
        return torch.float16
    if override == "bf16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    # bf16 only on compute capability >= 8 (Ampere+). Turing (RTX 20xx) -> fp16 (+ GradScaler when training)
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def hf_load(cls, name: str, dtype, quant: str | None = None, **extra):
    import transformers
    kw = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": dtype}
    kw.update(extra)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0}
    m = cls.from_pretrained(name, **kw)
    if quant != "4bit":
        m = m.to(device=ce_device(), dtype=dtype)
    return m


def run_workers(jobs: list[tuple[int, list[str]]]) -> None:
    """One subprocess per GPU (CUDA_VISIBLE_DEVICES pinned), each re-running this file with a worker command."""
    procs = []
    par = "true" if os.environ.get("BER_TOK_PARALLEL", "1") not in ("0", "false") else "false"
    # SLURM (and Docker, and k8s) already set CUDA_VISIBLE_DEVICES to the cards this job was granted, and the
    # ids in it are physical. Writing a bare "0" over that would point a worker at a GPU outside the
    # allocation, which either fails outright or -- worse -- lands on someone else's job. Index into the
    # existing list instead, so `--gpus 0,1` always means "the first two cards I was actually given".
    vis = [d.strip() for d in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if d.strip()]
    for gpu, argv in jobs:
        dev = vis[gpu] if gpu < len(vis) else str(gpu)
        if vis and gpu >= len(vis):
            log(f"WARNING: asked for gpu {gpu} but this job was allocated {len(vis)} ({','.join(vis)})")
        # Tokenizing tens of millions of pairs on one Python thread is a real share of CE wall time, and each
        # worker owns its GPU, so let the Rust tokenizer use threads (no forking happens after this point).
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, TOKENIZERS_PARALLELISM=par)
        cmd = [sys.executable, os.path.abspath(__file__)] + argv + CFG_ARGV
        log(f"launch gpu{gpu} (CUDA_VISIBLE_DEVICES={dev}): {' '.join(argv)}")
        procs.append((gpu, subprocess.Popen(cmd, env=env)))
    bad = [(g, rc) for g, rc in ((g, p.wait()) for g, p in procs) if rc != 0]
    if bad:
        raise RuntimeError(f"GPU workers failed (gpu, returncode): {bad}")


# =============================================================================================================
# CROSS-ENCODER: Qwen3-Reranker (yes/no head) + LoRA, trained on the SAMPLE (one CV fold per GPU).
# Each fold model scores its OOF fold of the sample AND one shard of the test pairs, so every pair (train OOF
# or test) is scored by exactly one model -- same score distribution, and the test costs 1x instead of 4x.
# Optional gate (ce.gate=[lo,hi] on p1): only uncertain pairs are scored; L2 fills the rest with p1.
# z = logit("yes") - logit("no") at the final position.
# =============================================================================================================
RR_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
             "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def ce_spec(cfg: dict, name: str) -> dict:
    c = cfg.get("ce", {})
    base = CE_PRESETS.get(c.get("preset") or name)
    if base is None:
        raise KeyError(f"no CE preset for '{name}'; pass --set ce.preset=ce06 (presets: {sorted(CE_PRESETS)})")
    spec = dict(base)
    spec.update({k: v for k, v in c.items() if k in base or k in ("dtype", "instruction")})
    return spec


def select_ce_train(df: pd.DataFrame, spec: dict, rng) -> pd.DataFrame:
    """All positives + the hardest negatives per S1 (by p1). Optional cap on #pairs by sampling whole S1 groups."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].sort_values("p1", ascending=False).groupby("i", sort=False).head(spec["neg_per_s1"])
    out = pd.concat([pos, neg], ignore_index=True)
    cap = int(spec.get("max_train_pairs") or 0)
    if cap and len(out) > cap:
        g = out["i"].unique()
        keep = set(rng.choice(g, size=max(1, int(len(g) * cap / len(out))), replace=False))
        out = out[out["i"].isin(keep)].reset_index(drop=True)
    return out


class Reranker:
    """Two architectures behind one interface (`encode` -> `logit`).

    encoder: AutoModelForSequenceClassification(num_labels=1) on a (text_a, text_b) pair -- one scalar logit,
             no prompt boilerplate, fully fine-tuned. This is the fast path.
    causal : the original Qwen3-Reranker yes/no head with LoRA adapters.
    """

    def __init__(self, spec: dict, train: bool):
        import torch
        from transformers import AutoTokenizer
        self.torch, self.spec = torch, spec
        self.arch = spec.get("arch", "causal")
        self.device = ce_device()
        self.dtype = pick_dtype(torch, spec.get("dtype", "auto"))
        self.max_len = int(spec["max_len"])
        self.pad_multiple = int(spec.get("pad_multiple") or 0) or None
        self._keep = True
        trc = {"trust_remote_code": True} if spec.get("trust_remote_code") else {}
        if self.arch == "encoder":
            from transformers import AutoModelForSequenceClassification
            self.tok = AutoTokenizer.from_pretrained(spec["model"], **trc)   # right padding: CLS stays at 0
            # num_labels=1 keeps a pretrained reranker's existing 1-logit head; on an MLM backbone it creates
            # a fresh one. ignore_mismatched_sizes lets a 2-class head be replaced without a load error.
            m = hf_load(AutoModelForSequenceClassification, spec["model"], self.dtype, spec.get("quant"),
                        num_labels=1, ignore_mismatched_sizes=True, **trc)
            if train:
                # The freshly initialized head must be fp32 or it will not train stably under autocast.
                m = m.float() if self.dtype == torch.float16 else m
                if spec.get("grad_ckpt"):
                    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
                log(f"  encoder CE: {spec['model']}, {n_tr / 1e6:.0f}M trainable params, max_len={self.max_len}")
        else:
            from transformers import AutoModelForCausalLM
            self.tok = AutoTokenizer.from_pretrained(spec["model"], padding_side="left", **trc)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            m = hf_load(AutoModelForCausalLM, spec["model"], self.dtype, spec.get("quant"), **trc)
            m.config.use_cache = False
            self.yes = self.tok.convert_tokens_to_ids("yes")
            self.no = self.tok.convert_tokens_to_ids("no")
            self.pre = self.tok.encode(RR_PREFIX, add_special_tokens=False)
            self.suf = self.tok.encode(RR_SUFFIX, add_special_tokens=False)
            self.max_body = self.max_len - len(self.pre) - len(self.suf)
            self.instr = spec.get("instruction") or CE_INSTRUCTION
            if train:
                from peft import LoraConfig, get_peft_model
                if spec.get("quant") == "4bit":
                    from peft import prepare_model_for_kbit_training
                    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True,
                                                        gradient_checkpointing_kwargs={"use_reentrant": False})
                elif spec.get("grad_ckpt", True):
                    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    m.enable_input_require_grads()
                m = get_peft_model(m, LoraConfig(r=spec["lora_r"], lora_alpha=spec["lora_alpha"],
                                                 lora_dropout=spec["lora_dropout"], task_type="CAUSAL_LM",
                                                 target_modules=LORA_TARGETS))
                for p in m.parameters():  # trainable params in fp32: required by GradScaler, better for AdamW
                    if p.requires_grad and p.dtype != torch.float32:
                        p.data = p.data.float()
                m.print_trainable_parameters()
        self.model = m
        if spec.get("compile"):
            try:
                self.model = torch.compile(self.model, dynamic=True)
                log("  torch.compile enabled (first batches will be slow while it warms up)")
            except Exception as e:  # compile is an optimization, never a requirement
                log(f"  torch.compile unavailable ({e}); continuing eager")

    def n_tokens(self, a: list[str], b: list[str]) -> np.ndarray:
        """Token length of each pair, for length bucketing. Cheap char-based proxy, no tokenizer call."""
        return np.fromiter(((len(x) + len(y)) for x, y in zip(a, b)), np.int32, count=len(a))

    def encode(self, q: list[str], d: list[str]) -> dict:
        if self.arch == "encoder":
            batch = self.tok(list(q), list(d), truncation="longest_first", max_length=self.max_len,
                             padding=True, pad_to_multiple_of=self.pad_multiple, return_tensors="pt")
        else:
            body = self.tok([f"<Instruct>: {self.instr}\n<Query>: {a}\n<Document>: {b}" for a, b in zip(q, d)],
                            add_special_tokens=False, truncation=True, max_length=self.max_body)["input_ids"]
            batch = self.tok.pad({"input_ids": [self.pre + x + self.suf for x in body]}, padding=True,
                                 pad_to_multiple_of=self.pad_multiple, return_tensors="pt")
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def logit(self, enc: dict):
        if self.arch == "encoder":
            return self.model(**enc).logits[:, 0].float()
        if self._keep:
            try:
                out = self.model(**enc, logits_to_keep=1)
            except TypeError:  # very old transformers
                self._keep = False
                out = self.model(**enc)
        else:
            out = self.model(**enc)
        z = out.logits[:, -1, :].float()
        return z[:, self.yes] - z[:, self.no]

    def save(self, path: str) -> None:
        m = getattr(self.model, "_orig_mod", self.model)   # unwrap torch.compile
        m.save_pretrained(path)


def _grad_scaler(torch, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _bucketed_batches(lens: np.ndarray, micro: int, rng, block: int = 64) -> list[np.ndarray]:
    """Shuffle, then sort within large blocks, then cut into micro-batches and shuffle the batch order.

    Random batches pad every batch to the longest member, which on short business strings wastes 30-50% of the
    tokens. Sorting inside a block makes each batch nearly uniform in length while keeping the sample order
    random enough for SGD (batches are drawn from all over the data, and their order is reshuffled).
    """
    n = len(lens)
    perm = rng.permutation(n)
    span = micro * block
    parts = []
    for s in range(0, n, span):
        blk = perm[s:s + span]
        parts.append(blk[np.argsort(lens[blk], kind="stable")])
    order = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    batches = [order[s:s + micro] for s in range(0, n, micro)]
    rng.shuffle(batches)
    return batches


def _group_batches(gid: np.ndarray, lens: np.ndarray, micro: int, rng, max_group: int = 24,
                   block: int = 64) -> list[np.ndarray]:
    """Batches of WHOLE S1 groups (<= micro rows each), so a listwise loss can see a full candidate list.

    Groups are shuffled, then sorted by length inside large blocks before packing (same padding trick as
    `_bucketed_batches`), and the finished batches are reshuffled so the optimizer does not walk the data in
    length order.
    """
    order = np.argsort(gid, kind="stable")
    g = gid[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if len(g) else np.zeros(0, np.int64)
    bounds = np.r_[starts, len(g)]
    groups = [order[bounds[k]:bounds[k + 1]][:max_group] for k in range(len(starts))]
    if not groups:
        return []
    glen = np.array([int(lens[x].max()) for x in groups])
    perm = rng.permutation(len(groups))
    span = max(1, block)
    ordered = []
    for s in range(0, len(perm), span):
        blk = perm[s:s + span]
        ordered.extend(blk[np.argsort(glen[blk], kind="stable")])
    batches, cur, n = [], [], 0
    for k in ordered:
        grp = groups[k]
        if n + len(grp) > micro and cur:
            batches.append(np.concatenate(cur))
            cur, n = [], 0
        cur.append(grp)
        n += len(grp)
    if cur:
        batches.append(np.concatenate(cur))
    idx = rng.permutation(len(batches))
    return [batches[k] for k in idx]


def _listwise_loss(z, y_t, spans, torch):
    """Softmax over each S1's candidate list plus an explicit 'none' option pinned at logit 0.

    The metric is macro per-entity F0.5 over a SET chosen per S1, and a record belongs to at most one S1, so
    the pairs inside one S1 group compete with each other and with predicting nothing. Independent per-pair BCE
    never sees that competition. Anchoring 'none' at logit 0 makes this the exact training-time counterpart of
    the `exclusive()` posterior o/(1 + sum o) the decision layer already applies, and it gives singleton S1s
    (no positives at all) a real gradient instead of just "all negative".
    """
    import torch.nn.functional as TF
    tot = z.new_zeros(())
    for s, e in spans:
        zg, yg = z[s:e], y_t[s:e]
        logp = TF.log_softmax(torch.cat([zg, zg.new_zeros(1)]), dim=0)
        pos = yg.sum()
        if float(pos) > 0:
            tgt = torch.cat([yg / pos, yg.new_zeros(1)])          # mass spread over the true matches
        else:
            tgt = torch.cat([torch.zeros_like(yg), yg.new_ones(1)])  # singleton: all mass on 'none'
        tot = tot - (tgt * logp).sum()
    return tot / max(1, len(spans))


def _spans(gb: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of equal group id inside a batch (batches are concatenations of whole groups)."""
    if len(gb) == 0:
        return []
    st = np.flatnonzero(np.r_[True, gb[1:] != gb[:-1]])
    return list(zip(st.tolist(), np.r_[st[1:], len(gb)].tolist()))


def _param_groups(model, spec: dict):
    """Backbone at `lr`, freshly initialized classification head at `head_lr` (it needs to move much faster)."""
    head_lr = spec.get("head_lr")
    if not head_lr:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": spec["lr"]}]
    head, body = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if ("classifier" in name or "score" in name or "pooler" in name) else body).append(p)
    groups = [{"params": body, "lr": spec["lr"]}]
    if head:
        groups.append({"params": head, "lr": float(head_lr)})
    return groups


def ce_fit(rr: Reranker, df: pd.DataFrame, spec: dict, rng, tag: str) -> dict:
    torch = rr.torch
    import torch.nn.functional as TF
    from transformers import get_cosine_schedule_with_warmup
    params = [p for p in rr.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(_param_groups(rr.model, spec), lr=spec["lr"], weight_decay=spec["wd"])
    micro, accum, epochs = int(spec["micro"]), int(spec["accum"]), int(spec["epochs"])
    n_steps = max(1, (math.ceil(len(df) / micro) * epochs) // accum)
    sch = get_cosine_schedule_with_warmup(opt, int(spec["warmup"] * n_steps), n_steps)
    scaler = _grad_scaler(torch, rr.dtype == torch.float16)
    amp = rr.dtype != torch.float32
    ta, tb = df["text_a"].values, df["text_b"].values
    y = df["label"].values.astype(np.float32)
    lens = rr.n_tokens(list(ta), list(tb))
    lw = float(spec.get("listwise_w") or 0.0)
    gid = pd.factorize(df["i"].values)[0] if lw > 0 else None
    # Competitor negatives are only a partial slice of their S1's candidate list, so a group softmax over them
    # would wrongly teach "this S1 matches nothing". They stay in the BCE term and sit out the listwise one.
    lwok = (df["lw_ok"].values.astype(bool) if "lw_ok" in df.columns else np.ones(len(df), bool))
    if lw > 0:
        log(f"  [{tag}] loss = BCE + {lw} x listwise (group softmax over each S1's candidates + 'none'); "
            f"{int(lwok.sum()):,}/{len(df):,} rows in listwise groups")
    rr.model.train()
    k = step = skipped = oom = 0
    ema, t0, seen = None, time.time(), 0
    for ep in range(epochs):
        batches = _group_batches(gid, lens, micro, rng) if lw > 0 else _bucketed_batches(lens, micro, rng)
        for b in batches:
            if len(b) == 0:
                continue
            sw = rng.random(len(b)) < spec["swap_p"]  # symmetric augmentation: S1 is not special
            q = np.where(sw, tb[b], ta[b]).tolist()
            d = np.where(sw, ta[b], tb[b]).tolist()
            try:
                with torch.autocast(rr.device, dtype=rr.dtype, enabled=amp):
                    z = rr.logit(rr.encode(q, d))
                yb = torch.from_numpy(y[b]).to(rr.device)
                loss = TF.binary_cross_entropy_with_logits(z.float(), yb)
                if lw > 0:
                    spans = [(s, e) for s, e in _spans(gid[b]) if lwok[b[s]]]
                    if spans:
                        loss = loss + lw * _listwise_loss(z.float(), yb, spans, torch)
                if torch.isfinite(loss):
                    scaler.scale(loss / accum).backward()
                    lv = float(loss.detach())
                    ema = lv if ema is None else 0.98 * ema + 0.02 * lv
                else:
                    skipped += 1
            except _oom_error(torch):
                # Drop this micro-batch rather than losing the run; if it keeps happening, lower ce micro.
                oom += 1
                opt.zero_grad(set_to_none=True)
                _empty_cache(torch)
                if oom == 1:
                    log(f"  [{tag}] CUDA OOM on a micro-batch; skipping it. "
                        f"If this repeats, re-run with --set ce.micro={max(1, micro // 2)}")
                continue
            seen += len(b)
            k += 1
            if k % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()
                step += 1
                if step % 50 == 0 or step == n_steps:
                    el = time.time() - t0
                    log(f"  [{tag}] ep {ep} step {step}/{n_steps} loss {ema:.4f} lr {sch.get_last_lr()[0]:.2e} "
                        f"{seen / el:,.0f} pairs/s ETA {el / step * (n_steps - step) / 60:.0f} min "
                        f"mem {_cuda_mem_gb(torch):.1f}G skipped {skipped} oom {oom}")
    del opt
    _empty_cache(torch)
    return {"steps": step, "final_loss_ema": ema, "skipped_nonfinite": skipped, "oom_batches": oom,
            "train_pairs": len(df), "listwise_w": lw,
            "pairs_per_s": round(seen / max(1e-9, time.time() - t0), 1),
            "minutes": round((time.time() - t0) / 60, 1)}


def ce_score(rr: Reranker, a: list[str], b: list[str], bs: int, tag: str = "", t0: float | None = None,
             done: int = 0, total: int | None = None) -> np.ndarray:
    """Score pairs, longest first, in length-sorted batches so padding is near-minimal."""
    torch = rr.torch
    rr.model.eval()
    if len(a) == 0:
        return np.zeros(0, np.float32)
    order = np.argsort(-rr.n_tokens(a, b), kind="stable")
    out = np.zeros(len(a), np.float32)
    n_bad = 0
    t0 = t0 or time.time()
    total = total or len(a)
    amp = rr.dtype != torch.float32
    s = 0
    with torch.no_grad():
        while s < len(order):
            k = order[s:s + bs]
            try:
                with torch.autocast(rr.device, dtype=rr.dtype, enabled=amp):
                    z = rr.logit(rr.encode([a[t] for t in k], [b[t] for t in k])).float().cpu().numpy()
            except _oom_error(torch):
                _empty_cache(torch)
                if bs <= 8:
                    raise
                bs = max(8, bs // 2)
                log(f"  CUDA OOM while scoring; halving infer_bs to {bs}")
                continue
            bad = ~np.isfinite(z)
            n_bad += int(bad.sum())
            z[bad] = 0.0
            out[k] = z
            s += len(k)
            if (s // max(1, bs)) % 200 == 0:
                el = time.time() - t0
                rate = (done + s) / max(1e-9, el)
                log(f"  score{tag} {done + s:,}/{total:,} ({rate:,.0f} pairs/s, "
                    f"eta {(total - done - s) / max(1.0, rate) / 60:.1f} min)")
    if n_bad:
        log(f"WARNING: {n_bad} non-finite CE logits set to 0 (fp16 overflow?)")
    return out


def ce_score_parquet(rr: Reranker, path: str, bs: int, gate=None, shards=None, batch: int = 200000,
                     want_label: bool = False, tag: str = "") -> pd.DataFrame:
    """Stream a CE-input parquet file through the model.

    The test file holds one `text_a`/`text_b` pair per pruned candidate -- tens of millions of rows and many GB
    of Python strings. Reading it whole (pd.read_parquet) is the most likely way for this stage to die on a
    cloud box, so rows are pulled a batch at a time, filtered by gate/shard, scored, and only the tiny
    (r, i, j, z) columns are kept.
    """
    pf = pq.ParquetFile(path)
    cols = ["r", "i", "j", "p1", "text_a", "text_b"] + (["label"] if want_label else [])
    cols += ["shard"] if (shards is not None and "shard" in pf.schema_arrow.names) else []
    cols += ["fold", "in_pruned"] if (want_label and "fold" in pf.schema_arrow.names) else []
    have = [c for c in dict.fromkeys(cols) if c in pf.schema_arrow.names]
    total = pf.metadata.num_rows
    parts, done, t0 = [], 0, time.time()
    for rb in pf.iter_batches(batch_size=batch, columns=have):
        d = rb.to_pandas()
        if shards is not None and "shard" in d.columns:
            d = d[d["shard"].isin(list(shards))]
        if gate:
            d = d[_gate(d["p1"].values, gate)]
        if len(d) == 0:
            continue
        z = ce_score(rr, d["text_a"].tolist(), d["text_b"].tolist(), bs, tag=tag, t0=t0, done=done, total=total)
        keep = {"r": d["r"].values, "i": d["i"].values, "j": d["j"].values, "z": z}
        if want_label and "label" in d.columns:
            keep["label"] = d["label"].values
        parts.append(pd.DataFrame(keep))
        done += len(d)
    if not parts:
        empty = {"r": [], "i": [], "j": [], "z": []}
        return pd.DataFrame({**empty, **({"label": []} if want_label else {})})
    out = pd.concat(parts, ignore_index=True)
    log(f"  scored{tag} {len(out):,} pairs of {total:,} rows in {(time.time() - t0) / 60:.1f} min")
    return out


def ce_in_path(cfg: dict, split: str) -> str:
    d = os.path.join(cfg["work_dir"], "ce_in")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{split}.parquet")


def _text_writer(u: Universe, path: str, cols: dict, I: np.ndarray, J: np.ndarray, chunk: int = 500000) -> None:
    """Write pair rows + raw-text serializations in chunks (texts are built on the fly, never all in memory)."""
    u.load(["name", "address", "country"])

    def texts(ix, dedup):
        n, a, c = (u.s(k, ix, dedup=dedup) for k in ("name", "address", "country"))
        return [record_text(x, y, z) for x, y, z in zip(n, a, c)]

    writer = None
    try:
        for st in range(0, len(I), chunk):
            sl = slice(st, st + chunk)
            t = {k: v[sl] for k, v in cols.items()}
            t["text_a"], t["text_b"] = texts(I[sl], True), texts(J[sl], False)
            tab = pa.table(t)
            writer = writer or pq.ParquetWriter(path + ".tmp", tab.schema)
            writer.write_table(tab)
    finally:
        if writer:
            writer.close()
    if writer is None:
        pq.write_table(pa.table({**{k: v[:0] for k, v in cols.items()}, "text_a": pa.array([], pa.string()),
                                 "text_b": pa.array([], pa.string())}), path + ".tmp")
    os.replace(path + ".tmp", path)


def export_ce_inputs(cfg: dict) -> None:
    """train: sampled S1 entities only -- pruned pairs (scored OOF) + a wider hard-negative pool (training);
    test: pruned pairs, each assigned to ONE fold model (shard = S1 hash % n_folds), exactly like OOF rows."""
    t0 = time.time()
    nf = cfg["cv"]["n_folds"]
    tr = build_universe(cfg, "train")
    ps = PairStore(cfg, "train")
    I, J, p1, L = ps.load("i").astype(np.int64), ps.load("j").astype(np.int64), ps.load("p1"), ps.load("label")
    pruned = np.zeros(len(I), bool)
    pruned[ps.load("pruned")] = True
    samp, folds = sample_mask(cfg, tr), get_folds(cfg, tr)
    rk = group_context(I, np.where(np.isfinite(p1), p1, -1.0))[0]
    rows = np.flatnonzero(samp[I] & (pruned | (rk <= cfg["ce"].get("train_rank_s1", 10))))
    # Record-side competitor negatives: for each record already in the pool, the best-scoring OTHER S1
    # entities claiming it -- even when those S1s are outside the training sample. These are the chain-branch
    # confusions ("same franchise, different address") that decide most false merges, and without them the CE
    # only ever sees one S1's view of a record. Training-only (lw_ok=0, in_pruned=0): never scored as OOF.
    npr = int(cfg["ce"].get("neg_per_rec") or 0)
    extra = np.zeros(0, np.int64)
    if npr > 0 and len(rows):
        invol = np.zeros(tr.n, bool)
        invol[J[rows]] = True
        cand = np.flatnonzero(invol[J] & ~samp[I] & np.isfinite(p1))
        if len(cand):
            extra = cand[group_context(J[cand], p1[cand])[0] <= npr]
    all_rows = np.sort(np.concatenate([rows, extra])) if len(extra) else rows
    is_extra = ~samp[I[all_rows]]
    _text_writer(tr, ce_in_path(cfg, "train"),
                 {"r": all_rows, "i": I[all_rows], "j": J[all_rows], "p1": p1[all_rows].astype(np.float32),
                  "label": L[all_rows], "fold": folds[I[all_rows]],
                  "in_pruned": (pruned[all_rows] & ~is_extra).astype(np.int8),
                  "lw_ok": (~is_extra).astype(np.int8)}, I[all_rows], J[all_rows])
    rows = all_rows
    te = build_universe(cfg, "test")
    pt = PairStore(cfg, "test")
    rows_t = pt.load("pruned")
    It, Jt = pt.load("i")[rows_t].astype(np.int64), pt.load("j")[rows_t].astype(np.int64)
    _text_writer(te, ce_in_path(cfg, "test"),
                 {"r": rows_t, "i": It, "j": Jt, "p1": pt.load("p1")[rows_t].astype(np.float32),
                  "shard": (_hash01(It) * nf).astype(np.int64) % nf}, It, Jt)
    log(f"ce inputs: train {len(rows):,} pairs ({int(pruned[rows].sum()):,} pruned, {len(extra):,} "
        f"competitor negatives) for {int(samp.sum()):,} sampled S1; "
        f"test {len(rows_t):,} pairs ({time.time() - t0:.0f}s)")


def _ce_inputs_fresh(cfg: dict) -> bool:
    out = [ce_in_path(cfg, s) for s in ("train", "test")]
    src = [PairStore(cfg, s).path("pruned") for s in ("train", "test")]
    return all(os.path.exists(p) for p in out) and min(map(os.path.getmtime, out)) >= max(map(os.path.getmtime, src))


def _gate(p1: np.ndarray, gate) -> np.ndarray:
    if not gate:
        return np.ones(len(p1), bool)
    lo, hi = float(gate[0]), float(gate[1])
    return (p1 > lo) & (p1 < hi)


def ce_plan(cfg: dict) -> list[dict]:
    """Which CE models to train, and what each one scores.

    The original scheme trains one model per CV fold: 4 models, each on 3/4 of the sample = 300% of the data.
    Every model is only ever used to score its own held-out fold, so the folds can be grouped instead:
    with n_models=2 each model trains on half the sample and scores the other half OOF plus its half of the
    test shards -- identical OOF and test coverage for 1/3 of the training compute.
    n_models=1 trains a single model on half the folds and scores ALL of test; OOF then covers half the sample,
    which is still plenty for the L2 stacker but leaves the other half of the sample without a CE feature.
    """
    nf = int(cfg["cv"]["n_folds"])
    nm = max(1, min(int(cfg["ce"].get("n_models") or nf), nf))
    plans = []
    for k, g in enumerate(np.array_split(np.arange(nf), nm)):
        oof = [int(x) for x in g]
        train = [f for f in range(nf) if f not in oof]
        shards = oof
        if not train:                       # nm == 1
            half = max(1, nf // 2)
            train, oof, shards = list(range(half)), list(range(half, nf)), list(range(nf))
        plans.append({"id": k, "train_folds": train, "oof_folds": oof, "test_shards": shards})
    return plans


def ce_fold(cfg: dict, name: str, mid: int) -> None:
    """Train CE model `mid` of the plan and score its OOF folds and its test shards."""
    import torch
    spec = ce_spec(cfg, name)
    gate = cfg["ce"].get("gate")
    plan = {p["id"]: p for p in ce_plan(cfg)}
    if mid not in plan:
        raise KeyError(f"no CE model {mid} in the plan (ids {sorted(plan)}); check ce.n_models")
    pl = plan[mid]
    d = ce_dir(cfg, name)
    oof_p, test_p = os.path.join(d, f"oof_f{mid}.parquet"), os.path.join(d, f"test_f{mid}.parquet")
    if os.path.exists(oof_p) and os.path.exists(test_p):
        log(f"ce {name} model {mid}: exists, skipping")
        return
    tr_path = _need(ce_in_path(cfg, "train"), "ce")
    rng = np.random.default_rng(int(spec["seed"]) + mid)
    torch.manual_seed(int(spec["seed"]) + mid)
    fit_df = pq.read_table(tr_path, filters=[("fold", "in", pl["train_folds"])]).to_pandas()
    rows = select_ce_train(fit_df, spec, rng)
    del fit_df
    log(f"ce {name} model {mid}: train on folds {pl['train_folds']} -> {len(rows):,} pairs "
        f"({int(rows['label'].sum()):,} pos); OOF folds {pl['oof_folds']}, test shards {pl['test_shards']}, "
        f"gate={gate}; spec={spec}")
    rr = Reranker(spec, train=True)
    info = ce_fit(rr, rows, spec, rng, f"{name}/m{mid}")
    del rows
    rr.save(os.path.join(d, f"adapter_f{mid}"))
    va = pq.read_table(tr_path, filters=[("fold", "in", pl["oof_folds"]), ("in_pruned", "=", 1)]).to_pandas()
    if gate:
        va = va[_gate(va["p1"].values, gate)]
    zv = ce_score(rr, va["text_a"].tolist(), va["text_b"].tolist(), int(spec["infer_bs"]), tag=" oof")
    pd.DataFrame({"r": va["r"].values, "i": va["i"].values, "j": va["j"].values, "z": zv,
                  "label": va["label"].values}).to_parquet(oof_p, index=False)
    info.update(binary_metrics(va["label"].values, sigmoid(zv)))
    del va
    T = ce_score_parquet(rr, _need(ce_in_path(cfg, "test"), "ce"), int(spec["infer_bs"]), gate=gate,
                         shards=pl["test_shards"], tag=" test")
    T.to_parquet(test_p, index=False)
    info["test_pairs_scored"] = int(len(T))
    log(f"ce {name} model {mid} done: {info}")
    log_exp(cfg, f"ce_model{mid}", info, note=name)


def ce_merge(cfg: dict, name: str) -> dict:
    d = ce_dir(cfg, name)
    ids = [p["id"] for p in ce_plan(cfg)]
    oofs = [os.path.join(d, f"oof_f{k}.parquet") for k in ids]
    tests = [os.path.join(d, f"test_f{k}.parquet") for k in ids]
    miss = [p for p in oofs + tests if not os.path.exists(p)]
    if miss:
        raise RuntimeError(f"ce {name}: missing model outputs {miss}")
    O = pd.concat([pd.read_parquet(p) for p in oofs], ignore_index=True)
    T = pd.concat([pd.read_parquet(p) for p in tests], ignore_index=True)
    O[["r", "i", "j", "z"]].to_parquet(os.path.join(d, "train.parquet"), index=False)
    T[["r", "i", "j", "z"]].to_parquet(os.path.join(d, "test.parquet"), index=False)
    met = binary_metrics(O["label"].values, sigmoid(O["z"].values)) if len(O) else {}
    log(f"ce {name} merged: OOF {met}; {len(O):,} OOF / {len(T):,} test pairs")
    log_exp(cfg, "ce_merge", met, note=name)
    return met


def stage_ce(cfg: dict, name: str, gpus: list[int], models: list[int] | None = None) -> None:
    t0 = time.time()
    if not _ce_inputs_fresh(cfg):
        export_ce_inputs(cfg)
    plans = ce_plan(cfg)
    ids = models if models is not None else [p["id"] for p in plans]
    log(f"ce {name}: {len(plans)} model(s) on gpus {gpus} -- "
        + "; ".join(f"m{p['id']}: train{p['train_folds']} oof{p['oof_folds']} shards{p['test_shards']}"
                    for p in plans if p["id"] in ids))
    per_gpu = {g: [f for k, f in enumerate(ids) if k % len(gpus) == gi] for gi, g in enumerate(gpus)}
    run_workers([(g, ["ce-worker", "--name", name, "--folds", ",".join(map(str, fl))])
                 for g, fl in per_gpu.items() if fl])
    ce_merge(cfg, name)
    log(f"ce {name}: total {(time.time() - t0) / 60:.1f} min")


def stage_gpu_check(cfg: dict, name: str) -> None:
    """Load the CE model, run one train step and a scoring pass on a few pairs; report dtype, memory, speed."""
    import torch
    log(f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
    for g in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(g)
        log(f"  gpu{g}: {p.name} cc={p.major}.{p.minor} mem={p.total_memory / 2 ** 30:.1f}G")
    spec = ce_spec(cfg, name)
    log(f"dtype -> {pick_dtype(torch, spec.get('dtype', 'auto'))}")
    if os.path.exists(ce_in_path(cfg, "train")):
        df = pd.read_parquet(ce_in_path(cfg, "train")).head(256)
    else:
        a = record_text("Shree Ganesh Traders Pvt Ltd", "12 MG Road, Near SBI ATM, Pune 411001", "India")
        b = record_text("Sri Ganesh Traders", "12, M.G. Rd, Pune, Maharashtra", "India")
        df = pd.DataFrame({"text_a": [a] * 256, "text_b": [b] * 256, "label": [1, 0] * 128, "p1": 0.5, "i": 0})
    rr = Reranker(spec, train=True)
    torch.cuda.reset_peak_memory_stats()
    info = ce_fit(rr, df.head(4 * spec["micro"]).reset_index(drop=True), dict(spec, epochs=1, accum=1),
                  np.random.default_rng(0), "gpu-check")
    log(f"train ok: {info}; peak mem {_cuda_mem_gb(torch):.2f}G")
    t0 = time.time()
    z = ce_score(rr, df["text_a"].tolist(), df["text_b"].tolist(), int(spec["infer_bs"]))
    rate = len(z) / max(1e-9, time.time() - t0)
    log(f"scoring ok: {rate:.0f} pairs/s on this GPU; z[:4]={np.round(z[:4], 3).tolist()}")
    # How much work the CE actually faces, and what the gate removes: the numbers that decide the wall clock.
    gate = cfg["ce"].get("gate")
    for split in ("train", "test"):
        p = ce_in_path(cfg, split)
        if not os.path.exists(p):
            continue
        n = pq.ParquetFile(p).metadata.num_rows
        kept = n
        if gate:
            kept = 0
            for rb in pq.ParquetFile(p).iter_batches(batch_size=500000, columns=["p1"]):
                kept += int(_gate(rb.column("p1").to_numpy(), gate).sum())
        log(f"  {split} ce inputs: {n:,} pairs -> {kept:,} after gate={gate} ({kept / max(1, n):.1%})")
        if split == "test" and rate > 0:
            n_models = len(ce_plan(cfg))
            log(f"  => est. test scoring {kept / rate / 3600:.2f} GPU-hours total across {n_models} model(s); "
                f"ungated would be {n / rate / 3600:.2f}")


# =============================================================================================================
# DATA DOWNLOAD (Hugging Face mirror of the official student_resource kit)
# =============================================================================================================
def stage_download(cfg: dict, repo: str | None = None, out: str = ".") -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install -U huggingface_hub   (optionally: pip install hf_xet  for faster downloads)")
    repo = repo or cfg["hf_repo"]
    t0 = time.time()
    log(f"downloading dataset repo {repo} -> {os.path.abspath(out)} (~2.5 GB)")
    path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out,
                             allow_patterns=["dataset/*", "utils/*", "Documentation_template.md"],
                             ignore_patterns=["*.DS_Store"])
    need = [os.path.join(path, "dataset", s, f"{s}_source{k}.tsv") for s in ("train", "test") for k in (1, 2, 3)]
    need.append(os.path.join(path, "dataset", "train", "train_ground_truth.tsv"))
    for p in need:
        log(f"  {'OK ' if os.path.exists(p) else 'MISSING'} {p} "
            f"({os.path.getsize(p) / 2 ** 20:,.0f} MB)" if os.path.exists(p) else f"  MISSING {p}")
    if not all(map(os.path.exists, need)):
        raise SystemExit("download incomplete")
    if os.path.abspath(os.path.join(out, "dataset")) != os.path.abspath(cfg["data_dir"]):
        log(f"NOTE: data_dir is '{cfg['data_dir']}'; run with --set data_dir={os.path.join(out, 'dataset')}")
    log(f"download done in {(time.time() - t0) / 60:.1f} min")

# =============================================================================================================
# SYNTHETIC DATA (competition format) -- for smoke-testing ONLY; scores on it say nothing about the leaderboard
# =============================================================================================================
_SW = {
    "us": {"a": ["Blue", "Golden", "Summit", "Pioneer", "Liberty", "Maple", "Eagle", "Atlas", "Harbor", "Cedar",
                 "Northstar", "Redwood", "Silver", "Granite", "Evergreen", "Lakeside", "Prairie", "Keystone"],
           "b": ["Bakery", "Dental Care", "Auto Repair", "Consulting", "Pharmacy", "Grill", "Hardware", "Logistics",
                 "Insurance Agency", "Fitness", "Plumbing", "Realty", "Law Group", "Coffee", "Printing", "Cleaners"],
           "legal": ["LLC", "Inc", "Inc.", "Corp", "Corporation", "Co", "Company", "", "", ""],
           "street": ["Main", "Oak", "Pine", "Washington", "Lincoln", "Park", "Lake", "Hill", "Elm", "Madison"],
           "stype": [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr")],
           "city": [("Springfield", "IL", "627"), ("Austin", "TX", "787"), ("Denver", "CO", "802"),
                    ("Columbus", "OH", "432"), ("Portland", "OR", "972"), ("Raleigh", "NC", "276")]},
    "in": {"a": ["Shree", "Sri", "Laxmi", "Ganesh", "Balaji", "Krishna", "Sai", "Om", "Durga", "Mahalaxmi",
                 "Jai Hind", "Bharat", "Annapurna", "Siddhi Vinayak", "Gurukripa", "Ambika"],
           "b": ["Traders", "Enterprises", "Textiles", "Medicals", "Sweets", "Hardware", "Electricals", "Jewellers",
                 "Motors", "Kirana Store", "Steel Works", "Agencies", "Caterers", "Opticals"],
           "legal": ["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "LLP", "", "", ""],
           "street": ["Gandhi", "Nehru", "Shivaji", "Tilak", "Patel", "Subhash", "Ambedkar", "Rajiv"],
           "stype": [("Road", "Rd"), ("Nagar", "Ngr"), ("Marg", "Mg"), ("Colony", "Col"), ("Market", "Mkt")],
           "city": [("Pune", "Maharashtra", "411"), ("Jaipur", "Rajasthan", "302"), ("Indore", "Madhya Pradesh", "452"),
                    ("Lucknow", "Uttar Pradesh", "226"), ("Surat", "Gujarat", "395")],
           "landmark": ["Near SBI ATM", "Opp. City Mall", "Behind Bus Stand", "Near Hanuman Mandir"]},
    "fr": {"a": ["Boulangerie", "Pharmacie", "Garage", "Cabinet", "Librairie", "Café", "Boucherie", "Fromagerie"],
           "b": ["du Centre", "de la Gare", "Martin", "Dupont", "Saint-Michel", "des Arts", "Bernard", "Lefèvre"],
           "legal": ["SARL", "SAS", "SA", "EURL", "", "", ""],
           "street": ["de la République", "Victor Hugo", "du Général Leclerc", "Jean Jaurès", "de l'Église"],
           "stype": [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl."), ("Impasse", "Imp.")],
           "city": [("Paris", "", "750"), ("Lyon", "", "690"), ("Marseille", "", "130"), ("Toulouse", "", "310")]},
}
_SCOUNTRY = {"us": "US", "in": "India", "fr": "France"}
# Devanagari forms of the synthetic India vocabulary, so `smoke` exercises the cross-script path that
# 23% of real India S2 true pairs depend on.
_DEVA = {
    "Shree": "\u0936\u094d\u0930\u0940", "Sri": "\u0936\u094d\u0930\u0940", "Laxmi": "\u0932\u0915\u094d\u0937\u094d\u092e\u0940",
    "Ganesh": "\u0917\u0923\u0947\u0936", "Balaji": "\u092c\u093e\u0932\u093e\u091c\u0940", "Krishna": "\u0915\u0943\u0937\u094d\u0923\u093e",
    "Sai": "\u0938\u093e\u0908", "Om": "\u0913\u092e", "Durga": "\u0926\u0941\u0930\u094d\u0917\u093e", "Mahalaxmi": "\u092e\u0939\u093e\u0932\u0915\u094d\u0937\u094d\u092e\u0940",
    "Bharat": "\u092d\u093e\u0930\u0924", "Ambika": "\u0905\u0902\u092c\u093f\u0915\u093e", "Annapurna": "\u0905\u0928\u094d\u0928\u092a\u0942\u0930\u094d\u0923\u093e",
    "Traders": "\u091f\u094d\u0930\u0947\u0921\u0930\u094d\u0938", "Enterprises": "\u090f\u0902\u091f\u0930\u092a\u094d\u0930\u093e\u0907\u091c\u0947\u091c",
    "Textiles": "\u091f\u0947\u0915\u094d\u0938\u091f\u093e\u0907\u0932\u094d\u0938", "Medicals": "\u092e\u0947\u0921\u093f\u0915\u0932\u094d\u0938",
    "Sweets": "\u0938\u094d\u0935\u0940\u091f\u094d\u0938", "Hardware": "\u0939\u093e\u0930\u094d\u0921\u0935\u0947\u092f\u0930",
    "Electricals": "\u0907\u0932\u0947\u0915\u094d\u091f\u094d\u0930\u093f\u0915\u0932\u094d\u0938", "Jewellers": "\u091c\u094d\u0935\u0947\u0932\u0930\u094d\u0938",
    "Motors": "\u092e\u094b\u091f\u0930\u094d\u0938", "Agencies": "\u090f\u091c\u0947\u0902\u0938\u0940\u091c", "Caterers": "\u0915\u0947\u091f\u0930\u0930\u094d\u0938",
    "Opticals": "\u0911\u092a\u094d\u091f\u093f\u0915\u0932\u094d\u0938",
}
_DEVA_STATE = {"Maharashtra": "\u092e\u0939\u093e\u0930\u093e\u0937\u094d\u091f\u094d\u0930", "Rajasthan": "\u0930\u093e\u091c\u0938\u094d\u0925\u093e\u0928",
               "Gujarat": "\u0917\u0941\u091c\u0930\u093e\u0924", "Uttar Pradesh": "\u0909\u0924\u094d\u0924\u0930 \u092a\u094d\u0930\u0926\u0947\u0936",
               "Madhya Pradesh": "\u092e\u0927\u094d\u092f \u092a\u094d\u0930\u0926\u0947\u0936"}
_DEVA_DIGITS = str.maketrans("0123456789", "\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f")
_TRANSLIT = {"Shree": ["Sri", "Shri"], "Sri": ["Shree", "Shri"], "Laxmi": ["Lakshmi"], "Mahalaxmi": ["Mahalakshmi"],
             "Ganesh": ["Ganesha"], "Krishna": ["Krushna"], "Jewellers": ["Jewelers"]}


def _typo(s: str, rng: random.Random) -> str:
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.random()
    if op < 0.33:
        return s[:i] + s[i + 1:]
    if op < 0.66:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + rng.choice("aeiourstn") + s[i:]


def _make_entity(c, rng, chain_of=None, colo_addr=None):
    w = _SW[c]
    name = chain_of["name"] if chain_of else f"{rng.choice(w['a'])} {rng.choice(w['b'])}"
    if colo_addr:
        addr = dict(colo_addr)
    elif chain_of is not None and rng.random() < 0.5:  # branch on the same street: only the number differs
        addr = dict(chain_of["addr"])
        addr["num"] = str(rng.randint(1, 999 if c != "us" else 9999))
    else:
        city, state, pfx = rng.choice(w["city"])
        st_full, st_abbr = rng.choice(w["stype"])
        addr = {"num": str(rng.randint(1, 999 if c != "us" else 9999)), "street": rng.choice(w["street"]),
                "st_full": st_full, "st_abbr": st_abbr, "city": city, "state": state,
                "postal": pfx + str(rng.randint(0, 99 if c != "in" else 999)).zfill(2 if c != "in" else 3),
                "landmark": rng.choice(w.get("landmark", [""]))}
    return {"c": c, "name": name, "legal": rng.choice(w["legal"]), "addr": addr}


def _render(e, rng, noisy):
    c, a, name, legal = e["c"], e["addr"], e["name"], e["legal"]
    if noisy:
        toks = [rng.choice(_TRANSLIT[t]) if t in _TRANSLIT and rng.random() < 0.5 else t for t in name.split()]
        if rng.random() < 0.15 and len(toks) > 1:
            toks[0], toks[1] = toks[1], toks[0]
        if rng.random() < 0.35:
            j = rng.randrange(len(toks))
            toks[j] = _typo(toks[j], rng)
        name = " ".join(toks)
        if c == "fr" and rng.random() < 0.5:
            name = name.replace("é", "e").replace("è", "e")
        r = rng.random()
        legal = "" if r < 0.3 else (rng.choice(_SW[c]["legal"]) if r < 0.5 else legal)
        if rng.random() < 0.1:
            name = name.upper()
    full_name = f"{name} {legal}".strip()
    num = "" if (noisy and rng.random() < 0.25) else a["num"]  # missing house numbers make branches ambiguous
    st = a["st_abbr"] if (noisy and rng.random() < 0.5) else a["st_full"]
    drop_postal, drop_state = noisy and rng.random() < 0.3, noisy and rng.random() < 0.4
    if c == "us":
        addr = f"{num} {a['street']} {st}, {a['city']}".strip() + ("" if drop_state else f", {a['state']}") + \
               ("" if drop_postal else f" {a['postal']}")
    elif c == "in":
        parts = [f"{num}, {a['street']} {st}" if num else f"{a['street']} {st}"]
        if not noisy or rng.random() < 0.5:
            parts.append(a["landmark"])
        parts.append(a["city"])
        if not drop_state:
            parts.append(a["state"] + ("" if drop_postal else f" {a['postal']}"))
        addr = ", ".join(parts)
    else:
        addr = f"{num} {st} {a['street']}, ".lstrip() + ("" if drop_postal else f"{a['postal']} ") + a["city"]
        if noisy and rng.random() < 0.5:
            addr = addr.replace("é", "e").replace("è", "e")
    if noisy and rng.random() < 0.15:
        addr = _typo(addr, rng)
    # ~28% of real India S2/S3 names arrive in a native script while their S1 stays ASCII. Reproduce that here
    # so the smoke test covers the cross-script case instead of quietly passing on Latin-only data.
    if noisy and c == "in" and rng.random() < 0.28:
        toks = [_DEVA.get(t, t) for t in name.split()]
        if any(t in _DEVA.values() for t in toks):
            full_name = " ".join(toks)
        for full, dev in _DEVA_STATE.items():
            addr = addr.replace(full, dev)
        if rng.random() < 0.5:
            addr = addr.translate(_DEVA_DIGITS)
    return full_name, addr


def generate_synthetic(out_dir: str, n_entities: int = 3000, seed: int = 0) -> None:
    """Competition-format synthetic data with the real data's shape: random 9-digit ids, ~4 matches per S1
    spread over S2 and S3, ~8% singletons, chains / co-located businesses, unseen France in test."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    for split, countries in (("train", ["us", "in"]), ("test", ["us", "in", "fr"])):
        ents, by_c = [], {c: [] for c in countries}
        for _ in range(n_entities):
            c = rng.choice(countries)
            r = rng.random()
            same = by_c[c]
            if r < 0.08 and same:
                e = _make_entity(c, rng, chain_of=rng.choice(same))
            elif r < 0.13 and same:
                e = _make_entity(c, rng, colo_addr=rng.choice(same)["addr"])
            else:
                e = _make_entity(c, rng)
            ents.append(e)
            same.append(e)
        s1, s2, s3, gt = [], [], [], []
        order = list(range(len(ents)))
        rng.shuffle(order)
        for k in order:
            e = ents[k]
            in_s1 = rng.random() < 0.8
            if in_s1 and rng.random() < 0.08:
                n2 = n3 = 0
            elif in_s1:
                n2, n3 = rng.choice([0, 1, 1, 2, 2, 3, 4]), rng.choice([0, 1, 1, 2, 2, 3, 4])
                if n2 + n3 == 0:
                    n3 = 1
            else:
                n2, n3 = rng.choice([0, 1, 1, 2]), rng.choice([0, 1, 1, 2])
            ids2, ids3 = [len(s2) + i for i in range(n2)], [len(s3) + i for i in range(n3)]
            s2 += [(_render(e, rng, True), _SCOUNTRY[e["c"]]) for _ in range(n2)]
            s3 += [(_render(e, rng, True), _SCOUNTRY[e["c"]]) for _ in range(n3)]
            if in_s1:
                s1.append((_render(e, rng, False), _SCOUNTRY[e["c"]]))
                gt.append((len(s1) - 1, ids2, ids3))
        idn = {p: nrng.choice(10 ** 9 - 10 ** 6, size=n, replace=False) + 10 ** 6
               for p, n in (("S1", len(s1)), ("S2", len(s2)), ("S3", len(s3)))}
        name = {p: [f"{p}-{x}" for x in idn[p]] for p in idn}
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        for rows, pref, fn in ((s1, "S1", 1), (s2, "S2", 2), (s3, "S3", 3)):
            perm = list(range(len(rows)))
            rng.shuffle(perm)
            with open(os.path.join(d, f"{split}_source{fn}.tsv"), "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("".join(f"{name[pref][k]}\t{rows[k][0][0]}\t{rows[k][0][1]}\t{rows[k][1]}\n" for k in perm))
        path = os.path.join(d, f"{split}_ground_truth.tsv" if split == "train" else f"{split}_ground_truth_HIDDEN.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for i, ids2, ids3 in gt:
                f.write(f"{name['S1'][i]}\t{','.join([name['S2'][x] for x in ids2] + [name['S3'][x] for x in ids3])}\n")
    log(f"synthetic data written to {out_dir}")


def _read_lists(path: str) -> dict:
    t = read_tsv_arrow(path, _header(path)[:2])
    a, b = t.column(0).to_pylist(), t.column(1).to_pylist()
    return {x: set(y for y in z.split(",") if y) for x, z in zip(a, b)}


def stage_smoke(base: str = "smoke_run", n_entities: int = 3000) -> None:
    """End-to-end test on synthetic data: sampled training (sample < #S1), several feature parts, forked workers,
    a FAKE cross-encoder with a gate (exercises the L2 merge path), LOCO, submission checks and packaging."""
    if os.path.exists(base):
        shutil.rmtree(base)
    generate_synthetic(os.path.join(base, "dataset"), n_entities, 0)
    cfg = load_config(None, [f"data_dir={base}/dataset", f"work_dir={base}/work", f"output_dir={base}/output",
                             "n_jobs=2", f"sample_s1={int(n_entities * 0.45)}", "features.chunk_pairs=6000",
                             "lgb.num_boost_round=400", "prep.chunk=4000"])
    os.makedirs(cfg["work_dir"], exist_ok=True)
    stage_prep(cfg)
    stage_eda(cfg)
    stage_cpu(cfg)
    export_ce_inputs(cfg)
    rng = np.random.default_rng(0)
    d = ce_dir(cfg, "fakece")
    # fake CE = noisy function of the L1 logit (NOT a model; only tests plumbing). Real one: `ce` command.
    E = pd.read_parquet(ce_in_path(cfg, "train"))
    E = E[(E["in_pruned"] == 1) & _gate(E["p1"].values, [0.01, 0.99])]
    z = logit(E["p1"].values) + rng.normal(0, 1.0, len(E))
    pd.DataFrame({"r": E["r"], "i": E["i"], "j": E["j"], "z": z.astype(np.float32)}).to_parquet(os.path.join(d, "train.parquet"))
    T = pd.read_parquet(ce_in_path(cfg, "test"))
    T = T[_gate(T["p1"].values, [0.01, 0.99])]
    z = logit(T["p1"].values) + rng.normal(0, 1.0, len(T))
    pd.DataFrame({"r": T["r"], "i": T["i"], "j": T["j"], "z": z.astype(np.float32)}).to_parquet(os.path.join(d, "test.parquet"))
    stage_l2(cfg)
    stage_decide(cfg)
    stage_loco(cfg)
    te = build_universe(cfg, "test")
    truth = _read_lists(os.path.join(cfg["test_dir"], "test_ground_truth_HIDDEN.tsv"))
    pred = _read_lists(os.path.join(cfg["output_dir"], "matching_results.tsv"))
    ids, cty = te.s("entity_id", te.s1_rows), te.cty_str(te.s1_rows)
    for g in pd.unique(cty):
        m = cty == g
        log(f"SMOKE synthetic test {g}: F0.5={macro_f05(pred, truth, ids[m]):.5f} (n={int(m.sum())})")
    log(f"SMOKE synthetic test ALL: {macro_f05(pred, truth, ids):.5f}")
    stage_package(cfg, "smoke_team", out_dir=base)


# =============================================================================================================
# PACKAGING
# =============================================================================================================
REQUIREMENTS_TXT = """numpy>=1.26
pandas>=2.1
scipy>=1.11
scikit-learn>=1.3
lightgbm>=4.3
rapidfuzz>=3.6
pyarrow>=14
sparse_dot_topn>=1.1
pyyaml>=6
huggingface_hub>=0.25   # `download` command only
# GPU stages (ce / gpu-check)
torch>=2.3
transformers>=4.51
sentencepiece>=0.2      # xlm-roberta / mdeberta tokenizers
peft>=0.13              # only for the causal presets (ce06 / ce4b)
accelerate>=0.33
bitsandbytes>=0.43      # only for --name ce4b (4-bit)
"""

README_MD = """# Business Entity Resolution (Amazon ML Challenge 2026)

Single-file pipeline: `src/entity_resolution.py`. Python >= 3.10. `pip install -r requirements.txt`.
Run from the folder that contains `dataset/` (the student_resource root). Outputs go to `output/`.

```
python src/entity_resolution.py download                  # optional: fetch dataset/ from the HF mirror of the kit
python src/entity_resolution.py cpu                       # prep, block, feat, l1, prune, l2, decide -> output/*.tsv
python src/entity_resolution.py ce --name ce06 --gpus 0,1,2,3   # optional cross-encoder (one CV fold per GPU)
python src/entity_resolution.py l2 && python src/entity_resolution.py decide
python src/entity_resolution.py loco                      # leave-one-country-out report
python3 utils/validate_submission.py --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Pipeline: normalization -> hashed token-IDF blocking (name / address / mixed views, per country, both directions)
-> LightGBM L1 on ~100 pair features (trained on a sample of S1 entities, scores every pair) -> pruning
(= candidate_pairs.tsv) -> optional Qwen3-Reranker-0.6B + LoRA cross-encoder -> LightGBM L2 stacker with
competition / exclusivity / neighbour context -> cross-fitted F0.5-optimal decision layer.
All experiment metrics: `work/experiments.jsonl`. Models: Qwen/Qwen3-Reranker-0.6B (Apache-2.0). No external data.
"""


def build_documentation(cfg: dict) -> str:
    lines = []
    exp = os.path.join(cfg["work_dir"], "experiments.jsonl")
    if os.path.exists(exp):
        with open(exp) as f:
            recs = [json.loads(x) for x in f if x.strip()]
        last = {}
        for r in recs:
            last[(r["stage"], r.get("run"))] = r
        for (st, run), r in sorted(last.items(), key=lambda kv: kv[1]["time"]):
            lines.append(f"- `{st}` ({run}): {json.dumps(r['metrics'], default=str)[:400]}")
    results = "\n".join(lines) or "- (run the pipeline to fill this section)"
    return f"""# Documentation -- Business Entity Resolution

## 1. Methodology
Cascade: country-agnostic normalization -> scalable token-IDF blocking -> level-1 LightGBM on ~100 pair
features -> pruning -> optional fine-tuned cross-encoder (Qwen3-Reranker, LoRA, yes/no head) -> level-2
LightGBM stacker with competition / exclusivity / neighbour-support features -> decision layer that maximizes
macro per-entity F0.5. Only the provided training data is used; no external data, lookups or geocoding.

## 2. Blocking strategy
Each record is turned into hashed tokens: core-name words, 4-character prefixes and phonetic skeletons (name view);
address words, postal code and house number + street word (address view); first name word combined with the
postal prefix and with the last address words (mixed tokens, full view only). Per country (open set; unseen
countries get their own statistics), tokens are IDF-weighted; very frequent tokens are not used for retrieval.
Cosine top-k is computed with sparse matrix products in both directions (S1 -> records: 15/6/6 for the
full/name/address views; records -> S1: 3/1/1). Every candidate pair is scored by every view, plus rarity-aware
shared/unmatched token masses and token-set similarities, with rank/gap/margin context inside the S1 group and
the record group. L1 LightGBM then prunes to p >= 0.002, rank <= 12 per S1 and <= 6 per record; that set is
exactly `candidate_pairs.tsv`, and matches are always a subset of it.

## 3. Model and features
Features: fuzzy similarities (ratio, token-set/sort, partial, Jaro-Winkler, Levenshtein) on normalized name,
core name, phonetic skeleton, address, address words and address tail; postal/house-number/legal-form/digit
agreement (agree/conflict/missing); acronym matches; IDF-weighted shared and unmatched token mass; chain-name and
shared-address frequencies; landmark and DBA similarity; blocking scores with competition context.
Scale: models are trained on a random sample of S1 entities (with all their candidates); a full-sample model
scores every other pair, so all train scores used downstream are out-of-sample.
Decision: records belong to at most one S1, so scores can be converted with an exclusivity posterior; the policy
(single threshold, top-1/rest thresholds, or exact expected-F0.5 maximization) is chosen by cross-fitting.

## 4. Validation
Folds by S1 entity (4 folds; candidates generated on the full train universe so distractor density matches
test); same folds for every stacking level; decision parameters cross-fitted; leave-one-country-out as a proxy
for the unseen country.

## 5. Results (from work/experiments.jsonl)
{results}
"""


def stage_package(cfg: dict, team: str, out_dir: str = ".") -> str:
    path = os.path.join(out_dir, f"{team}_submission.zip")
    out = cfg["output_dir"]
    doc = "Documentation_template.md"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(_need(os.path.join(out, fn), "decide"), f"output/{fn}")
        z.write(os.path.abspath(__file__), "code/business_entity_resolution/src/entity_resolution.py")
        z.writestr("code/business_entity_resolution/README.md", README_MD)
        z.writestr("code/business_entity_resolution/requirements.txt", REQUIREMENTS_TXT)
        if os.path.exists(doc):  # your own filled template wins
            z.write(doc, doc)
        else:
            z.writestr(doc, build_documentation(cfg))
    log(f"packaged {path} (fill in Documentation_template.md before the final submission)")
    return path


# =============================================================================================================
# CLI
# =============================================================================================================
COMMANDS = ["download", "eda", "prep", "block", "feat", "l1", "prune", "l2", "decide", "cpu", "loco", "ce",
            "ce-export", "ce-worker", "ce-merge", "gpu-check", "synth", "smoke", "package"]


def main(argv=None) -> None:
    global CFG_ARGV
    ap = argparse.ArgumentParser(description="Business Entity Resolution -- Amazon ML Challenge 2026",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", default=None, help="optional YAML/JSON config merged over the defaults")
    ap.add_argument("--set", nargs="*", default=[], help="overrides: key.sub=value ...")
    ap.add_argument("--gpus", default=None, help="comma list of GPU ids (default from config)")
    ap.add_argument("--name", default=None, help="cross-encoder run name / preset (ce06, ce4b)")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--team", default="team")
    ap.add_argument("--repo", default=None, help="download: Hugging Face dataset repo id")
    ap.add_argument("--out", default=None, help="download: target folder (default .); synth: output folder")
    ap.add_argument("--n-entities", type=int, default=3000)
    ap.add_argument("--fast", action="store_true",
                    help="speed profile: smaller training sample, shorter LightGBM, one 1-epoch CE model")
    a = ap.parse_args(argv)
    CFG_ARGV = (["--config", a.config] if a.config else []) + (["--fast"] if a.fast else []) \
        + (["--set"] + list(a.set) if a.set else [])
    if a.command == "smoke":
        return stage_smoke(n_entities=a.n_entities)
    if a.command == "synth":
        return generate_synthetic(a.out or "dataset_synth", a.n_entities)
    cfg = load_config(a.config, a.set)
    if a.fast:  # profile over the config, then re-apply --set so explicit overrides still win
        cfg = _merge(cfg, FAST_PROFILE)
        for ov in a.set or []:
            key, val = ov.split("=", 1)
            node, parts = cfg, key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = _parse_val(val)
        log(f"--fast: sample_s1={cfg['sample_s1']}, ce={cfg['ce']['name']} "
            f"n_models={cfg['ce']['n_models']} gate={cfg['ce']['gate']}")
    if a.command == "download":
        return stage_download(cfg, a.repo, a.out or ".")
    os.makedirs(cfg["work_dir"], exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    ce_name = a.name or cfg["ce"]["name"]
    simple = {"eda": stage_eda, "prep": stage_prep, "block": stage_block, "feat": stage_feat, "l1": stage_l1,
              "prune": stage_prune, "l2": stage_l2, "decide": stage_decide, "cpu": stage_cpu, "loco": stage_loco,
              "ce-export": export_ce_inputs}
    if a.command in simple:
        simple[a.command](cfg)
    elif a.command == "ce":
        stage_ce(cfg, ce_name, gpu_list(a.gpus or cfg["ce"]["gpus"]),
                 [int(x) for x in a.folds.split(",")] if a.folds else None)
    elif a.command == "ce-worker":
        for f in [int(x) for x in a.folds.split(",") if x != ""]:
            ce_fold(cfg, ce_name, f)
    elif a.command == "ce-merge":
        ce_merge(cfg, ce_name)
    elif a.command == "gpu-check":
        stage_gpu_check(cfg, ce_name)
    elif a.command == "package":
        stage_package(cfg, a.team)


if __name__ == "__main__":
    main()
