#!/bin/bash
#SBATCH --job-name=ber
#SBATCH --output=ber-%j.out
#SBATCH --error=ber-%j.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=96:00:00
#SBATCH -A research
#SBATCH --qos=medium
#
# -w gnode075 is deliberately NOT set: that node showed 38/40 CPUs allocated, so pinning to it means
# queueing behind everything already on it. Add it back only if you need that specific machine.
#
# --mem=40G: the l2 stage now peaks around 10 GB (narrow context frame + one feature part at a time)
# instead of the ~28 GB it used to need, so a modest request is enough and schedules far sooner.
# Raise to 64G if `feat` logs that it is shrinking chunk_pairs more than you like.
#
# --cpus-per-task=12: 18 would wait for a nearly-full node. More cores mainly speed up `feat`; if the
# queue is empty, raising this is the single best way to cut wall time.

set -euo pipefail

# --- Environment setup ---
if [[ -z "${BER_SKIP_CONDA:-}" ]]; then
  source ~/.bashrc
  conda activate pyg
fi

# =====================================================================================================
# Paths. Everything heavy lives in scratch; only the small final artefacts are copied back to $HOME.
# =====================================================================================================
# SRC_DIR: the folder holding dataset/ and entity_resolution.py. The folder name has been spelled both
# "student_resource" and "student-resource" on this cluster, so probe the plausible spellings instead of
# hard-coding one: a wrong guess used to kill the job at the `cp` below, minutes after submission.
if [[ -n "${BER_SRC_DIR:-}" ]]; then
  SRC_DIR="$BER_SRC_DIR"
else
  SRC_DIR=""
  for c in "$HOME/amazonml/student-resource" "$HOME/amazonml/student_resource" \
           "$HOME/amazonml/student-resource/student_resource" "$PWD"; do
    [[ -f "$c/entity_resolution.py" ]] && { SRC_DIR="$c"; break; }
  done
  if [[ -z "$SRC_DIR" ]]; then
    echo "FATAL: could not find entity_resolution.py in any of the expected source folders." >&2
    echo "       Set BER_SRC_DIR=/path/to/folder and resubmit." >&2
    exit 2
  fi
fi
[[ -d "$SRC_DIR/dataset" ]] || { echo "FATAL: $SRC_DIR has no dataset/ subfolder" >&2; exit 2; }
echo "source dir  : $SRC_DIR"

SCRATCH="${BER_SCRATCH:-/scratch/$USER}"

# RUN_ID must be STABLE across resubmissions, otherwise the .done markers below land in a fresh directory
# every time and nothing resumes -- which silently threw away ~3.5 h of completed prep+block work each time
# the job was requeued. Stage artefacts are keyed by config, not by job, so sharing one run dir is correct.
# Set BER_FRESH=1 for a clean run, or BER_RUN_ID=<name> to keep separate experiments side by side.
RUN_ID="${BER_RUN_ID:-ber_main}"
RUN_DIR="$SCRATCH/$RUN_ID"
if [[ -n "${BER_FRESH:-}" ]]; then
  echo "BER_FRESH set - discarding $RUN_DIR and starting over"
  rm -rf "$RUN_DIR"
fi
DATA_DIR="$RUN_DIR/dataset"
WORK_DIR="$RUN_DIR/work"
OUT_DIR="$RUN_DIR/output"
LOG_DIR="$RUN_DIR/logs"
DONE_DIR="$RUN_DIR/.done"                                 # stage markers, so a resubmit resumes
KEEP_DIR="${BER_KEEP_DIR:-$HOME/amazonml/results}/${RUN_ID}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"  # per-job: never overwritten

mkdir -p "$DATA_DIR" "$WORK_DIR" "$OUT_DIR" "$LOG_DIR" "$DONE_DIR" "$KEEP_DIR"
echo "run directory: $RUN_DIR"

# =====================================================================================================
# Config passed to every stage. The GPU worker subprocesses inherit it automatically.
# =====================================================================================================
CFG=(--set
  "data_dir=$DATA_DIR"
  "work_dir=$WORK_DIR"
  "output_dir=$OUT_DIR"
  "n_jobs=${N_JOBS:-$((${SLURM_CPUS_PER_TASK:-2} - 1))}"
  # --- blocking recall ----------------------------------------------------------------------------------
  # The first real-data run measured recall=0.869, i.e. 13% of true matches never reached the models, which
  # hard-caps F0.5 no matter how good L1/CE/L2 are. Two separate causes, addressed separately:
  #
  # 1. k_rec was starved. recall_full_rec=0.839 at k_rec=3 BEAT recall_full_s1=0.704 at k_s1=15: the
  #    record->S1 direction is far more discriminative, because a record has ~one correct S1 entity while an
  #    S1 entity has ~4.7 correct records competing for its k slots. And both directions reuse the SAME
  #    sparse matmul -- k only controls how many results are kept per row -- so raising k_rec costs
  #    essentially nothing in blocking time, only more pairs downstream. This is the cheapest recall there is.
  # 2. Tokens were over-pruned. max_df=5000 drops any token in >0.1% of a 5M-row country group, and
  #    max_products then drops more of the survivors; a row whose every token is zeroed becomes all-zero and
  #    is unreachable in that view at any k. Loosening both costs blocking time roughly linearly, so the
  #    increase is moderate, and blocking.diagnose (below) now measures whether it was the binding cause.
  "blocking.k_rec.full=8" "blocking.k_rec.name=3" "blocking.k_rec.addr=3"
  "blocking.max_df=20000"
  "blocking.max_products=1.2e10"
  "blocking.diagnose=true"                                 # splits every miss into unreachable vs ranked-out
  "ce.n_models=2"                                          # one model per GPU, one wave
  "ce.micro=32" "ce.accum=4"                               # 11 GB Turing cards
  "ce.infer_bs=192"
)
CE_NAME="ce-gte"                                           # relevance-pretrained; ce-xlmr is the fallback
# RTX 2080 Ti: 11 GB and Turing (cc 7.5), so no bfloat16 -- the code already picks fp16 + GradScaler.
# micro=32 keeps training near ~5.4 GB, leaving headroom if the card is not completely empty.
PY="python -u $RUN_DIR/entity_resolution.py"

# =====================================================================================================
# Helpers
# =====================================================================================================
# Which config prefixes each stage's output actually depends on (including its upstream stages'). A .done
# marker stores the hash of just those settings, so changing e.g. blocking re-runs block/feat/l1/prune/l2 but
# leaves prep alone, and changing a ce knob does not throw away hours of blocking.
stage_deps () {
  case "$1" in
    prep)   echo "prep. seed" ;;
    eda)    echo "prep. seed" ;;
    block)  echo "prep. blocking. seed" ;;
    feat)   echo "prep. blocking. features. seed" ;;
    l1)     echo "prep. blocking. features. l1. lgb. sample_s1 cv. seed" ;;
    prune)  echo "prep. blocking. features. l1. lgb. sample_s1 cv. seed" ;;
    ce)     echo "prep. blocking. features. l1. lgb. sample_s1 cv. ce. seed" ;;
    l2)     echo "prep. blocking. features. l1. lgb. sample_s1 cv. ce. l2. seed" ;;
    *)      echo "ALL" ;;
  esac
}

stage_key () {                                             # stage_key <name> -> hash of the relevant config
  local name="$1" deps line="" p
  deps="$(stage_deps "$name")"
  for p in "${CFG[@]}"; do
    [[ "$p" == "--set" ]] && continue
    if [[ "$deps" == "ALL" ]]; then
      line+="$p;"
    else
      for d in $deps; do
        [[ "$p" == "$d"* ]] && { line+="$p;"; break; }
      done
    fi
  done
  printf '%s|%s' "$name" "$line" | md5sum | cut -d' ' -f1
}

stage () {                                                 # stage <name> <args...>
  local name="$1"; shift
  local key; key="$(stage_key "$name")"
  if [[ -f "$DONE_DIR/$name" ]]; then
    if [[ "$(cat "$DONE_DIR/$name")" == "$key" ]]; then
      echo "[$(date +%H:%M:%S)] SKIP  $name (already done; delete $DONE_DIR/$name to force)"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] STALE $name (config it depends on changed) - re-running"
    rm -f "$DONE_DIR/$name"
  fi
  echo "============================================================================"
  echo "[$(date +%H:%M:%S)] START $name"
  echo "============================================================================"
  local t0 rc
  t0=$SECONDS
  set +e
  $PY "$@" "${CFG[@]}" 2>&1 | tee "$LOG_DIR/$name.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] FAILED $name after $(( (SECONDS-t0)/60 )) min (exit $rc). Log: $LOG_DIR/$name.log"
    exit $rc
  fi
  echo "$key" > "$DONE_DIR/$name"
  echo "[$(date +%H:%M:%S)] DONE  $name in $(( (SECONDS-t0)/60 )) min"
}

banner () { echo; echo "### $* ###"; echo; }

# =====================================================================================================
# 0. Diagnostics and dependencies
# =====================================================================================================
banner "environment"
echo "host        : $(hostname)"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-?}  (n_jobs=${N_JOBS:-$((${SLURM_CPUS_PER_TASK:-2} - 1))})"
echo "mem         : ${SLURM_MEM_PER_NODE:-?} MB"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python -c "import sys; print('python', sys.version.split()[0])"

banner "dependencies"
python - <<'PYCHECK' || { echo "installing missing packages into the user site"; \
    pip install --user -q numpy pandas pyarrow scipy scikit-learn lightgbm rapidfuzz \
                          sparse_dot_topn pyyaml tqdm transformers sentencepiece accelerate; }
import importlib.util, sys
need = ["numpy","pandas","pyarrow","scipy","sklearn","lightgbm","rapidfuzz",
        "sparse_dot_topn","yaml","tqdm","torch","transformers"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
print("missing:", missing or "none")
sys.exit(1 if missing else 0)
PYCHECK
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
                               'devices', torch.cuda.device_count())"

# =====================================================================================================
# 1. Stage the code and data into scratch (node-local disk is much faster than NFS for the pair store)
# =====================================================================================================
banner "staging code and data into scratch"
cp "$SRC_DIR/entity_resolution.py" "$RUN_DIR/"
[[ -d "$SRC_DIR/utils" ]] && cp -r "$SRC_DIR/utils" "$RUN_DIR/" || true
if command -v rsync >/dev/null; then
  rsync -a --info=progress2 "$SRC_DIR/dataset/" "$DATA_DIR/"
else
  cp -r "$SRC_DIR/dataset/." "$DATA_DIR/"
fi
du -sh "$DATA_DIR"
ls -la "$DATA_DIR/train" "$DATA_DIR/test"

# =====================================================================================================
# 2. Sanity check: full pipeline on synthetic data, ~2 minutes. Fails fast if the env is wrong.
# =====================================================================================================
if [[ ! -f "$DONE_DIR/smoke" ]]; then
  banner "smoke test (synthetic, ~2 min)"
  cd "$RUN_DIR"
  python -u entity_resolution.py smoke --n-entities 3000 2>&1 | tee "$LOG_DIR/smoke.log"
  grep -q "checks: PASS" "$LOG_DIR/smoke.log" || { echo "SMOKE FAILED - stopping before the real run"; exit 1; }
  rm -rf "$RUN_DIR/smoke_run"
  touch "$DONE_DIR/smoke"
fi

# =====================================================================================================
# 3. CPU pipeline
# =====================================================================================================
cd "$RUN_DIR"
stage prep  prep
stage eda   eda
stage block block

banner "BLOCKING RECALL - this caps everything downstream"
grep -E '"recall"|recall_country' "$LOG_DIR/block.log" | tail -5 || true

# The k_rec increase above roughly doubles the candidate count (measured 1.79x on synthetic data), and feat
# writes ~90 float32 features per pair, so this is where scratch fills up if it is going to. Checking costs
# nothing and beats dying several hours into the stage.
avail_gb=$(df -BG --output=avail "$WORK_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "scratch free: ${avail_gb:-?} GB"
if [[ -n "${avail_gb:-}" && "$avail_gb" -gt 0 && "$avail_gb" -lt 150 ]]; then
  echo "WARNING: only ${avail_gb} GB free on scratch. feat needs roughly 100-150 GB at the current pair"
  echo "         volume. Free space, or lower blocking.k_rec.* in CFG, before this stage gets far."
fi

stage feat  feat
stage l1    l1
stage prune prune

banner "ORACLE F0.5 - the best score reachable from this candidate set"
grep "prune train" "$LOG_DIR/prune.log" | tail -1 || true

# =====================================================================================================
# 4. Cross-encoder on both GPUs
# =====================================================================================================
if [[ -n "${BER_SKIP_CE:-}" ]]; then
  # CPU-only fallback: L2 stacks on the L1 score alone. Weaker, but a complete, valid submission --
  # the right move if the GPUs are unavailable or the queue is about to expire.
  echo "BER_SKIP_CE set - skipping the cross-encoder, L2 will stack on L1 only"
else
  banner "GPU check - measured pairs/s and gate coverage on the real data"
  set +e
  $PY gpu-check --name "$CE_NAME" "${CFG[@]}" 2>&1 | tee "$LOG_DIR/gpu-check.log"
  gpu_rc=${PIPESTATUS[0]}
  set -e
  if [[ $gpu_rc -ne 0 ]]; then
    echo "gpu-check FAILED (exit $gpu_rc) - not starting the cross-encoder."
    echo "Re-submit with BER_SKIP_CE=1 to finish CPU-only, or fix the GPU / model download first."
    exit $gpu_rc
  fi
  stage ce ce --name "$CE_NAME" --gpus "${BER_GPUS:-0,1}"
fi

# =====================================================================================================
# 5. Stack, decide, validate, package
# =====================================================================================================
stage l2     l2
stage decide decide
stage loco   loco

banner "validating submission"
if [[ -f "$RUN_DIR/utils/validate_submission.py" ]]; then
  python "$RUN_DIR/utils/validate_submission.py" \
      --matching  "$OUT_DIR/matching_results.tsv" \
      --candidate "$OUT_DIR/candidate_pairs.tsv" \
      --test-dir  "$DATA_DIR/test" 2>&1 | tee "$LOG_DIR/validate.log"
else
  echo "utils/validate_submission.py not found - relying on the built-in checks"
fi

stage package package --team YourTeamName

# =====================================================================================================
# 6. Copy the small artefacts off scratch
# =====================================================================================================
banner "copying results to $KEEP_DIR"
cp -v "$OUT_DIR"/*.tsv                          "$KEEP_DIR/" 2>/dev/null || true
cp -v "$RUN_DIR"/*_submission.zip               "$KEEP_DIR/" 2>/dev/null || true
cp -v "$WORK_DIR/experiments.jsonl"             "$KEEP_DIR/" 2>/dev/null || true
cp -v "$WORK_DIR/eda.json"                      "$KEEP_DIR/" 2>/dev/null || true
cp -v "$WORK_DIR"/train/decision*.json          "$KEEP_DIR/" 2>/dev/null || true
cp -rv "$LOG_DIR"                               "$KEEP_DIR/" 2>/dev/null || true
cp -v "$RUN_DIR/entity_resolution.py"                "$KEEP_DIR/" 2>/dev/null || true

banner "SUMMARY"
echo "total wall time : $(( SECONDS/3600 ))h $(( (SECONDS%3600)/60 ))m"
echo "scratch run dir : $RUN_DIR   (volatile - do not rely on it)"
echo "kept results    : $KEEP_DIR"
echo
echo "final score estimate:"
grep -h "cross-fitted F0.5" "$LOG_DIR/decide.log" 2>/dev/null | tail -2 || true
grep -h "cross-fitted F0.5" "$LOG_DIR/loco.log"   2>/dev/null | tail -1 || true
echo
du -sh "$WORK_DIR" || true
