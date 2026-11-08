# A/B dataset experiment on one Colab G4. Re-runnable: each stage skips when its output is on Drive.
#   A = existing 200M SFT model (run-200m-sft, step 1500, templated coding tasks)
#   B = same base (run-200m step 7600), same config (code-sft-200m.json, 1500 steps), same tokenizer,
#       same non-coding rows; only the 70,000 templated coding-task rows replaced by verified teacher rows
# Both are scored with identical suites: core, exam, HumanEval; each with the original decoding and plain
# decoding. Env: N (teacher requests, default 20000)
D=/content/drive/MyDrive/LocalBrain-200m
AB=$D/ab
N=${N:-20000}
log(){ echo "[$(date -u +%F' '%T)] $*" | tee -a $AB/ab.log; }
mkdir -p $AB/A $AB/B
set -e
if [ ! -d /content/lbcur ]; then
  mkdir -p /content/lbcur && cd /content/lbcur && unzip -q /content/repo-cur.zip && pip install -q -e . > /tmp/pip.log 2>&1
fi
cd /content/lbcur
command -v lua5.4 >/dev/null 2>&1 || (apt-get update -y && apt-get install -y lua5.4) > /tmp/lua.log 2>&1
mkdir -p /content/raw-hf/eval && cp /content/humaneval.parquet /content/mbpp-*.parquet /content/raw-hf/eval/
export LOCALBRAIN_SANDBOX=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "start N=$N gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"

evaluate(){  # $1 = A|B, $2 = checkpoint file, $3 = run dir (for tokenizer)
  for plain in "" "--plain"; do
    sfx=${plain:+-plain}
    [ -s $AB/$1/code-eval$sfx.json ] || python -m localbrain.cli code-eval --run $3 --checkpoint $2 --suite core \
      --samples 8 --device cuda $plain --output $AB/$1/code-eval$sfx.json > /content/eval-$1.out 2>&1
    [ -s $AB/$1/exam-eval$sfx.json ] || python -m localbrain.cli code-eval --run $3 --checkpoint $2 --suite exam \
      --device cuda $plain --output $AB/$1/exam-eval$sfx.json >> /content/eval-$1.out 2>&1
    [ -s $AB/$1/humaneval$sfx.json ] || python -m localbrain.cli humaneval --run $3 --checkpoint $2 \
      --data /content/raw-hf/eval/humaneval.parquet --device cuda $plain --output $AB/$1/humaneval$sfx.json >> /content/eval-$1.out 2>&1
    log "eval $1$sfx done"
  done
}

log "EVAL A (run-200m-sft latest.pt = step 1500)"
evaluate A $D/run-200m-sft/latest.pt $D/run-200m-sft

if [ ! -s $AB/distill2-raw.jsonl ]; then
  if ! /content/venv-vllm/bin/python -c 'import vllm' 2>/dev/null; then
    log "install vllm"
    rm -rf /content/venv-vllm && pip install -q uv > /tmp/uv.log 2>&1
    uv venv -q /content/venv-vllm --python 3 > /tmp/vllm-pip.log 2>&1
    uv pip install -q --python /content/venv-vllm/bin/python vllm hf_transfer ninja pyarrow >> /tmp/vllm-pip.log 2>&1
  fi
  log "generate start N=$N"
  PATH=/content/venv-vllm/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 /content/venv-vllm/bin/python scripts/distill_generate.py \
    --n $N --out /content/distill2-raw.jsonl > /content/gen.out 2>&1
  cp /content/distill2-raw.jsonl $AB/distill2-raw.jsonl
  log "generate done: $(wc -l < $AB/distill2-raw.jsonl)"
fi
if [ ! -s $AB/distill2-verified.jsonl ]; then
  python scripts/distill_verify.py --raw $AB/distill2-raw.jsonl --evalraw /content/raw-hf \
    --out /content/distill2-verified.jsonl > /content/verify.out 2>&1
  cp /content/distill2-verified.jsonl /content/distill2-verified.report.json $AB/
  log "verify: $(python -c "import json;r=json.load(open('$AB/distill2-verified.report.json'));print('items',r['kept_items'],'of',r['raw'],'rows',r['rows'])")"
fi
if [ ! -s $AB/sft-200m-B.jsonl ]; then
  python scripts/build_dataset_b.py --a $D/sft-200m.jsonl --teacher $AB/distill2-verified.jsonl \
    --out $AB/sft-200m-B.jsonl --tokenizer $D/run-200m/tokenizer.json > /content/buildb.out 2>&1
  log "B built: $(python -c "import json;r=json.load(open('$AB/sft-200m-B.report.json'));print({k:r[k] for k in ('A','B','repeat_per_teacher_row','coding_task_slots_after_repeat')})")"
  python scripts/contamination_report.py $D/sft-200m.jsonl $AB/sft-200m-B.jsonl --raw /content/raw-hf \
    --output $AB/contamination.json > /dev/null 2>&1
  log "contamination: $(python -c "import json;print([(r['rows'],r['hits']) for r in json.load(open('$AB/contamination.json'))])")"
fi
if [ ! -f $AB/dataset-B/meta.json ]; then
  python -m localbrain.cli prepare-instructions $AB/sft-200m-B.jsonl --output $AB/dataset-B.partial \
    --config configs/code-sft-200m.json --tokenizer $D/run-200m/tokenizer.json > /content/prep.out 2>&1
  rm -rf $AB/dataset-B && mv $AB/dataset-B.partial $AB/dataset-B
  log "dataset B: $(python -c "import json;print(json.load(open('$AB/dataset-B/meta.json'))['counts'])")"
fi
set +e
if ! grep -q '"step": 1500' $AB/run-B/metrics.jsonl 2>/dev/null; then
  R="--init $D/run-200m/latest.pt"; [ -f $AB/run-B/latest.pt ] && R="--resume"
  log "TRAIN B start $R (same config as A: configs/code-sft-200m.json)"
  python -m localbrain.cli train --data $AB/dataset-B --run $AB/run-B --config configs/code-sft-200m.json $R > /content/trainB.out 2>&1
  log "TRAIN-B-EXIT $? $(tail -n 1 $AB/run-B/metrics.jsonl)"
fi
log "EVAL B (run-B latest.pt)"
evaluate B $AB/run-B/latest.pt $AB/run-B
python scripts/ab_summary.py --dir $AB --train-a $D/sft-200m.jsonl --train-b $AB/sft-200m-B.jsonl \
  --humaneval /content/raw-hf/eval/humaneval.parquet > /content/summary.out 2>&1
log "SUMMARY $(python -c "import json;print(json.dumps(json.load(open('$AB/summary.json'))))")"
log "AB-DONE"
