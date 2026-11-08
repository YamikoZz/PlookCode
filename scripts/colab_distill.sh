# Distillation round (phase 1 pilot) on one Colab GPU (G4 96 GB). Re-runnable: every stage skips if its
# output already exists on Drive.
#   teacher : Qwen/Qwen2.5-Coder-32B-Instruct (Apache-2.0) in its own venv; only its outputs are kept
#   verify  : run every reference solution on its tests (sandboxed runner), drop eval overlap / near-dups
#   train   : continue the 200M SFT model on verified rows + replay of the old SFT mix
#   eval    : core + exam, same suites as before
# Env: N (exercises to request, default 6000), REPLAY (old SFT rows mixed in, default 12000), TAG (default distill1)
D=/content/drive/MyDrive/LocalBrain-200m
TAG=${TAG:-distill1}
N=${N:-6000}
REPLAY=${REPLAY:-12000}
RAW=$D/$TAG-raw.jsonl
VER=$D/$TAG-verified.jsonl
MIX=$D/$TAG-mix.jsonl
DS=$D/dataset-$TAG
RUN=$D/run-200m-$TAG
log(){ echo "[$(date -u +%F' '%T)] $*" | tee -a $D/$TAG.log; }
set -e
if [ ! -d /content/lbcur ]; then
  mkdir -p /content/lbcur && cd /content/lbcur && unzip -q /content/repo-cur.zip && pip install -q -e . > /tmp/pip.log 2>&1
fi
cd /content/lbcur
command -v lua5.4 >/dev/null 2>&1 || (apt-get update -y && apt-get install -y lua5.4) > /tmp/lua.log 2>&1
log "start TAG=$TAG N=$N gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) disk=$(df -h /content | tail -1 | awk '{print $4}') free"
if [ ! -s $RAW ]; then
  if ! /content/venv-vllm/bin/python -c 'import vllm' 2>/dev/null; then
    log "install vllm (separate venv so training torch is untouched)"
    rm -rf /content/venv-vllm
    pip install -q uv > /tmp/uv.log 2>&1
    uv venv -q /content/venv-vllm --python 3 > /tmp/vllm-pip.log 2>&1
    uv pip install -q --python /content/venv-vllm/bin/python vllm hf_transfer >> /tmp/vllm-pip.log 2>&1
    log "vllm $(/content/venv-vllm/bin/python -c 'import vllm;print(vllm.__version__)')"
  fi
  /content/venv-vllm/bin/python - <<'PY' > /tmp/license.txt 2>&1 || true
import json, urllib.request
info = json.load(urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-32B-Instruct"))
print("license:", info.get("cardData", {}).get("license"), "| tags:", [t for t in info.get("tags", []) if t.startswith("license")])
PY
  log "teacher $(cat /tmp/license.txt)"
  uv pip install -q --python /content/venv-vllm/bin/python ninja >> /tmp/vllm-pip.log 2>&1
  log "generate start (download ~65 GB + generation)"
  PATH=/content/venv-vllm/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 /content/venv-vllm/bin/python scripts/distill_generate.py --n $N --out /content/$TAG-raw.jsonl > /content/gen.out 2>&1
  cp /content/$TAG-raw.jsonl $RAW.partial && mv $RAW.partial $RAW
  log "generate done: $(wc -l < $RAW) rows"
fi
if [ ! -s $VER ]; then
  mkdir -p /content/raw-hf/eval && cp /content/humaneval.parquet /content/mbpp-*.parquet /content/raw-hf/eval/
  python scripts/distill_verify.py --raw $RAW --evalraw /content/raw-hf --out /content/$TAG-verified.jsonl > /content/verify.out 2>&1
  cp /content/$TAG-verified.jsonl $VER && cp /content/$TAG-verified.report.json $D/$TAG-verified.report.json
  log "verify done: $(python -c "import json;r=json.load(open('$D/$TAG-verified.report.json'));print('kept',r['kept'],'of',r['raw'],r['kept_by_language'])")"
fi
if [ ! -f $DS/meta.json ]; then
  python - <<PY
import json, random
random.seed(7)
rows = [json.loads(l) for l in open("$VER", encoding="utf-8") if l.strip()]
old = [l for l in open("$D/sft-200m.jsonl", encoding="utf-8") if l.strip()]
random.shuffle(old)
with open("$MIX", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
    for l in old[:$REPLAY]:
        f.write(l)
print("mix", len(rows), "+", min($REPLAY, len(old)))
PY
  python -m localbrain.cli prepare-instructions $MIX --output $DS.partial --config configs/distill-200m.json \
    --tokenizer $D/run-200m/tokenizer.json > /content/prep.out 2>&1
  rm -rf $DS && mv $DS.partial $DS
  log "dataset: $(python -c "import json;print(json.load(open('$DS/meta.json'))['counts'])")"
fi
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -f $RUN/best.pt ] || ! grep -q '"step": 400' $RUN/metrics.jsonl 2>/dev/null; then
  R="--init $D/run-200m-sft/best.pt"; [ -f $RUN/latest.pt ] && R="--resume"
  log "TRAIN start $R"
  python -m localbrain.cli train --data $DS --run $RUN --config configs/distill-200m.json $R > /content/train.out 2>&1
  log "TRAIN-EXIT $? $(tail -n 1 $RUN/metrics.jsonl)"
fi
# best.pt is chosen by val loss on the replay-heavy val split and can be step 1 (= the old model);
# always evaluate the finished run (latest.pt) from a separate folder.
EV=$RUN-final; mkdir -p $EV && cp $RUN/latest.pt $RUN/tokenizer.json $RUN/config.json $EV/ && RUN=$EV
log "EVAL core (latest.pt, $(python -c "import torch;print('step',torch.load('$RUN/latest.pt',map_location='cpu',weights_only=True)['step'])"))"
python -m localbrain.cli code-eval --run $RUN --samples 8 --suite core --device cuda > /content/eval_core.out 2>&1
log "CORE $(python -c "import json;d=json.load(open('$RUN/code-eval.json'));print({k:(round(v['pass@1_greedy']*v['tasks']),v['tasks']) for k,v in d.items() if isinstance(v,dict) and 'tasks' in v})")"
python -m localbrain.cli code-eval --run $RUN --suite exam --device cuda > /content/eval_exam.out 2>&1
log "EXAM $(python -c "import json;d=json.load(open('$RUN/exam-eval.json'));print({k:(round(v['pass@1_greedy']*v['tasks']),v['tasks']) for k,v in d.items() if isinstance(v,dict) and 'tasks' in v})")"
log "ALL-DONE"
