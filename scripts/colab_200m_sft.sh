# 200M SFT stage: tokenize the instruction mix, fine-tune the pretrained 200M base, then evaluate.
# Safe to re-run on a new machine: dataset + run live on Drive and every stage resumes.
#   source : /content/sft-200m.jsonl.gz (uploaded) -> $D/sft-200m.jsonl -> tokenized $D/dataset-sft-200m
#   train  : $D/run-200m-sft, --init from $D/run-200m, checkpoint every 250 steps
D=/content/drive/MyDrive/LocalBrain-200m
RUN=$D/run-200m-sft
SRC=$D/sft-200m.jsonl
DS=$D/dataset-sft-200m
BASE=$D/run-200m
log(){ echo "[$(date -u +%F' '%T)] $*" | tee -a $D/sft.log; }
set -e
cd /content/LocalBrain
[ -f $SRC ] || { log "decompress source"; gunzip -c /content/sft-200m.jsonl.gz > $SRC.partial && mv $SRC.partial $SRC; }
if [ ! -f $DS/meta.json ]; then
  log "prepare-instructions start (context 2048, reuse pretrain tokenizer)"
  python -m localbrain.cli prepare-instructions $SRC --output $DS.partial \
    --config configs/code-sft-200m.json --tokenizer $BASE/tokenizer.json > /content/prep.out 2>&1
  rm -rf $DS && mv $DS.partial $DS
  log "prepare-instructions done: $(python -c 'import json;print(json.load(open("'$DS'/meta.json"))["counts"])')"
fi
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=""; INIT="--init $BASE/latest.pt"
[ -f $RUN/latest.pt ] && { R="--resume"; INIT=""; }
log "SFT START resume='$R' gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m localbrain.cli train --data $DS --run $RUN --config configs/code-sft-200m.json $R $INIT > /content/sft.out 2>&1
log "SFT-EXIT $?"
