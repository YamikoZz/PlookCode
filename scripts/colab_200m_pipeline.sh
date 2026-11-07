# 200M pipeline for one Colab L4 session. Safe to re-run on a new machine: every stage resumes from Drive.
#   corpus: build-corpus --checkpoint (progress appended to Drive every 60 min)
#   train : run dir on Drive, checkpoint every 150 steps (~60 min at 23.6 s/step)
D=/content/drive/MyDrive/LocalBrain-200m
RUN=$D/run-200m
log(){ echo "[$(date -u +%F' '%T)] $*" | tee -a $D/pipeline.log; }
set -e
rm -rf /content/LocalBrain && mkdir -p /content/LocalBrain && cd /content/LocalBrain && unzip -q /content/repo.zip
pip install -q -e . > /tmp/pip.log 2>&1
log "setup ok"
if [ ! -f $D/dataset-code-200m/meta.json ]; then
  mkdir -p /content/raw && cd /content/raw
  C=https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/main/data
  F=https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT
  for i in $(seq 0 109); do n=$(printf %05d $i); echo "$C/train-$n-of-00880.parquet code-$n.parquet"; done > list.txt
  for i in $(seq 0 7); do n=$(printf %03d $i); echo "$F/${n}_00000.parquet fineweb-$n.parquet"; done >> list.txt
  log "download start"
  cat list.txt | xargs -P 8 -n 2 sh -c '[ -s "$1" ] || curl -sSL --retry 5 -o "$1" "$0"'
  log "download done $(du -sh /content/raw | cut -f1)"
  cd /content/LocalBrain
  rm -rf /content/dataset-build
  log "corpus build start (checkpoint every 60 min)"
  python -m localbrain.cli build-corpus --code /content/raw/code-*.parquet --english /content/raw/fineweb-*.parquet \
    --output /content/dataset-build --max-tokens 4000000000 --tokenizer /content/tokenizer.json \
    --checkpoint $D/corpus-ckpt --checkpoint-minutes 60 > /content/corpus.out 2>&1
  log "corpus build done; copying to Drive"
  rm -rf $D/dataset-code-200m.partial
  cp -r /content/dataset-build $D/dataset-code-200m.partial
  mv $D/dataset-code-200m.partial $D/dataset-code-200m
  mv /content/dataset-build /content/data
  log "CORPUS-ON-DRIVE $(cat $D/dataset-code-200m/meta.json | python -c 'import json,sys;print(json.load(sys.stdin)["tokens"])')"
fi
if [ ! -f /content/data/meta.json ]; then
  log "copy corpus from Drive"
  rm -rf /content/data.partial && mkdir -p /content/data.partial
  cp $D/dataset-code-200m/* /content/data.partial/ && mv /content/data.partial /content/data
fi
cd /content/LocalBrain
python - <<'PY'
import json
c = json.load(open("configs/code-pretrain-200m.json"))
c["save_every"] = 150
json.dump(c, open("/content/pretrain-200m.json", "w"), indent=2)
PY
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=""; [ -f $RUN/latest.pt ] && R="--resume"
log "PRETRAIN START resume='$R'"
python -m localbrain.cli train --data /content/data --run $RUN --config /content/pretrain-200m.json $R > /content/train.out 2>&1
log "PRETRAIN-EXIT $?"
