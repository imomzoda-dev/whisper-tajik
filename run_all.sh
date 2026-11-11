#!/bin/zsh
# Full pipeline: finish downloading -> train -> evaluate. Safe to re-run (everything resumes).
# Run:  caffeinate -i nohup ./run_all.sh > logs_pipeline.txt 2>&1 &
cd "$(dirname "$0")"
py() { .venv/bin/python -I "$@"; }
OUT=models/tajik-small-v2
export HF_XET_HIGH_PERFORMANCE=1
# cap the MPS allocator so training never pushes the Mac into swap (146 s/step -> 11 s/step)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5

# thermal guard: pauses training if GPU/CPU >= 95°C or macOS reports "serious" heat; resumes at 82°C
nohup .venv/bin/python -I scripts/thermal_guard.py > /dev/null 2>&1 &

echo "== $(date) waiting for running download"
while pgrep -f "prepare_data.py fetch" >/dev/null; do sleep 60; done

for i in 1 2 3; do            # retry interrupted downloads
  echo "== $(date) fetch pass $i"
  py scripts/prepare_data.py fetch --workers 4 && break
  sleep 60
done

echo "== $(date) train"
RESUME=""; [[ -d $OUT/last ]] && RESUME="--resume"
py scripts/train.py --model models/whisper-small-tajik-base --out $OUT \
    --train data/manifest_yt.tsv data/manifest_fleurs_train.tsv --dev data/manifest_fleurs_dev.tsv \
    --epochs 3 --bs 8 --accum 4 --grad-ckpt --workers 4 --save-every 300 $RESUME || exit 1

echo "== $(date) evaluate"
py scripts/evaluate.py $OUT/best --set all
echo "== $(date) ALL DONE"
