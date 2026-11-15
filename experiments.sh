#!/usr/bin/env bash
# Train the requested runs and evaluate their saved checkpoints on validation images.
# Usage: DATA=data DEVICE=cuda bash experiments.sh r1 r2 r3
set -euo pipefail

DATA=${DATA:-data}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-200}
COMMON=(--data "$DATA" --device "$DEVICE" --epochs "$EPOCHS" --save-every 20 --width 64 --flip --ema 0.999)
DIFFAUG=(--diffaug color,translation,cutout)
mkdir -p output

run() {
  local name=$1
  shift
  python train.py "${COMMON[@]}" --output "output/$name" "$@" 2>&1 | tee "output/${name}_train.log"
  sweep "$name"
}

sweep() {
  local name=$1
  python evaluate.py --checkpoint output/"$name"/epoch_*.pt --split val --data "$DATA" \
    --device "$DEVICE" --output "output/val_$name"
  echo "Test-split command: python evaluate.py --checkpoint <lowest-FID checkpoint> --split test" \
       "--data $DATA --device $DEVICE --output output/test_$name"
}

runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then
  runs=(reference r1 r2 r3)
fi

for name in "${runs[@]}"; do
  case $name in
    reference) python reference.py --data "$DATA" --device "$DEVICE" --output output/reference ;;
    r0) sweep dcgan128 ;;  # baseline checkpoints from the original 100-epoch run
    r1) run r1 ;;
    r2) run r2 "${DIFFAUG[@]}" ;;
    r3) run r3 "${DIFFAUG[@]}" --upsample nearest --d-norm sn --loss hinge ;;
    *) echo "Unknown run: $name (expected reference, r0, r1, r2, r3)" >&2; exit 1 ;;
  esac
done
