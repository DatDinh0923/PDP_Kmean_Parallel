#!/usr/bin/env bash
set -uo pipefail

ITERS=${1:-40}
SEEDS="42 7"
IMAGES="data/test_512.png data/test_1024.png data/test_2048.png"
KS="4 8 16 32 64"

pass=0; fail=0; failed_cells=""

for img in $IMAGES; do
  for K in $KS; do
    for seed in $SEEDS; do
      ref=bin/vref
      ./bin/seq_O3 "$img" "$K" "$seed" "$ITERS" "$ref" --reps 1 --no-png >/dev/null || {
          echo "  ERROR running seq_O3 on $img K=$K seed=$seed"; fail=$((fail+1)); continue; }

      for backend in O0 O1 O2 v1 v2 v3 v4; do
        out=bin/vcmp
        case $backend in
          O*) ./bin/seq_$backend "$img" "$K" "$seed" "$ITERS" "$out" \
                  --reps 1 --no-png >/dev/null ;;
          v*) ./bin/kmeans_cuda "$img" "$K" "$seed" "$ITERS" "$out" \
                  --version "${backend#v}" --reps 1 --no-png >/dev/null ;;
        esac
        if ./bin/compare "$ref" "$out" >/dev/null 2>&1; then
          pass=$((pass+1))
        else
          fail=$((fail+1))
          cell="$(basename $img) K=$K seed=$seed $backend"
          failed_cells="$failed_cells\n    $cell"
          echo "  MISMATCH: $cell"
        fi
      done
    done
  done
  echo "  ...$(basename $img) done  (pass=$pass fail=$fail)"
done

echo
echo "======================================================"
echo "  verification matrix: $pass passed, $fail failed"
if [ "$fail" -ne 0 ]; then
  echo -e "  failing cells:$failed_cells"
  echo "======================================================"
  exit 1
fi
echo "  ALL BACKENDS BIT-IDENTICAL TO seq -O3"
echo "======================================================"
