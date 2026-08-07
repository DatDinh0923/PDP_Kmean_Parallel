#!/usr/bin/env bash
# bench.sh -- performance sweep across resolution, K and backend.
#
# Emits results.csv. All backends do IDENTICAL work for a given cell (same
# seeding, same iteration count), so the timings are directly comparable.
#
# Usage: ./bench.sh [max_iter]
set -uo pipefail

ITERS=${1:-30}
SEED=42
OUT=results.csv
IMAGES="data/test_512.png data/test_1024.png data/test_2048.png data/test_4096.png"
KS="4 8 16 32 64"

echo "backend,tag,image,w,h,n,K,seed,iters,load_ms,seed_ms,lloyd_ms,write_ms,h2d_ms,d2h_ms" > $OUT

echo "== optimisation-level sweep (1024x1024, K=16) =="
for O in O0 O1 O2 O3; do
  ./bin/seq_$O data/test_1024.png 16 $SEED $ITERS bin/b --reps 3 --no-png \
    | grep '^CSV' | cut -d, -f2- >> $OUT
  echo "   seq -$O done"
done

echo "== main sweep =="
for img in $IMAGES; do
  [ -f "$img" ] || { echo "   skipping missing $img"; continue; }
  for K in $KS; do
    ./bin/seq_O3 "$img" "$K" $SEED $ITERS bin/b --reps 1 --no-png \
      | grep '^CSV' | cut -d, -f2- >> $OUT
    for v in 1 2 3 4; do
      ./bin/kmeans_cuda "$img" "$K" $SEED $ITERS bin/b --version $v --reps 3 --no-png \
        | grep '^CSV' | cut -d, -f2- >> $OUT
    done
    echo "   $(basename $img) K=$K done"
  done
done

echo "== large-K sweep (1024x1024) -- exposes the constant-cache limit =="
for K in 128 256 512 1024; do
  ./bin/seq_O3 data/test_1024.png "$K" $SEED $ITERS bin/b --reps 1 --no-png \
    | grep '^CSV' | cut -d, -f2- >> $OUT
  for v in 1 2 3 4; do
    ./bin/kmeans_cuda data/test_1024.png "$K" $SEED $ITERS bin/b --version $v --reps 3 --no-png \
      | grep '^CSV' | cut -d, -f2- >> $OUT
  done
  echo "   1024 K=$K done"
done

echo
echo "wrote $OUT ($(($(wc -l < $OUT) - 1)) rows)"
