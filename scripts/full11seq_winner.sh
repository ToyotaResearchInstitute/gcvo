#!/bin/bash
# Full 11-sequence run with the winning config: cf_B_eigclamp + calib=0.205°
# Eigenvalue clamping [0.01, 10.0] + ℓ²=0 should fix the cumulative drift across all seqs.

set -e

OUT=~/code/docker_home/cvo/gcvo/results/2026-06-01/winner_eigclamp_calib
mkdir -p "$OUT"

BIN=~/code/docker_home/cvo/gcvo/build/gcvo_kitti_f2f
KITTI=$HOME/code/docker_home/cvo/data/kitti/dataset
PARAMS=~/code/docker_home/cvo/gcvo/gcvo_params/cf_B_eigclamp.yaml

SEQS=(00 01 02 03 04 05 06 07 08 09 10)

# Run all in parallel (cap at 4 concurrent)
N=0
for SEQ in "${SEQS[@]}"; do
  out_kitti="$OUT/winner_${SEQ}.kitti"
  log_file="$OUT/winner_${SEQ}.log"

  echo "=== seq $SEQ (background) ==="
  $BIN \
    --params $PARAMS \
    --kitti_root $KITTI \
    --sequence $SEQ \
    --start 0 \
    --count 999999 \
    --voxel_mode centroid \
    --voxel_size 0.25 \
    --first_frame_l_init 1.0 \
    --kitti_vert_calib_deg 0.205 \
    --traj_file "$out_kitti" \
    > "$log_file" 2>&1 &

  N=$((N + 1))
  # Wait if too many jobs in parallel
  if [[ $((N % 4)) -eq 0 ]]; then
    wait
  fi
done

wait
echo "DONE all 11 seqs"
