#!/bin/bash
# Preprocess ETH3D with FPS sampling (cached), then run H4+conn registration.

set -e
GCVO=~/code/docker_home/cvo/gcvo
DATASET=~/code/docker_home/cvo/data/eth3d/training
PREPROCESS=$GCVO/scripts/eth3d_preprocess.py
PCD_BASE=$GCVO/results/2026-06-04/eth3d_fps_pcds
OUT=$GCVO/results/2026-06-04/eth3d_fps_h4_conn
YAML=$GCVO/gcvo_params/eth3d_h4_conn.yaml
BIN=$GCVO/build/gcvo_run_pcd
mkdir -p $PCD_BASE $OUT/params
cp $YAML $OUT/params/best_config.yaml

ktum() { python3 - "$1" "$2" "$3" <<'PY'
import sys, numpy as np
traj, ts_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
poses = np.loadtxt(traj)
with open(ts_file) as f: ts = [l.strip() for l in f if l.strip()]
n = min(len(poses), len(ts))
lines = []
for t, row in zip(ts[:n], poses[:n]):
    T = np.eye(4); T[:3,:] = row.reshape(3,4); R = T[:3,:3]
    K = np.array([[R[0,0]-R[1,1]-R[2,2],0,0,0],[R[0,1]+R[1,0],R[1,1]-R[0,0]-R[2,2],0,0],
                  [R[0,2]+R[2,0],R[1,2]+R[2,1],R[2,2]-R[0,0]-R[1,1],0],
                  [R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1],R[0,0]+R[1,1]+R[2,2]]])/3.0
    ev, evec = np.linalg.eigh(K); q = evec[:, np.argmax(ev)]
    if q[3] < 0: q = -q
    lines.append(f"{t} {T[0,3]:.7f} {T[1,3]:.7f} {T[2,3]:.7f} {q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}")
with open(out, "w") as f: f.write("\n".join(lines) + "\n")
PY
}

for SEQ in sfm_bench plant_1 table_3 sfm_lab_room_1 planar_2; do
  SRC=$DATASET/$SEQ
  PCD=$PCD_BASE/$SEQ
  TRAJ_K=$OUT/${SEQ}.kitti
  TRAJ_T=$OUT/${SEQ}.tum
  LOG=$OUT/${SEQ}.log

  # 1. Preprocess (cached)
  if [[ ! -f $PCD/timestamps.txt ]]; then
    echo "$(date '+%H:%M:%S') [preprocess FPS] $SEQ"
    cp $SRC/associated.txt $SRC/assoc.txt 2>/dev/null || true
    cat $SRC/calibration.txt > $SRC/cvo_calib.txt
    echo " 5000.0 " >> $SRC/cvo_calib.txt
    python3 $PREPROCESS --dataset_dir $SRC --out_dir $PCD \
      --num_points 5000 --selection fps --fps_pre_subsample 20000 \
      > ${LOG}.preproc 2>&1
  fi

  # 2. Register
  echo "$(date '+%H:%M:%S') [register] $SEQ"
  $BIN --params $YAML --pcd_dir $PCD --type rgb --traj_file $TRAJ_K > $LOG 2>&1
  ktum $TRAJ_K $PCD/timestamps.txt $TRAJ_T

  # 3. Evaluate
  GT=$SRC/groundtruth.txt
  evo_rpe tum $GT $TRAJ_T --pose_relation full --delta 1 --delta_unit f --align_origin > ${LOG}.rpe 2>&1 || true
  evo_ape tum $GT $TRAJ_T --pose_relation full --align > ${LOG}.ape 2>&1 || true
  RPE=$(grep -m1 "rmse" ${LOG}.rpe 2>/dev/null | awk '{print $NF}')
  APE=$(grep -m1 "rmse" ${LOG}.ape 2>/dev/null | awk '{print $NF}')
  echo "  $SEQ: RPE=$RPE APE=$APE"
done

# Summary
{
  printf "%-20s %12s %12s\n" "sequence" "RPE_RMSE" "APE_RMSE"
  echo "----------------------------------------------"
  for SEQ in sfm_bench plant_1 table_3 sfm_lab_room_1 planar_2; do
    LOG=$OUT/${SEQ}.log
    RPE=$(grep -m1 "rmse" ${LOG}.rpe 2>/dev/null | awk '{print $NF}')
    APE=$(grep -m1 "rmse" ${LOG}.ape 2>/dev/null | awk '{print $NF}')
    printf "%-20s %12s %12s\n" "$SEQ" "${RPE:-NA}" "${APE:-NA}"
  done
} | tee $OUT/summary.txt
