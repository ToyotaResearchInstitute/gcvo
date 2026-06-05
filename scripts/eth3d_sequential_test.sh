#!/bin/bash
# Sequential ETH3D tests — one variant×seq at a time, no parallel, no timeout.
# Reuses cached DSO PCDs.
#
# Usage: bash eth3d_sequential_test.sh <variant_label> <yaml>
#   e.g.   bash eth3d_sequential_test.sh h4_conn  gcvo_params/eth3d_h4_conn.yaml

set -e
LABEL=$1
YAML=$2
[[ -z "$LABEL" || -z "$YAML" ]] && { echo "usage: $0 <label> <yaml>"; exit 2; }

GCVO=~/code/docker_home/cvo/gcvo
BIN=$GCVO/build/gcvo_run_pcd
DATASET=~/code/docker_home/cvo/data/eth3d/training
PCD_BASE=$GCVO/results/2026-06-02/eth3d_dso_pcds
OUT=$GCVO/results/2026-06-04/eth3d_${LABEL}
mkdir -p $OUT/params
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

# Run each seq sequentially — no timeout, no parallel
for SEQ in sfm_bench plant_1 table_3 sfm_lab_room_1 planar_2; do
  PCD=$PCD_BASE/$SEQ
  TRAJ_K=$OUT/${SEQ}.kitti
  TRAJ_T=$OUT/${SEQ}.tum
  LOG=$OUT/${SEQ}.log
  echo "$(date '+%H:%M:%S') [run] $LABEL / $SEQ"
  $BIN --params $YAML --pcd_dir $PCD --type rgb --traj_file $TRAJ_K > $LOG 2>&1
  ktum $TRAJ_K $PCD/timestamps.txt $TRAJ_T
  GT=$DATASET/$SEQ/groundtruth.txt
  evo_rpe tum $GT $TRAJ_T --pose_relation full --delta 1 --delta_unit f --align_origin > ${LOG}.rpe 2>&1 || true
  evo_ape tum $GT $TRAJ_T --pose_relation full --align > ${LOG}.ape 2>&1 || true
  RPE=$(grep -m1 "rmse" ${LOG}.rpe 2>/dev/null | awk '{print $NF}')
  APE=$(grep -m1 "rmse" ${LOG}.ape 2>/dev/null | awk '{print $NF}')
  echo "  $SEQ: RPE=$RPE APE=$APE"
done

# Summary
SUM=$OUT/summary.txt
{
  printf "%-20s %12s %12s\n" "sequence" "RPE_RMSE" "APE_RMSE"
  echo "----------------------------------------------"
  for SEQ in sfm_bench plant_1 table_3 sfm_lab_room_1 planar_2; do
    LOG=$OUT/${SEQ}.log
    RPE=$(grep -m1 "rmse" ${LOG}.rpe 2>/dev/null | awk '{print $NF}')
    APE=$(grep -m1 "rmse" ${LOG}.ape 2>/dev/null | awk '{print $NF}')
    printf "%-20s %12s %12s\n" "$SEQ" "${RPE:-NA}" "${APE:-NA}"
  done
} | tee $SUM
