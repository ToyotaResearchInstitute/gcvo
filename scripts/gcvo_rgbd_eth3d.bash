#!/bin/bash
# GCVO ETH3D RGB-D frame-to-frame tracking experiment.
# Mirrors RKHS_BA's scripts/cvo_intensity_rgbd_eth3d.bash, replacing the
# RKHS_BA loader+solver with: (1) Python preprocessor that converts each
# RGB+depth pair into a PCD, (2) gcvo_run_pcd for sequential registration,
# (3) trajectory conversion to TUM format, (4) evo evaluation.
#
# Reference: gcvo.pdf Section 4.3 / Table 3.
# Sequences: sfm_bench, plant_1, table_3, sfm_lab_room_1, planar_2.
#
# Usage:
#   bash scripts/gcvo_rgbd_eth3d.bash [out_dir]
# Default out_dir: results/<date>/gcvo_eth3d_rgbd/

set -e

DATE=$(date +%Y-%m-%d)
GCVO_ROOT=$HOME/code/docker_home/cvo/gcvo
OUT_DIR=${1:-$GCVO_ROOT/results/${DATE}/gcvo_eth3d_rgbd}
DATASET_DIR=$HOME/code/docker_home/cvo/data/eth3d/training
PARAMS=$GCVO_ROOT/gcvo_params/gcvo_eth3d_rgbd.yaml
BIN=$GCVO_ROOT/build/gcvo_run_pcd
PREPROCESS=$GCVO_ROOT/scripts/eth3d_preprocess.py
NUM_POINTS=5000   # matches paper 2nd-order setting (Table 3)

mkdir -p "$OUT_DIR/params"
cp "$PARAMS" "$OUT_DIR/params/best_config.yaml"

# 5 sequences from gcvo.pdf Table 3.
SEQS=(sfm_bench plant_1 table_3 sfm_lab_room_1 planar_2)

for SEQ in "${SEQS[@]}"; do
  echo ""
  echo "/********************** $SEQ *************************/"

  SRC=$DATASET_DIR/$SEQ
  PCD_DIR=$OUT_DIR/pcds_${SEQ}
  TRAJ_KITTI=$OUT_DIR/${SEQ}.kitti
  TRAJ_TUM=$OUT_DIR/${SEQ}.tum
  LOG=$OUT_DIR/${SEQ}.log

  # 1. Build cvo_calib.txt (calibration + depth_scale appended), matching RKHS_BA flow.
  cp "$SRC/associated.txt" "$SRC/assoc.txt" 2>/dev/null || true
  cat "$SRC/calibration.txt" > "$SRC/cvo_calib.txt"
  echo " 5000.0 " >> "$SRC/cvo_calib.txt"

  # 2. Preprocess: convert RGB+depth to per-frame PCDs (DSO-style edge sampling, 3000 pts).
  echo "[preprocess] $SEQ -> $PCD_DIR"
  python3 "$PREPROCESS" \
    --dataset_dir "$SRC" \
    --out_dir "$PCD_DIR" \
    --num_points $NUM_POINTS \
    --selection edges \
    > "${LOG}.preproc" 2>&1

  # 3. Run gcvo_run_pcd to sequentially register all consecutive pairs.
  echo "[register] $SEQ"
  $BIN \
    --params "$PARAMS" \
    --pcd_dir "$PCD_DIR" \
    --type rgb \
    --traj_file "$TRAJ_KITTI" \
    > "$LOG" 2>&1

  # 4. Convert KITTI-format trajectory (3x4 row-major per frame) to TUM format,
  #    pairing each pose with the RGB timestamp from associated.txt.
  echo "[tum]    $SEQ"
  python3 - <<PY
import numpy as np
poses = np.loadtxt("${TRAJ_KITTI}")
with open("${PCD_DIR}/timestamps.txt") as f:
    ts = [line.strip() for line in f if line.strip()]
# Trajectory has 1 pose per frame (identity first).
assert len(poses) == len(ts), f"poses={len(poses)} ts={len(ts)}"
out = []
for i, (t, row) in enumerate(zip(ts, poses)):
    T = np.eye(4); T[:3,:] = row.reshape(3,4)
    tx, ty, tz = T[:3, 3]
    R = T[:3, :3]
    # Rotation -> quaternion
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[0,1]+R[1,0], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[0,2]+R[2,0], R[1,2]+R[2,1], R[2,2]-R[0,0]-R[1,1], 0],
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    if q[3] < 0: q = -q
    qx, qy, qz, qw = q
    out.append(f"{t} {tx:.7f} {ty:.7f} {tz:.7f} {qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}")
with open("${TRAJ_TUM}", "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {len(out)} TUM poses")
PY

  # 5. Evaluate RPE (delta=1 frame) and APE vs ETH3D ground truth.
  GT=$SRC/groundtruth.txt
  echo "[evo]    $SEQ"
  evo_rpe tum "$GT" "$TRAJ_TUM" --pose_relation full --delta 1 --delta_unit f \
      --align_origin > "${LOG}.rpe" 2>&1 || true
  evo_ape tum "$GT" "$TRAJ_TUM" --pose_relation full \
      --align > "${LOG}.ape" 2>&1 || true

  RPE_RMSE=$(grep -m1 "rmse" "${LOG}.rpe" 2>/dev/null | awk '{print $NF}')
  APE_RMSE=$(grep -m1 "rmse" "${LOG}.ape" 2>/dev/null | awk '{print $NF}')
  echo "$SEQ  RPE_RMSE=${RPE_RMSE}  APE_RMSE=${APE_RMSE}"
done

# 6. Summary table.
echo ""
echo "=== Summary (RPE Δ=1, APE — see paper Table 3) ==="
printf "%-20s %12s %12s\n" "sequence" "RPE_RMSE" "APE_RMSE"
echo "----------------------------------------------"
for SEQ in "${SEQS[@]}"; do
  LOG=$OUT_DIR/${SEQ}.log
  RPE=$(grep -m1 "rmse" "${LOG}.rpe" 2>/dev/null | awk '{print $NF}')
  APE=$(grep -m1 "rmse" "${LOG}.ape" 2>/dev/null | awk '{print $NF}')
  printf "%-20s %12s %12s\n" "$SEQ" "${RPE:-NA}" "${APE:-NA}"
done | tee "$OUT_DIR/summary.txt"

echo ""
echo "Done. Artifacts: $OUT_DIR"
