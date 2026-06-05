#!/usr/bin/env python3
"""Build a full per-sequence comparison table (t_rel + r_rel) for the writeup.

Columns:
  - GCVO winner (eigclamp + calib 0.205°)
  - GCVO no-calib baseline (centroid voxel, ff=1.0)
  - GCVO calib-only baseline (no regularization)
  - RKHS_BA cvo_aniso_gn
"""

import subprocess, os, pathlib, math

HOME = pathlib.Path.home()
EVAL = str(HOME / "code/docker_home/cvo/odometry_eval/KITTI/cpp/evaluate_odometry")
GT_DIR = str(HOME / "code/docker_home/cvo/RKHS_BA/ground_truth/kitti/cam0")
SCRIPT = str(HOME / "code/docker_home/cvo/gcvo/scripts/trajectory_change_basis.py")

SEQS = [f"{i:02d}" for i in range(11)]

GCVO_DIR = HOME / "code/docker_home/cvo/gcvo"
WINNER_DIR = GCVO_DIR / "results/2026-06-01/kitti_eigclamp_calib_winner"
RKHS_DIR = HOME / "code/docker_home/cvo/RKHS_BA/lidar_results"

# Variants: label -> {seq -> source kitti path}
VARIANTS = {
    "GCVO winner (eigclamp+calib)": {
        seq: str(WINNER_DIR / f"winner_{seq}.cam0.kitti") for seq in SEQS
    },
    "GCVO no-calib": {
        seq: str(GCVO_DIR / f"eval_centroid_ff10_{seq}.cam0.kitti") for seq in SEQS
    },
    "GCVO calib-only": {
        seq: str(GCVO_DIR / f"eval_final_{seq}.cam0.kitti") for seq in SEQS
    },
    "RKHS_BA cvo_aniso_gn": {
        seq: str(RKHS_DIR / f"{seq}.cvo_aniso_gn.kitti") for seq in SEQS
    },
}


def ensure_cam0(path, seq):
    """Convert lidar-frame to cam0 if path is in lidar_results/, else return as-is."""
    if not os.path.exists(path):
        return None
    if "lidar_results" in path:
        cam0 = path.replace(".kitti", ".cam0.kitti")
        if not os.path.exists(cam0):
            r = subprocess.run(["python3", SCRIPT, path, cam0, seq, "0", "-1"],
                               capture_output=True)
            if r.returncode != 0:
                return None
        return cam0
    return path


def eval_seq(traj, seq):
    if not traj or not os.path.exists(traj):
        return None, None
    r = subprocess.run([EVAL, f"{GT_DIR}/{seq}.txt", traj],
                       capture_output=True, text=True)
    t_rel = r_rel = None
    for line in r.stdout.splitlines():
        if "t_rel" in line:
            t_rel = float(line.split()[-1])
        elif "r_rel" in line:
            r_rel = float(line.split()[-1])
    return t_rel, r_rel


# results[variant][seq] = (t_rel, r_rel)
results = {label: {} for label in VARIANTS}
for label, paths in VARIANTS.items():
    for seq in SEQS:
        cam0 = ensure_cam0(paths[seq], seq)
        results[label][seq] = eval_seq(cam0, seq)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2: return m, 0.0
    s = math.sqrt(sum((x - m)**2 for x in vals) / (len(vals) - 1))
    return m, s


# Header
print("| seq | " + " | ".join(label for label in VARIANTS) + " |")
print("|" + "-----|" * (len(VARIANTS) + 1))

# t_rel table
print("\n## t_rel (%, lower = better)\n")
print("| seq | " + " | ".join(VARIANTS.keys()) + " |")
print("|" + "-----|" * (len(VARIANTS) + 1))
for seq in SEQS:
    row = [seq]
    for label in VARIANTS:
        t, _ = results[label][seq]
        row.append(f"{t:.3f}" if t is not None else "—")
    print("| " + " | ".join(row) + " |")
# Mean ± std row
row = ["mean ± std"]
for label in VARIANTS:
    vals = [results[label][s][0] for s in SEQS]
    m, s = mean_std(vals)
    row.append(f"**{m:.3f} ± {s:.3f}**" if m is not None else "—")
print("| " + " | ".join(row) + " |")

# r_rel table
print("\n## r_rel (deg/m, lower = better)\n")
print("| seq | " + " | ".join(VARIANTS.keys()) + " |")
print("|" + "-----|" * (len(VARIANTS) + 1))
for seq in SEQS:
    row = [seq]
    for label in VARIANTS:
        _, r = results[label][seq]
        row.append(f"{r:.5f}" if r is not None else "—")
    print("| " + " | ".join(row) + " |")
row = ["mean ± std"]
for label in VARIANTS:
    vals = [results[label][s][1] for s in SEQS]
    m, s = mean_std(vals)
    row.append(f"**{m:.5f} ± {s:.5f}**" if m is not None else "—")
print("| " + " | ".join(row) + " |")
