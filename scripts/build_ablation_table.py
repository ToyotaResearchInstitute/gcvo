#!/usr/bin/env python3
"""Build comparison table for hessian ablation."""
import subprocess, os, pathlib, math

HOME = pathlib.Path.home()
EVAL = str(HOME / "code/docker_home/cvo/odometry_eval/KITTI/cpp/evaluate_odometry")
GT_DIR = str(HOME / "code/docker_home/cvo/RKHS_BA/ground_truth/kitti/cam0")
SCRIPT = str(HOME / "code/docker_home/cvo/gcvo/scripts/trajectory_change_basis.py")

SEQS = [f"{i:02d}" for i in range(11)]

WIN_DIR = HOME / "code/docker_home/cvo/gcvo/results/2026-06-01/kitti_eigclamp_calib_winner"
ABL_DIR = HOME / "code/docker_home/cvo/gcvo/results/2026-06-01/hessian_ablation"
RKHS_DIR = HOME / "code/docker_home/cvo/RKHS_BA/lidar_results"

VARIANTS = [
    ("winner (H4+conn)",      lambda s: str(WIN_DIR / f"winner_{s}.cam0.kitti")),
    ("noconn (H4)",           lambda s: str(ABL_DIR / f"noconn_{s}.cam0.kitti")),
    ("h2h4 (H4+H2+conn)",     lambda s: str(ABL_DIR / f"h2h4_{s}.cam0.kitti")),
    ("h1h2h4 (H4+H1+H2+conn)",lambda s: str(ABL_DIR / f"h1h2h4_{s}.cam0.kitti")),
    ("RKHS_BA cvo_aniso_gn",  lambda s: str(RKHS_DIR / f"{s}.cvo_aniso_gn.cam0.kitti")),
]

# Ensure all cam0 files exist for ablation
def ensure_cam0(lidar_path, seq):
    cam0 = lidar_path.replace(".kitti", ".cam0.kitti").replace(".cam0.cam0", ".cam0")
    # If already cam0
    if lidar_path.endswith(".cam0.kitti"):
        return lidar_path
    if not os.path.exists(lidar_path):
        return None
    if not os.path.exists(cam0):
        subprocess.run(["python3", SCRIPT, lidar_path, cam0, seq, "0", "-1"], capture_output=True)
    return cam0 if os.path.exists(cam0) else None


def eval_traj(path, seq):
    if not path or not os.path.exists(path):
        return None, None
    r = subprocess.run([EVAL, f"{GT_DIR}/{seq}.txt", path], capture_output=True, text=True)
    t = r_ = None
    for line in r.stdout.splitlines():
        if "t_rel" in line: t = float(line.split()[-1])
        elif "r_rel" in line: r_ = float(line.split()[-1])
    return t, r_


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2: return m, 0.0
    s = math.sqrt(sum((x - m)**2 for x in vals) / (len(vals) - 1))
    return m, s


# Convert all ablation lidar->cam0 first
for var in ("noconn", "h2h4", "h1h2h4"):
    for seq in SEQS:
        lid = ABL_DIR / f"{var}_{seq}.kitti"
        cam = ABL_DIR / f"{var}_{seq}.cam0.kitti"
        if lid.exists() and not cam.exists():
            subprocess.run(["python3", SCRIPT, str(lid), str(cam), seq, "0", "-1"],
                           capture_output=True)
# RKHS_BA conversion
for seq in SEQS:
    lid = RKHS_DIR / f"{seq}.cvo_aniso_gn.kitti"
    cam = RKHS_DIR / f"{seq}.cvo_aniso_gn.cam0.kitti"
    if lid.exists() and not cam.exists():
        subprocess.run(["python3", SCRIPT, str(lid), str(cam), seq, "0", "-1"],
                       capture_output=True)

# Gather results
results = {label: {} for label, _ in VARIANTS}
for label, path_fn in VARIANTS:
    for seq in SEQS:
        results[label][seq] = eval_traj(path_fn(seq), seq)


def fmt(v, sig=3):
    if v is None: return "—"
    return f"{v:.{sig}f}"


labels = [lbl for lbl, _ in VARIANTS]

print("## t_rel (%, lower is better)\n")
print("| seq | " + " | ".join(labels) + " |")
print("|" + "----|" * (len(labels) + 1))
for seq in SEQS:
    row = [seq]
    for lbl in labels:
        t, _ = results[lbl][seq]
        row.append(fmt(t))
    print("| " + " | ".join(row) + " |")
row = ["**mean ± std**"]
for lbl in labels:
    m, s = mean_std([results[lbl][seq][0] for seq in SEQS])
    row.append(f"**{fmt(m)} ± {fmt(s)}**" if m is not None else "—")
print("| " + " | ".join(row) + " |")

print("\n## r_rel (deg/m, lower is better)\n")
print("| seq | " + " | ".join(labels) + " |")
print("|" + "----|" * (len(labels) + 1))
for seq in SEQS:
    row = [seq]
    for lbl in labels:
        _, r = results[lbl][seq]
        row.append(fmt(r, 5))
    print("| " + " | ".join(row) + " |")
row = ["**mean ± std**"]
for lbl in labels:
    m, s = mean_std([results[lbl][seq][1] for seq in SEQS])
    row.append(f"**{fmt(m, 5)} ± {fmt(s, 5)}**" if m is not None else "—")
print("| " + " | ".join(row) + " |")
