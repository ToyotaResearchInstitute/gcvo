#!/usr/bin/env python3
"""Convert an ETH3D RGB-D sequence into per-frame PCD files for gcvo_run_pcd.

For each (timestamp, rgb_path, depth_path) tuple in `associated.txt`:
  - Load RGB (uint8) and depth (uint16, scale=5000 -> meters)
  - Backproject depth pixels to 3D points using fx/fy/cx/cy
  - Sub-sample to N points (DSO-style edge sampling if --selection edges; else random)
  - Save as `<out_dir>/<timestamp>.pcd` (PointXYZRGB) so they sort by time

Also writes:
  - <out_dir>/timestamps.txt   one timestamp per line, matching the frame order
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from dso_pixel_selector import dso_select_pixels


def load_calib(path):
    """Read fx fy cx cy [depth_scale] from a CVO calib file."""
    with open(path) as f:
        nums = [float(x) for x in f.read().split()]
    fx, fy, cx, cy = nums[:4]
    depth_scale = nums[4] if len(nums) > 4 else 5000.0
    return fx, fy, cx, cy, depth_scale


def backproject(depth_m, fx, fy, cx, cy):
    """Return (xyz, valid_mask) where xyz is HxWx3 (float32)."""
    H, W = depth_m.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    valid = depth_m > 0
    x = (uu - cx) * depth_m / fx
    y = (vv - cy) * depth_m / fy
    z = depth_m
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    return xyz, valid


def select_dso_edges(rgb_u8, valid_mask, num_points):
    """DSO-style: sample points with strong image gradient. Falls back to random."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    score = np.sqrt(gx * gx + gy * gy)
    score = score * valid.astype(np.float32) if (valid := valid_mask) is not None else score
    flat = score.flatten()
    n_valid = int((flat > 0).sum())
    if n_valid == 0:
        return None
    k = min(num_points, n_valid)
    idx = np.argpartition(-flat, k - 1)[:k]
    return idx


def select_random(valid_mask, num_points, rng):
    """Random sample from valid pixels."""
    valid_idx = np.flatnonzero(valid_mask)
    if len(valid_idx) == 0:
        return None
    k = min(num_points, len(valid_idx))
    return rng.choice(valid_idx, size=k, replace=False)


def select_fps(xyz_3d, candidate_flat_idx, num_points, pre_subsample, rng):
    """Farthest-point sampling on backprojected 3D points (open3d C++ backend).

    xyz_3d: HxWx3 backprojected positions (float32)
    candidate_flat_idx: 1D indices into xyz_3d.reshape(-1, 3) of valid points
    num_points: target output count
    pre_subsample: random subsample to this many candidates first (cap runtime)
    rng: numpy RNG
    Returns: 1D flat indices of selected points (subset of candidate_flat_idx).
    """
    import open3d as o3d
    if len(candidate_flat_idx) == 0:
        return None
    if len(candidate_flat_idx) > pre_subsample:
        sub = rng.choice(candidate_flat_idx, size=pre_subsample, replace=False)
    else:
        sub = candidate_flat_idx
    pts = xyz_3d.reshape(-1, 3)[sub].astype(np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    k = min(num_points, len(pts))
    fps_pc = pc.farthest_point_down_sample(k)
    fps_pts = np.asarray(fps_pc.points)
    # Map back from FPS-output positions to candidate indices via KDTree lookup
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, local_idx = tree.query(fps_pts, k=1)
    return sub[local_idx]


def write_pcd_xyzrgb(path, xyz, rgb):
    """Write an ASCII PCD with PointXYZRGB. xyz: (N,3), rgb: (N,3) uint8."""
    n = xyz.shape[0]
    rgb_packed = (rgb[:, 0].astype(np.uint32) << 16
                  | rgb[:, 1].astype(np.uint32) << 8
                  | rgb[:, 2].astype(np.uint32))
    rgb_float = rgb_packed.view(np.float32)
    header = (
        f"VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {rgb_float[i]:.6e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="ETH3D sequence dir (must contain associated.txt + cvo_calib.txt)")
    ap.add_argument("--out_dir", required=True, help="Output directory for PCD files")
    ap.add_argument("--num_points", type=int, default=3000)
    ap.add_argument("--selection", choices=["edges", "random", "dso", "fps"], default="edges")
    ap.add_argument("--fps_pre_subsample", type=int, default=20000,
                    help="Random subsample to this many points before FPS (speedup)")
    ap.add_argument("--max_depth_m", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy, depth_scale = load_calib(os.path.join(args.dataset_dir, "cvo_calib.txt"))

    # Read associated.txt: each line "ts_rgb path_rgb ts_depth path_depth"
    pairs = []
    with open(os.path.join(args.dataset_dir, "associated.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts_rgb, p_rgb, ts_depth, p_depth = parts[0], parts[1], parts[2], parts[3]
            pairs.append((ts_rgb, p_rgb, p_depth))

    rng = np.random.default_rng(args.seed)
    timestamps = []
    for i, (ts, p_rgb, p_depth) in enumerate(pairs):
        rgb_path = os.path.join(args.dataset_dir, p_rgb)
        d_path = os.path.join(args.dataset_dir, p_depth)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)        # BGR uint8
        d_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)    # uint16

        if rgb is None or d_raw is None:
            print(f"[skip] missing {p_rgb} or {p_depth}")
            continue

        depth_m = d_raw.astype(np.float32) / depth_scale
        depth_m[depth_m > args.max_depth_m] = 0  # clip far returns

        xyz, valid = backproject(depth_m, fx, fy, cx, cy)

        if args.selection == "fps":
            valid_flat = np.flatnonzero(valid)
            idx = select_fps(xyz, valid_flat, args.num_points,
                             args.fps_pre_subsample, rng)
            if idx is None:
                idx = select_random(valid, args.num_points, rng)
        elif args.selection == "dso":
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            uv = dso_select_pixels(gray, args.num_points)
            if not uv:
                idx = select_random(valid, args.num_points, rng)
            else:
                # uv is list of (x, y); keep only those with valid depth.
                uv_arr = np.asarray(uv, dtype=np.int64)
                # idx into flattened HxW
                flat = uv_arr[:, 1] * depth_m.shape[1] + uv_arr[:, 0]
                valid_at = valid.reshape(-1)[flat]
                flat = flat[valid_at]
                idx = flat
                if len(idx) < args.num_points // 4:
                    idx = select_random(valid, args.num_points, rng)
        elif args.selection == "edges":
            idx = select_dso_edges(rgb, valid, args.num_points)
            if idx is None or len(idx) < args.num_points // 4:
                idx = select_random(valid, args.num_points, rng)
        else:
            idx = select_random(valid, args.num_points, rng)
        if idx is None:
            print(f"[skip] no valid points in frame {i}")
            continue

        H, W = depth_m.shape
        sel_xyz = xyz.reshape(-1, 3)[idx]
        sel_rgb_bgr = rgb.reshape(-1, 3)[idx]
        sel_rgb_rgb = sel_rgb_bgr[:, [2, 1, 0]]  # BGR -> RGB

        # Filename: 6-digit frame index so PCDs sort by frame order
        out_path = out_dir / f"{i:06d}.pcd"
        write_pcd_xyzrgb(str(out_path), sel_xyz, sel_rgb_rgb)
        timestamps.append(ts)

    with open(out_dir / "timestamps.txt", "w") as f:
        f.write("\n".join(timestamps) + "\n")

    print(f"Wrote {len(timestamps)} PCDs to {out_dir}")


if __name__ == "__main__":
    main()
