import argparse
import os
import glob
import numpy as np
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="eval_results/RememberColor3-v0/")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--max-episodes", type=int, default=-1, help="max episodes to process (-1 for all)")
args = parser.parse_args()

if os.path.isdir(args.path):
    npz_files = sorted(glob.glob(os.path.join(args.path, "*.npz")),
                        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
else:
    npz_files = [args.path]

if args.max_episodes > 0:
    npz_files = npz_files[:args.max_episodes]

print(f"found {len(npz_files)} episodes")

for npz_path in npz_files:
    d = np.load(npz_path)
    for k, v in d.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    rgb = d["rgb"]
    frames = np.concatenate([rgb[..., :3], rgb[..., 3:]], axis=2)

    base = os.path.splitext(npz_path)[0]
    video_path = base + "_video.mp4"
    imageio.mimsave(video_path, frames, fps=args.fps)

    # save individual frames as pngs
    frames_dir = base + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(os.path.join(frames_dir, f"frame_{i:03d}.png"), frame)

    success = d["success"].any()
    reward_sum = d["reward"].sum()
    print(f"  {os.path.basename(npz_path)}: {len(frames)} frames, reward={reward_sum:.2f}, success={success} → {video_path}")

print("done")
