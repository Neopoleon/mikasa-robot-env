"""Convert MIKASA-Robo unbatched NPZ episodes into a zarr store for MujocoSingleTrajDataset.

Target zarr structure (per task):
    data.zarr/
        .zgroup
        .zattrs  (episode_frame_nums)
        episode_0/
            third_person_camera   (T, 256, 256, 3) uint8
            wrist_camera          (T, 256, 256, 3) uint8
            robot0_8d             (T, 8) float32  — 7 arm qpos + 1 gripper qpos
            action0_8d            (T, 8) float32  — 7 arm joint deltas + 1 gripper
        episode_1/
        ...

NPZ keys:
    rgb:     (T, 256, 256, 6)  — [third_person | wrist] concatenated on last axis
    joints:  (T, 25)           — [tcp_pose(7) | qpos(9) | qvel(9)]
    action:  (T, 8)            — 7 arm deltas + 1 gripper
    reward, success, done
"""
import argparse
import os
import numpy as np
import zarr
from tqdm import tqdm
from pathlib import Path


def convert_task(task_dir: str, output_path: str, max_episodes: int = -1):
    npz_files = sorted(Path(task_dir).glob("*.npz"))
    if max_episodes > 0:
        npz_files = npz_files[:max_episodes]
    print(f"Found {len(npz_files)} episodes in {task_dir}")

    store = zarr.open(output_path, mode="w", zarr_format=2)
    episode_frame_nums = {}

    for ep_idx, npz_path in enumerate(tqdm(npz_files)):
        d = np.load(npz_path)
        T = d["rgb"].shape[0]

        # Split RGB into two camera views
        third_person = d["rgb"][..., :3]   # (T, 256, 256, 3)
        wrist_cam = d["rgb"][..., 3:]      # (T, 256, 256, 3)

        # Extract proprio: 7 arm qpos (joints[7:14]) + 1 gripper qpos (joints[14])
        arm_qpos = d["joints"][:, 7:14]            # (T, 7)
        gripper_qpos = d["joints"][:, 14:15]        # (T, 1)
        robot0_8d = np.concatenate([arm_qpos, gripper_qpos], axis=1).astype(np.float32)  # (T, 8)

        # Action is already 8-dim (7 arm deltas + 1 gripper)
        action0_8d = d["action"].astype(np.float32)  # (T, 8)

        ep_group = store.create_group(f"episode_{ep_idx}")
        ep_group.create_array("third_person_camera", data=third_person.astype(np.uint8), chunks=(10, 256, 256, 3))
        ep_group.create_array("robot0_wrist_camera", data=wrist_cam.astype(np.uint8), chunks=(10, 256, 256, 3))
        ep_group.create_array("robot0_8d", data=robot0_8d, chunks=(T, 8))
        ep_group.create_array("action0_8d", data=action0_8d, chunks=(T, 8))

        # Store episode metadata
        ep_group.attrs["episode_length"] = T
        ep_group.attrs["is_successful"] = bool(d["success"].any())
        ep_group.attrs["final_reward"] = float(d["reward"][-1])

        episode_frame_nums[str(ep_idx)] = T

    store.attrs["episode_frame_nums"] = episode_frame_nums
    print(f"Saved {len(npz_files)} episodes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/MIKASA-Robo/unbatched",
                        help="Directory containing task subdirs with NPZ files")
    parser.add_argument("--output-dir", type=str, default="/home/jeff/imitation-learning-policies/mikasa_data",
                        help="Output directory for zarr stores")
    parser.add_argument("--tasks", nargs="+",
                        default=["RememberColor3-v0", "InterceptMedium-v0", "ShellGameTouch-v0"],
                        help="Task names to convert")
    parser.add_argument("--max-episodes", type=int, default=-1,
                        help="Max episodes to convert per task (-1 for all)")
    args = parser.parse_args()

    for task in args.tasks:
        task_dir = os.path.join(args.input_dir, task)
        if not os.path.isdir(task_dir):
            print(f"Skipping {task}: {task_dir} not found")
            continue
        output_path = os.path.join(args.output_dir, task, "episode_data.zarr")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nConverting {task}...")
        convert_task(task_dir, output_path, max_episodes=args.max_episodes)
