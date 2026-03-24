"""evaluate imitation learning policies on MIKASA-Robo environments.

modes:
  eval:   python mikasa_eval.py --env-id ShellGameTouch-v0 --checkpoint path/to/ckpt --num-envs 16
  replay: python mikasa_eval.py --env-id RememberColor3-v0 --checkpoint path/to/ckpt \
              --replay-data path/to/episode_data.zarr --replay-episodes 0 1 5
          feeds expert obs to policy, overlays expert (green) vs policy (red) actions for debugging.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import zarr
from tqdm import tqdm

from eval_utils import (
    TASK_REGISTRY,
    ACTION_HORIZON,
    load_policy,
    make_eval_env,
    predict_action,
    EpisodeBuffers,
    handle_episode_completions,
    _divergence_stats,
    _annotate_replay_obs_frames,
    _save_video_and_frames,
)


# eval loop

def run_eval(env, inner_env, policy, normalizer, num_envs, num_eval_steps, device, seed,
             output_dir, env_id):
    """predict → execute ACTION_HORIZON steps → repeat. save episodes on completion.
    output: <output_dir>/<env_id>/episode_NNNN_{success|failure}/ + summary.json
    """
    task_dir = os.path.join(output_dir, env_id)
    os.makedirs(task_dir, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    buffers = EpisodeBuffers(num_envs)
    metrics = defaultdict(list)
    num_episodes = 0
    step = 0
    pbar = tqdm(total=num_eval_steps, desc="eval")

    while step < num_eval_steps:
        action_chunk = predict_action(obs, policy, normalizer, num_envs, device)

        for t in range(ACTION_HORIZON):
            if step >= num_eval_steps:
                break

            render_frame = inner_env.render()  # (B,H,W,3) with wrapper overlays
            buffers.append_obs(obs, render_frame=render_frame)
            action = action_chunk[:, t, :]
            buffers.append_action(action)

            obs, reward, _term, _trunc, infos = env.step(action)
            buffers.append_reward(reward)
            step += 1
            pbar.update(1)

            if "final_info" in infos:
                num_episodes = handle_episode_completions(
                    infos, buffers, num_envs, num_episodes, task_dir, metrics,
                )
                policy.reset()  # clear stale memory history
                obs, _ = env.reset(seed=seed)
                break           # discard remaining chunk actions

    pbar.close()

    results = {"num_episodes": num_episodes}
    for k, v in metrics.items():
        results[k] = torch.stack(v).float().mean().item()

    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {num_episodes} episodes to {task_dir}/")

    return results


# replay comparison — feed expert obs from zarr to policy, compare predicted vs expert actions.
# no env stepping — uses the exact observations the expert saw.

def _zarr_obs_to_policy_input(ep, t, device):
    """construct policy input from zarr episode at timestep t. no env needed.
    zarr keys: third_person_camera (T,256,256,3), robot0_wrist_camera (T,256,256,3), robot0_8d (T,8).
    """
    tp = torch.from_numpy(ep["third_person_camera"][t]).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
    wc = torch.from_numpy(ep["robot0_wrist_camera"][t]).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
    proprio = torch.from_numpy(ep["robot0_8d"][t]).float().unsqueeze(0).unsqueeze(1)
    return {
        "third_person_camera": tp.to(device),
        "robot0_wrist_camera": wc.to(device),
        "robot0_8d": proprio.to(device),
        "episode_idx": torch.zeros(1, device=device, dtype=torch.long),
    }


def run_replay(policy, normalizer, device, replay_data_path,
               replay_episodes, output_dir, env_id, fps=10):
    """replay expert episodes offline: feed expert obs to policy, compare predicted vs expert actions.
    no env needed — loads obs directly from zarr (the exact frames the expert saw).
    zarr structure: episode_N/{action0_8d, third_person_camera, robot0_wrist_camera, robot0_8d}.
    output per episode: obs video/frames with expert (green) vs policy (red) action overlays + metrics.
    """
    task_dir = os.path.join(output_dir, env_id, "replay")
    os.makedirs(task_dir, exist_ok=True)

    zroot = zarr.open_group(replay_data_path, mode="r")
    all_ep_keys = sorted(zroot.keys(), key=lambda k: int(k.split("_")[1]))

    if replay_episodes is not None:
        ep_keys = [f"episode_{i}" for i in replay_episodes if f"episode_{i}" in zroot]
    else:
        ep_keys = all_ep_keys

    print(f"replay: {len(ep_keys)} episodes from {replay_data_path}")

    for ep_key in tqdm(ep_keys, desc="replay episodes"):
        ep = zroot[ep_key]
        expert_actions = ep["action0_8d"][:]       # (T, 8)
        T = len(expert_actions)

        policy.reset()
        policy_actions = []

        for t in range(T):
            with torch.no_grad():
                policy_input = _zarr_obs_to_policy_input(ep, t, device)
                policy_input = normalizer.normalize(policy_input)
                action_dict = policy.predict_action(policy_input)
                action_dict = normalizer.unnormalize(action_dict)
                pred_action = action_dict["action0_8d"][0, 0, :].cpu().numpy()
            policy_actions.append(pred_action)

        # save episode
        ep_idx = int(ep_key.split("_")[1])
        episode_dir = os.path.join(task_dir, f"episode_{ep_idx:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        expert_arr = expert_actions                              # (T, 8)
        policy_arr = np.stack(policy_actions)                    # (T, 8)
        per_step_mse = np.mean((expert_arr - policy_arr) ** 2, axis=1)  # (T,)

        # obs side-by-side video: concat third_person + wrist horizontally
        tp_frames = ep["third_person_camera"][:]   # (T, 256, 256, 3)
        wc_frames = ep["robot0_wrist_camera"][:]   # (T, 256, 256, 3)
        obs_frames = np.concatenate([tp_frames, wc_frames], axis=2)  # (T, 256, 512, 3)

        # no reward available offline — pass zeros for overlay
        dummy_rewards = np.zeros(T, dtype=np.float32)
        obs_annotated = _annotate_replay_obs_frames(
            obs_frames.copy(), expert_arr, policy_arr, dummy_rewards,
        )
        _save_video_and_frames(episode_dir, "obs", obs_annotated, fps)

        # per-step divergence stats
        per_step_cos, per_step_rel = [], []
        for t in range(T):
            cs, re = _divergence_stats(expert_arr[t], policy_arr[t])
            per_step_cos.append(cs)
            per_step_rel.append(re)

        # metrics
        metrics = {
            "episode_idx": ep_idx,
            "num_steps": T,
            "mean_cosine_sim_%": float(np.mean(per_step_cos)),
            "mean_relative_err_%": float(np.mean(per_step_rel)),
            "mean_mse": float(per_step_mse.mean()),
            "max_mse": float(per_step_mse.max()),
            "max_mse_step": int(per_step_mse.argmax()),
            "per_step_cosine_sim_%": per_step_cos,
            "per_step_relative_err_%": per_step_rel,
            "per_step_mse": per_step_mse.tolist(),
        }
        with open(os.path.join(episode_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"saved {len(ep_keys)} replay episodes to {task_dir}/")


# cli

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IL policy on MIKASA-Robo")
    parser.add_argument("--env-id", type=str, required=True, choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-steps", type=int, default=None,
                        help="total env.step() calls (default: task episode_steps)")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    # replay comparison mode
    parser.add_argument("--replay-data", type=str, default=None,
                        help="path to expert zarr (episode_data.zarr). enables replay mode: "
                             "feed expert obs to policy offline, compare predictions per step.")
    parser.add_argument("--replay-episodes", type=int, nargs="+", default=None,
                        help="which episode indices to replay (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_cfg = TASK_REGISTRY[args.env_id]
    policy, normalizer = load_policy(args.checkpoint, device)

    if args.replay_data:
        # replay mode: no env needed, feed expert obs from zarr to policy offline
        print(f"\nreplay mode: {args.env_id} | expert data: {args.replay_data}")
        run_replay(
            policy, normalizer, device,
            replay_data_path=args.replay_data,
            replay_episodes=args.replay_episodes,
            output_dir=args.output_dir, env_id=args.env_id,
        )
    else:
        # normal eval mode
        num_eval_steps = args.num_eval_steps or task_cfg.episode_steps
        env, inner_env = make_eval_env(
            env_id=args.env_id, task_cfg=task_cfg, num_envs=args.num_envs,
            num_eval_steps=num_eval_steps, capture_video=args.capture_video,
            output_dir=args.output_dir, seed=args.seed, reward_mode=args.reward_mode,
        )
        print(f"\nevaluating {args.env_id} | {num_eval_steps} steps | {args.num_envs} envs")
        results = run_eval(
            env, inner_env, policy, normalizer, args.num_envs, num_eval_steps, device, args.seed,
            output_dir=args.output_dir, env_id=args.env_id,
        )
        env.close()

        print(f"\n{'='*50}")
        n = int(results.pop("num_episodes"))
        print(f"completed {n} episodes in {num_eval_steps * args.num_envs} total steps")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
