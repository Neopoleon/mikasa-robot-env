"""shared utilities for MIKASA-Robo evaluation and replay.

obs:    2 rgb cameras (third-person + wrist, 256x256) + 8d proprio (7 arm qpos + 1 gripper).
action: 8d joint-space deltas (7 arm pd_joint_delta_pos + 1 gripper).
policy: diffusion transformer (SigLIP2 + memory transformer), predicts action chunks.
"""

import json
import os
import sys
from dataclasses import dataclass, field

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import yaml

sys.path.insert(0, "/home/jeff/imitation-learning-policies")
sys.path.insert(0, "/home/jeff/robot-utils/src")

from omegaconf import OmegaConf
import hydra

import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from mikasa_robo_suite.memory_envs import *  # noqa: F403 — registers MIKASA envs
from mikasa_robo_suite.utils.wrappers import *  # noqa: F403
from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper

from imitation_learning.common.dataclasses import construct_data_meta_dict
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.policies.base_policy import BasePolicy
from robot_utils.config_utils import register_resolvers

import warnings
warnings.filterwarnings(
    "ignore", message=".*env\\.\\w+ to get variables from other wrappers is deprecated.*"
)

register_resolvers()


# task registry — add new tasks here.
# wrappers applied: _base_wrappers → extra_wrappers → FlattenRGBD → FlattenAction → VecEnv.

WrapperSpec = tuple[type, dict]


@dataclass
class TaskConfig:
    episode_steps: int
    extra_wrappers: list[WrapperSpec] = field(default_factory=list)


def _base_wrappers() -> list[WrapperSpec]:
    return [
        (StateOnlyTensorToDictWrapper, {}),            # raw tensor → dict with 'prompt'/'oracle_info' (required by FlattenRGBD)
        (InitialZeroActionWrapper, {"n_initial_steps": 0}),
        (RenderStepInfoWrapper, {}),
        (RenderRewardInfoWrapper, {}),
    ]


TASK_REGISTRY: dict[str, TaskConfig] = {
    "ShellGameTouch-v0": TaskConfig(
        episode_steps=90,
        extra_wrappers=[(ShellGameRenderCupInfoWrapper, {})],
    ),
    "InterceptMedium-v0": TaskConfig(episode_steps=90),
    "RememberColor3-v0": TaskConfig(
        episode_steps=60,
        extra_wrappers=[(RememberColorInfoWrapper, {})],
    ),
    "RememberColor6-v0": TaskConfig(
        episode_steps=60,
        extra_wrappers=[(RememberColorInfoWrapper, {})],
    ),
    "RememberColor9-v0": TaskConfig(
        episode_steps=60,
        extra_wrappers=[(RememberColorInfoWrapper, {})],
    ),
    "RememberShape3-v0": TaskConfig(
        episode_steps=60,
        extra_wrappers=[(RememberShapeInfoWrapper, {})],
    ),
}


# policy loading

def load_policy(checkpoint_path: str, device: torch.device) -> tuple[BasePolicy, FixedNormalizer]:
    """load policy + normalizer from checkpoint.
    ckpt keys: model_state_dict, normalizer_state_dict, cfg_str_unresolved (hydra yaml).
    """
    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = OmegaConf.create(ckpt["cfg_str_unresolved"])
    # Save config to a yaml file in the same path as the checkpoint
    with open(checkpoint_path.replace(".ckpt", ".yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f)
    print(f"Saved config to: {checkpoint_path.replace('.ckpt', '.yaml')}")
    policy = hydra.utils.instantiate(cfg["workspace"]["model"])
    assert isinstance(policy, BasePolicy)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()
    policy.reset()

    data_meta = construct_data_meta_dict(cfg["workspace"]["train_dataset"]["output_data_meta"])
    normalizer = FixedNormalizer(data_meta)
    normalizer.to(device)
    normalizer.load_state_dict(ckpt["normalizer_state_dict"])

    return policy, normalizer


# environment

def make_eval_env(
    env_id: str,
    task_cfg: TaskConfig,
    num_envs: int,
    num_eval_steps: int,
    capture_video: bool,
    output_dir: str,
    seed: int,
    reward_mode: str,
) -> tuple[ManiSkillVectorEnv, gym.Env]:
    """create gpu-vectorized env with wrapper stack.
    returns (vec_env, inner_env) — inner_env.render() has wrapper overlays,
    vec_env.render() bypasses them.
    """
    env_kwargs = dict(
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="all" if capture_video else "rgb_array",
        sim_backend="gpu",
        reward_mode=reward_mode,
    )
    env = gym.make(env_id, num_envs=num_envs, reconfiguration_freq=1, **env_kwargs)

    for wrapper_cls, wrapper_kwargs in _base_wrappers() + task_cfg.extra_wrappers:
        env = wrapper_cls(env, **wrapper_kwargs)

    # → {rgb: (B,H,W,6) uint8, joints: (B,25) float}
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False, joints=True)

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    inner_env = env  # keep ref for render() with overlays

    if capture_video:
        video_dir = os.path.join(output_dir, env_id, "videos")
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(
            env, output_dir=video_dir, save_trajectory=False,
            max_steps_per_video=num_eval_steps, video_fps=30,
        )

    vec_env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    return vec_env, inner_env


# obs → policy input

def obs_to_policy_input(
    obs: dict[str, torch.Tensor],
    num_envs: int,
    device: torch.device,
    no_proprio: bool = False,
) -> dict[str, torch.Tensor]:
    """env obs → policy input dict.
    rgb (B,H,W,6): [:3]=third_person, [3:]=wrist → (B,1,3,H,W) float [0,1] each.
    joints (B,25): [7:14]=arm_qpos, [14:15]=gripper_qpos → robot0_8d (B,1,8).
    episode_idx (B,): static arange for MemoryTransformer per-env history tracking.
    if no_proprio: omit robot0_8d (vision-only policy).
    """
    rgb = obs["rgb"]
    joints = obs["joints"]

    third_person = rgb[..., :3].float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(1)
    wrist_cam    = rgb[..., 3:].float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(1)

    result = {
        "third_person_camera": third_person.to(device),
        "robot0_wrist_camera": wrist_cam.to(device),
        "episode_idx":         torch.arange(num_envs, device=device),
    }
    if not no_proprio:
        proprio = torch.cat([joints[:, 7:14], joints[:, 14:15]], dim=-1).unsqueeze(1)
        result["robot0_8d"] = proprio.to(device)
    return result


# inference

ACTION_HORIZON = 1  # actions per prediction, matches training chunk size


@torch.no_grad()
def predict_action(
    obs: dict[str, torch.Tensor],
    policy: BasePolicy,
    normalizer: FixedNormalizer,
    num_envs: int,
    device: torch.device,
    no_proprio: bool = False,
) -> torch.Tensor:
    """obs → normalize → diffusion denoise → unnormalize → action chunk (B, ACTION_HORIZON, 8)."""
    batch = obs_to_policy_input(obs, num_envs, device, no_proprio=no_proprio)
    batch = normalizer.normalize(batch)
    action_dict = policy.predict_action(batch)
    action_dict = normalizer.unnormalize(action_dict)
    return action_dict["action0_8d"][:, :ACTION_HORIZON, :]


# frame annotation helpers

def _put_text(frame, text, pos, font_scale=0.5, thickness=1, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _annotate_frames(frames, actions, rewards, font_scale=0.35, thickness=1):
    """overlay step, reward, and action on each frame. for obs side-by-side (256x512)."""
    for i in range(len(frames)):
        y = 15
        _put_text(frames[i], f"step: {i}", (5, y), font_scale, thickness)
        y += 18
        _put_text(frames[i], f"rew: {rewards[i]:.3f}", (5, y), font_scale, thickness)
        y += 18
        act = actions[i]
        _put_text(frames[i], f"act: [{', '.join(f'{a:.2f}' for a in act)}]", (5, y), font_scale, thickness)
    return frames


def _annotate_action_on_frames(frames, actions, font_scale=0.5, thickness=1):
    """overlay action text on render frames (512x512). other overlays already applied by wrappers."""
    for i in range(len(frames)):
        act = actions[i]
        # split 8d action into two lines to fit within 512px width
        _put_text(frames[i], f"act[0:4]: [{', '.join(f'{a:.2f}' for a in act[:4])}]", (10, 150), font_scale, thickness)
        _put_text(frames[i], f"act[4:8]: [{', '.join(f'{a:.2f}' for a in act[4:])}]", (10, 175), font_scale, thickness)
    return frames


def _divergence_stats(ea, pa):
    """compute intuitive divergence metrics between expert and policy action vectors.
    returns (cosine_sim %, relative_err %) — cosine_sim: direction agreement (100%=identical),
    relative_err: mean |diff| / mean |expert| * 100 (0%=perfect match).
    """
    dot = float(np.dot(ea, pa))
    norm_e, norm_p = float(np.linalg.norm(ea)), float(np.linalg.norm(pa))
    cos_sim = dot / max(norm_e * norm_p, 1e-8) * 100.0
    rel_err = float(np.mean(np.abs(ea - pa))) / max(float(np.mean(np.abs(ea))), 1e-8) * 100.0
    return cos_sim, rel_err


def _annotate_replay_obs_frames(frames, expert_actions, policy_actions, rewards):
    """overlay expert (green) vs policy (red) actions + divergence on obs side-by-side frames (256x512)."""
    fs, th = 0.32, 1
    for i in range(len(frames)):
        y = 14
        _put_text(frames[i], f"step:{i}  rew:{rewards[i]:.3f}", (4, y), fs, th)
        ea, pa = expert_actions[i], policy_actions[i]
        cos_sim, rel_err = _divergence_stats(ea, pa)
        y += 16
        _put_text(frames[i], f"expert: [{', '.join(f'{a:.2f}' for a in ea)}]", (4, y), fs, th, color=(0, 255, 0))
        y += 16
        _put_text(frames[i], f"policy: [{', '.join(f'{a:.2f}' for a in pa)}]", (4, y), fs, th, color=(0, 0, 255))
        y += 16
        _put_text(frames[i], f"cos:{cos_sim:.0f}%  relErr:{rel_err:.0f}%", (4, y), fs, th, color=(255, 255, 0))
    return frames


def _annotate_replay_render_frames(frames, expert_actions, policy_actions):
    """overlay expert (green) vs policy (red) actions + divergence on render frames (512x512).
    step/reward/task overlays already applied by wrappers."""
    fs, th = 0.45, 1
    for i in range(len(frames)):
        ea, pa = expert_actions[i], policy_actions[i]
        cos_sim, rel_err = _divergence_stats(ea, pa)
        y = 145
        _put_text(frames[i], f"expert[0:4]: [{', '.join(f'{a:.2f}' for a in ea[:4])}]", (8, y), fs, th, color=(0, 255, 0))
        y += 22
        _put_text(frames[i], f"expert[4:8]: [{', '.join(f'{a:.2f}' for a in ea[4:])}]", (8, y), fs, th, color=(0, 255, 0))
        y += 22
        _put_text(frames[i], f"policy[0:4]: [{', '.join(f'{a:.2f}' for a in pa[:4])}]", (8, y), fs, th, color=(0, 0, 255))
        y += 22
        _put_text(frames[i], f"policy[4:8]: [{', '.join(f'{a:.2f}' for a in pa[4:])}]", (8, y), fs, th, color=(0, 0, 255))
        y += 22
        _put_text(frames[i], f"cos:{cos_sim:.0f}%  relErr:{rel_err:.0f}%", (8, y), fs, th, color=(255, 255, 0))
    return frames


def _save_video_and_frames(episode_dir, name, frames, fps):
    """save video and individual pngs under episode_dir/{name}."""
    imageio.mimsave(os.path.join(episode_dir, f"video_{name}.mp4"), frames, fps=fps)
    frames_dir = os.path.join(episode_dir, f"frames_{name}")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(os.path.join(frames_dir, f"frame_{i:03d}.png"), frame)


# episode recording

def save_episode(
    episode_dir: str,
    rgb_frames: list[np.ndarray],
    joints_frames: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    success: bool,
    ep_metrics: dict,
    render_frames: list[np.ndarray] | None = None,
    fps: int = 30,
):
    """save episode.npz (clean obs) + video/pngs (with overlays if available) + metrics json."""
    os.makedirs(episode_dir, exist_ok=True)
    T = len(rgb_frames)

    rgb = np.stack(rgb_frames)
    joints = np.stack(joints_frames)
    action = np.stack(actions)
    reward = np.array(rewards, dtype=np.float32)
    success_arr = np.full(T, int(success), dtype=np.int64)
    done = np.zeros(T, dtype=np.int64)
    done[-1] = 1

    np.savez_compressed(
        os.path.join(episode_dir, "episode.npz"),
        rgb=rgb, joints=joints, action=action,
        reward=reward, success=success_arr, done=done,
    )

    # raw obs: side-by-side two-camera view (T,H,2W,3)
    obs_frames = np.concatenate([rgb[..., :3], rgb[..., 3:]], axis=2)

    # overlay text on both obs and render frames
    obs_annotated = _annotate_frames(obs_frames.copy(), action, reward, font_scale=0.35, thickness=1)
    _save_video_and_frames(episode_dir, "obs", obs_annotated, fps)

    if render_frames:
        render_arr = np.stack(render_frames)
        # render already has step/reward/task overlays from wrappers; add action overlay
        render_annotated = _annotate_action_on_frames(render_arr.copy(), action, font_scale=0.7, thickness=2)
        _save_video_and_frames(episode_dir, "render", render_annotated, fps)

    with open(os.path.join(episode_dir, "metrics.json"), "w") as f:
        json.dump(ep_metrics, f, indent=2)


# per-env episode buffers

class EpisodeBuffers:
    """per-env rolling buffers. append() every step, flush() on episode completion."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.rgb = [[] for _ in range(num_envs)]
        self.joints = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]
        self.rewards = [[] for _ in range(num_envs)]
        self.render = [[] for _ in range(num_envs)]  # overlaid frames from env.render()

    def append_obs(self, obs: dict[str, torch.Tensor], render_frame=None):
        rgb_cpu = obs["rgb"].cpu().numpy()
        joints_cpu = obs["joints"].cpu().numpy()
        for i in range(self.num_envs):
            self.rgb[i].append(rgb_cpu[i])
            self.joints[i].append(joints_cpu[i])
        if render_frame is not None:
            if isinstance(render_frame, torch.Tensor):
                render_frame = render_frame.cpu().numpy()
            for i in range(self.num_envs):
                self.render[i].append(render_frame[i])

    def append_action(self, action: torch.Tensor):
        action_cpu = action.cpu().numpy()
        for i in range(self.num_envs):
            self.actions[i].append(action_cpu[i])

    def append_reward(self, reward: torch.Tensor):
        reward_cpu = reward.cpu().numpy()
        for i in range(self.num_envs):
            self.rewards[i].append(float(reward_cpu[i]))

    def flush(self, env_idx: int) -> tuple[list, list, list, list, list]:
        data = (
            list(self.rgb[env_idx]),
            list(self.joints[env_idx]),
            list(self.actions[env_idx]),
            list(self.rewards[env_idx]),
            list(self.render[env_idx]),
        )
        self.rgb[env_idx].clear()
        self.joints[env_idx].clear()
        self.actions[env_idx].clear()
        self.rewards[env_idx].clear()
        self.render[env_idx].clear()
        return data


# episode completion handling

def handle_episode_completions(
    infos: dict,
    buffers: EpisodeBuffers,
    num_envs: int,
    num_episodes: int,
    task_dir: str,
    metrics: dict[str, list],
) -> int:
    """save completed episodes to disk. returns updated episode count.
    ManiSkillVectorEnv sets infos["_final_info"] (B,) bool mask and
    infos["final_info"]["episode"] dict of (B,) metric tensors on episode end.
    """
    if "final_info" not in infos:
        return num_episodes

    mask = infos["_final_info"]
    ep_data = infos["final_info"]["episode"]

    for k, v in ep_data.items():
        metrics[k].append(v)

    for i in range(num_envs):
        if not mask[i]:
            continue

        ep_metrics = {k: float(v[i]) for k, v in ep_data.items()}
        success = ep_metrics.get("success_once", 0.0) > 0.5
        tag = "success" if success else "failure"
        ep_metrics["episode_idx"] = num_episodes

        rgb_frames, joints_frames, actions, rewards, render_frames = buffers.flush(i)
        save_episode(
            episode_dir=os.path.join(task_dir, f"episode_{num_episodes:04d}_{tag}"),
            rgb_frames=rgb_frames,
            joints_frames=joints_frames,
            actions=actions,
            rewards=rewards,
            success=success,
            ep_metrics=ep_metrics,
            render_frames=render_frames,
        )
        num_episodes += 1

    return num_episodes
