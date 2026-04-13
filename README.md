# MIKASA-Robo Env

Simulation side for evaluating Gated Memory Policies trained from [`imitation-learning-policies`](../imitation-learning-policies) on MIKASA memory-intensive manipulation tasks.

## Quick Start

```bash
conda env create -f env.yml
conda activate mikasa
pip install -e . --no-deps
```

## Evaluation

### Local (single checkpoint)

```bash
python eval/mikasa_eval.py \
    --env-id ShellGameTouch-v0 \
    --checkpoint path/to/epoch_50.ckpt \
    --num-envs 16
```

Three control modes are available:

| Flag              | What the policy predicts | Execution                          |
| ----------------- | ------------------------ | ---------------------------------- |
| _(default)_       | 8D joint deltas          | `pd_joint_delta_pos` PD controller |
| `--abs-joint-pos` | Absolute target qpos     | `pd_joint_pos` PD controller       |
| `--direct-qpos`   | Absolute joint positions | Teleport (no physics)              |

### Distributed (multi-checkpoint sweep)

Three terminals coordinate via [robotmq](https://pypi.org/project/robotmq/):

```bash
# 1. Policy server — loads checkpoints, serves inference (imitation-learning-policies env)
python scripts/run_remote_policy_server.py

# 2. Env server — runs MIKASA sims on GPU, evaluates, reports results (mikasa env)
bash shell_scripts/serve_mikasa_env.sh [GPU_ID] [PORT]

# 3. Orchestrator — queues checkpoints, collects results (imitation-learning-policies env)
python scripts/serve_mikasa_checkpoints.py
```

> **Note:** The env server defaults to **GPU 1** and port `18765`. On a single-GPU machine, pass `0` explicitly: `bash shell_scripts/serve_mikasa_env.sh 0`. Override the policy server address with `--policy-server` (env server) or `--port` (orchestrator).

## Camera Resolution

Environments render at **128x128** (both third-person and wrist cameras). The policy expects **256x256**, so the eval pipeline bilinear-upsamples automatically before inference (`eval/policy.py`). If you change the environment cameras to 256x256, the upsample is skipped.

To change rendering resolution:

- **Third-person** (`base_camera`): edit `CameraConfig` in each file under `mikasa_robo_suite/memory_envs/`
- **Wrist** (`hand_camera`): edit `panda_wristcam.py` in your `mani-skill` site-packages

## Repo Layout

```text
eval/
    mikasa_eval.py            Eval loops and CLI
    mikasa_env_server.py      Distributed env server (robotmq)
    remote_policy_client.py   RPC policy client
    policy.py                 Policy protocol, checkpoint loading, obs conversion
    recording.py              Episode buffers, saving, video annotation
    tasks.py                  Task registry, env creation helpers
shell_scripts/
    serve_mikasa_env.sh       Launch env server with GPU/EGL setup
tools/
    compress_all.sh           Compress episode directories
    convert_npz_to_zarr.py    Convert eval episodes (npz -> zarr)
    viz_episodes.py           Visualize episodes (npz or zarr)
```