#!/bin/bash
set -euo pipefail

REMOTE="jeff@yihuaidesktop001"
REMOTE_PREFIX="/data/yihuai/data/imitation-learning-policies/data"
LOCAL_PREFIX="mikasa_models"

# run path (no leading/trailing slash)
RUN_PATH="mikasa_remember_color_3/2026-03-26/04-36-22_remember_color_3_diffusion_memory_lr3e-4_state_only_mem_2step"
# RUN_PATH="mikasa_remember_color_3/2026-03-24/23-35-14_remember_color_3_diffusion_memory_lr3e-4_state_only_mem_2step_no_aug_1500episodes"
# RUN_PATH="mikasa_remember_color_3/2026-03-24/23-35-14_remember_color_3_diffusion_memory_lr3e-4_state_only_mem_2step_no_aug_1500episodes"
# RUN_PATH="mikasa_remember_shape_3/2026-03-24/18-04-43_remember_shape_3_diffusion_memory_lr3e-4_state_only_mem_1step"
# RUN_PATH="mikasa_remember_color_3/2026-03-24/18-07-41_remember_color_3_diffusion_memory_lr3e-4_state_only_mem_1step"

# env id — auto-derived from RUN_PATH folder name
# mikasa_remember_shape_3 -> RememberShape3-v0, mikasa_remember_color_3 -> RememberColor3-v0, etc.
TASK_FOLDER=$(echo "${RUN_PATH}" | cut -d/ -f1)  # e.g. mikasa_remember_shape_3
TASK_NAME=$(echo "${TASK_FOLDER}" | sed 's/^mikasa_//' \
    | sed -E 's/_([a-z])/\U\1/g' \
    | sed -E 's/_([0-9])/\1/g' \
    | sed -E 's/^([a-z])/\U\1/')  # remember_shape_3 -> RememberShape3
ENV_ID="${TASK_NAME}-v0"

# eval settings
NUM_ENVS=16
NUM_EVAL_STEPS=420
SEED=1000
# DIRECT_QPOS=""
DIRECT_QPOS="--direct-qpos"  # uncomment for direct-qpos (teleport) eval
# ABS_JOINT_POS="--abs-joint-pos"
ABS_JOINT_POS=""          # uncomment above for absolute joint pos (pd_joint_pos controller)
# NO_PROPRIO="--no-proprio"
NO_PROPRIO=""
# TRAIN_SEEDS_ONLY="--train-seeds-only seeds/train_seeds.json"
TRAIN_SEEDS_ONLY=""     # comment out above to use random seeds

# ============================================================================
# 1. Fetch from remote
# ============================================================================
LOCAL_DIR="${LOCAL_PREFIX}/${RUN_PATH}"
REMOTE_DIR="${REMOTE}:${REMOTE_PREFIX}/${RUN_PATH}/"

echo "=== Fetching from remote ==="
echo "  remote: ${REMOTE_DIR}"
echo "  local:  ${LOCAL_DIR}"
mkdir -p "${LOCAL_DIR}"
rsync -rzvP "${REMOTE_DIR}" "${LOCAL_DIR}"

# ============================================================================
# 2. Checkpoint selection
# ============================================================================
# Hardcode a specific checkpoint (relative to LOCAL_DIR), or leave empty to auto-pick latest epoch.
CKPT_OVERRIDE="checkpoints/epoch_20_train_mean_loss_0_000.ckpt"
# CKPT_OVERRIDE=""

if [ -n "${CKPT_OVERRIDE}" ]; then
    LATEST_CKPT="${LOCAL_DIR}/${CKPT_OVERRIDE}"
    if [ ! -f "${LATEST_CKPT}" ]; then
        echo "ERROR: hardcoded checkpoint not found: ${LATEST_CKPT}"
        exit 1
    fi
else
    CKPT_DIR="${LOCAL_DIR}/checkpoints"
    if [ ! -d "${CKPT_DIR}" ]; then
        echo "ERROR: no checkpoints dir at ${CKPT_DIR}"
        exit 1
    fi
    LATEST_CKPT=$(ls "${CKPT_DIR}"/epoch_*.ckpt 2>/dev/null \
        | sort -t/ -k1 -V \
        | tail -1)
    if [ -z "${LATEST_CKPT}" ]; then
        echo "ERROR: no epoch_*.ckpt found in ${CKPT_DIR}"
        exit 1
    fi
fi
echo "=== Using checkpoint: ${LATEST_CKPT} ==="

# auto-derive output dir from task name, run name suffix, and epoch
# e.g. 18-06-43_remember_shape_3_diffusion_memory_lr3e-4_state_only_mem_1step_include_hist_action
#   -> state_only_mem_1step_include_hist_action (strip timestamp + task + diffusion_memory_lr*)
RUN_SUFFIX=$(basename "${RUN_PATH}" | sed -E 's/^[0-9-]+_[a-z]+_[a-z]+_[0-9]+_diffusion_memory_lr[^_]+_//')
EPOCH=$(basename "${LATEST_CKPT}" | grep -oP 'epoch_\d+')  # e.g. epoch_6
MODE=""
[ -n "${DIRECT_QPOS}" ] && MODE="_dqpos"
[ -n "${ABS_JOINT_POS}" ] && MODE="_absjpos"
[ -n "${TRAIN_SEEDS_ONLY:-}" ] && MODE="${MODE}_trainseeds"
OUTPUT_DIR="eval_results/${TASK_NAME}_${RUN_SUFFIX}_${EPOCH}${MODE}"

# ============================================================================
# 3. Run eval
# ============================================================================
echo "=== Running eval ==="
echo "  checkpoint:  ${LATEST_CKPT}"
echo "  env:         ${ENV_ID}"
echo "  direct-qpos: ${DIRECT_QPOS:-off}"
echo "  abs-jpos:    ${ABS_JOINT_POS:-off}"
echo "  no-proprio:  ${NO_PROPRIO:-off}"
echo "  num-envs:    ${NUM_ENVS}"
echo "  eval-steps:  ${NUM_EVAL_STEPS}"
echo "  output:      ${OUTPUT_DIR}"
echo "  seed:        ${SEED}"

conda run --no-capture-output -n mikasa python -u eval/mikasa_eval.py \
    --env-id "${ENV_ID}" \
    --checkpoint "${LATEST_CKPT}" \
    ${DIRECT_QPOS:-} \
    ${ABS_JOINT_POS:-} \
    ${NO_PROPRIO:-} \
    ${TRAIN_SEEDS_ONLY:-} \
    --num-envs "${NUM_ENVS}" \
    --num-eval-steps "${NUM_EVAL_STEPS}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}"

# ============================================================================
# 4. Seed overlap analysis: in/out of training data vs success/failure
# ============================================================================
echo ""
echo "=== Eval episodes saved to: ${OUTPUT_DIR}/${ENV_ID} ==="
echo "=== Seed overlap analysis ==="
python3 -u eval/seed_analysis.py "${OUTPUT_DIR}/${ENV_ID}" "seeds/train_seeds.json"
