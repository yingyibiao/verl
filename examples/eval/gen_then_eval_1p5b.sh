#!/usr/bin/env bash
set -euo pipefail
set -x

# ---------------- 环境 ----------------
source "/code/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME="/checkpoints/yibiaoy-sandbox/HF"
export HF_HUB_OFFLINE=1

BASE_DIR="/checkpoints/yibiaoy-sandbox/skywork-or1/skywork-or1-1p5b-baseline-32k/merged_hf_model"
OUT_DIR="./outputs/evaluation/skywork-or1-1p5b-baseline-32k"
LOG_DIR="./logs/skywork-or1-1p5b-baseline-32k"
mkdir -p "$OUT_DIR" "$LOG_DIR"

COMMON_ARGS=(
  trainer.nnodes=1
  trainer.n_gpus_per_node=8
  data.n_samples=32
  data.batch_size=102400
  rollout.temperature=0.8
  rollout.prompt_length=2048
  rollout.response_length=32768
  rollout.top_k=-1
  rollout.top_p=1.0
  rollout.gpu_memory_utilization=0.9
  rollout.max_num_seqs=128
  rollout.max_num_batched_tokens=34816
  rollout.tensor_model_parallel_size=1
)

run_eval() {
  local model_path="$1"
  local step_name="$2"

  # AIME24
  python3 -m verl.trainer.main_generation_eval \
      "${COMMON_ARGS[@]}" \
      model.path="$model_path" \
      data.path=/data/yibiaoy-sandbox/skywork-or1/aime24_modified.parquet \
      data.output_path="$OUT_DIR/Aime24_${step_name}.pkl" \
    >"$LOG_DIR/${step_name}_aime24.log" 2>&1

  # AIME25
  python3 -m verl.trainer.main_generation_eval \
      "${COMMON_ARGS[@]}" \
      model.path="$model_path" \
      data.path=/data/yibiaoy-sandbox/skywork-or1/aime25.parquet \
      data.output_path="$OUT_DIR/Aime25_${step_name}.pkl" \
    >"$LOG_DIR/${step_name}_aime25.log" 2>&1
}

# -------- 只评测编号为 40 的倍数的 step --------
for step_dir in "$BASE_DIR"/global_step_*; do
  [[ -d "$step_dir" ]] || continue
  step_name=$(basename "$step_dir")          # global_step_120
  step_num=${step_name#global_step_}         # 120
  # 过滤：仅当 step_num 能被 40 整除时才评测
  if (( step_num % 40 != 0 )); then
    echo "Skip $step_name (not multiple of 40)"
    continue
  fi
  if (( step_num <= 2120 )); then
    echo "Skip $step_name (<= 2120)"
    continue
  fi
  echo "========== 评测 $step_name =========="
  run_eval "$step_dir" "$step_name"
done
