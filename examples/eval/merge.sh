#!/usr/bin/env bash
set -euo pipefail

# ======== 配置区域 ========
LOCAL_BASE="/checkpoints-fsx/yibiaoy-sandbox/skywork-or1/skywork-or1-1p5b-baseline-32k"
LOCAL_BASE="/checkpoints-fsx/yibiaoy-sandbox/skywork-or1/skywork-or1-1p5b-baseline"
TARGET_BASE="$LOCAL_BASE/merged_hf_model"
NUM_PARALLEL=16            # 根据机器 CPU/IO 能力调整并行数
PYTHON_EXE=python         # 如果有多版本 Python，可指定绝对路径
MERGER_SCRIPT="scripts/model_merger.py"
MIN_STEP=0             # 新增：设置需要转换的最小步骤数
# =========================

mkdir -p "$TARGET_BASE"

convert_one() {
  step_path="$1"                       # 绝对路径 …/global_step_xxx
  step_name=$(basename "$step_path")   # global_step_xxx

  # 从 "global_step_12345" 中提取数字 12345
  # ${string##prefix} 是一种bash参数扩展，用于从字符串开头删除最长的匹配前缀
  step_num="${step_name##global_step_}"

  # 检查步骤数是否小于或等于 MIN_STEP
  if [[ "$step_num" -le "$MIN_STEP" ]]; then
    echo "[`date '+%H:%M:%S'`] Skipping $step_name (step $step_num <= $MIN_STEP)"
    return 0 # 正常退出，不执行后续转换
  fi

  tgt="$TARGET_BASE/$step_name"
  mkdir -p "$tgt"
  echo "[`date '+%H:%M:%S'`] Converting $step_name -> $tgt"

  "$PYTHON_EXE" "$MERGER_SCRIPT" merge \
      --backend fsdp \
      --local_dir "$step_path/actor" \
      --target_dir "$tgt"
}
export -f convert_one
export TARGET_BASE MERGER_SCRIPT PYTHON_EXE MIN_STEP # 导出新增的变量

# 遍历并并行执行 (此部分无需改动)
find "$LOCAL_BASE" -maxdepth 1 -type d -name 'global_step_*' -print0 \
  | sort -zV \
  | xargs -0 -n1 -P "$NUM_PARALLEL" bash -c 'convert_one "$0"'