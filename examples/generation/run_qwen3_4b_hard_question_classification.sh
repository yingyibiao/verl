#!/bin/bash

source "/code/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME="/checkpoints/yibiaoy-sandbox/HF"
# export HF_HUB_OFFLINE=1

# 定义两个模型
export MODEL1=Qwen/Qwen3-32B
export EXPERIMENT_NAME1=Qwen3-32B-hard-question-classification


# 输入数据路径
# train_1p5b_math_file=/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_hard.parquet
# train_1p5b_math_file="/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_hard_mcp.parquet"
train_1p5b_math_file="/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_hard_json.parquet"


# 函数：运行生成任务
run_generation() {
    local model=$1
    local experiment_name=$2
    local save_path=$3
    
    echo "========================================="
    echo "Starting generation for model: $model"
    echo "Experiment: $experiment_name"
    echo "Output path: $save_path"
    echo "========================================="
    
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$train_1p5b_math_file \
        data.prompt_key=prompt \
        data.n_samples=5 \
        data.output_path=$save_path \
        data.batch_size=10000 \
        model.path=$model \
        +model.trust_remote_code=True \
        rollout.name=vllm \
        rollout.temperature=0.6 \
        rollout.top_k=20 \
        rollout.top_p=0.95 \
        rollout.prompt_length=3072 \
        rollout.response_length=16384 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.95 \
        rollout.max_num_seqs=256 \
        rollout.max_num_batched_tokens=38912 2>&1 | tee /checkpoints/yibiaoy-sandbox/skywork-or1/verl_terminal_log_$experiment_name.log
    
    echo "Completed generation for $experiment_name"
    echo "Waiting 30 seconds before next model..."
    sleep 5
    ray stop
}


save_path1=/data/yibiaoy-sandbox/skywork-or1/Qwen3-32B_hard_question_json.parquet
run_generation $MODEL1 $EXPERIMENT_NAME1 $save_path1
echo "========================================="
echo "All generation tasks completed!"
echo "Model 1 output: $save_path1"
echo "========================================="
