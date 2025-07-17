#!/bin/bash

source "/code/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME="/checkpoints/yibiaoy-sandbox/HF"
export HF_HUB_OFFLINE=1
export DIST_NNODES="${REPLICA}"

# 定义两个模型
export MODEL1=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export MODEL2=/checkpoints/yibiaoy-sandbox/skywork-or1/skywork-or1-1p5b-baseline/merged_hf_model/global_step_800

export EXPERIMENT_NAME1=Distill-Qwen-1.5B-generation
export EXPERIMENT_NAME2=Distill-Qwen-1.5B-800steps-generation

# Ray 集群启动 (只启动一次)
if [ "${HOSTNAME##*-}" -eq 0 ]; then
    ray start --head --port=6379
    until [ "$(ray status | grep node_ | wc -l | awk '{print $1}')" -eq $DIST_NNODES ]; do
        echo "waiting for all workers up..."
        sleep 10
    done
else
    HEAD_ADDR="${HOSTNAME%-*}-0"
    HEAD_PORT=6379

    echo "Waiting for head node (${HEAD_ADDR}:${HEAD_PORT}) to become reachable..."
    until (echo > /dev/tcp/${HEAD_ADDR}/${HEAD_PORT}) >/dev/null 2>&1; do
        sleep 5
    done

    echo "Head node is reachable, starting ray worker..."
    ray start --address="${HEAD_ADDR}:${HEAD_PORT}" --block &
    RAY_PID=$!
fi
echo "Ray all worker nodes started"

# 输入数据路径
train_1p5b_math_file=/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math.parquet

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
        trainer.nnodes=4 \
        trainer.n_gpus_per_node=8 \
        data.path=$train_1p5b_math_file \
        data.prompt_key=prompt \
        data.n_samples=6 \
        data.output_path=$save_path \
        data.batch_size=4096 \
        model.path=$model \
        +model.trust_remote_code=True \
        rollout.name=vllm \
        rollout.temperature=1.0 \
        rollout.top_k=-1 \
        rollout.top_p=1 \
        rollout.prompt_length=2048 \
        rollout.response_length=32768 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.95 \
        rollout.max_num_seqs=128 \
        rollout.max_num_batched_tokens=34816 2>&1 | tee /checkpoints/yibiaoy-sandbox/skywork-or1/verl_terminal_log_$experiment_name.log
    
    echo "Completed generation for $experiment_name"
    echo "Waiting 30 seconds before next model..."
    sleep 30
}

if [ "${HOSTNAME##*-}" -eq 0 ]; then
    # 第一个模型
    save_path1=/data/yibiaoy-sandbox/skywork-or1/distill_qwen_1p5b_generation.parquet
    run_generation $MODEL1 $EXPERIMENT_NAME1 $save_path1
    
    # 第二个模型  
    save_path2=/data/yibiaoy-sandbox/skywork-or1/distill_qwen_1p5b_800steps_generation.parquet
    run_generation $MODEL2 $EXPERIMENT_NAME2 $save_path2
    
    echo "========================================="
    echo "All generation tasks completed!"
    echo "Model 1 output: $save_path1"
    echo "Model 2 output: $save_path2"
    echo "========================================="
    
    # 关闭 Ray 集群
    ray stop
else
    # Worker 节点保持运行直到主节点完成所有任务
    wait $RAY_PID
fi