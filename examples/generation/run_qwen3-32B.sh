source "/code/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME="/checkpoints/yibiaoy-sandbox/HF"
export HF_HUB_OFFLINE=1
export DIST_NNODES="${REPLICA}"

export BASE_MODEL=Qwen/Qwen3-32B
export EXPERIMENT_NAME=qwen3-32b-generation

#!/bin/bash
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
    ray start --address="${HEAD_ADDR}:${HEAD_PORT}" --block
fi
echo "Ray all worker nodes started"


save_path=/data/yibiaoy-sandbox/skywork-or1/qwen3-32b_generation.parquet
train_1p5b_math_file=/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math.parquet

if [ "${HOSTNAME##*-}" -eq 0 ]; then
    # Command 1
    echo "Executing command 1 because DIST_NODE_RANK is 0"
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=4 \
        trainer.n_gpus_per_node=8 \
        data.path=$train_1p5b_math_file \
        data.prompt_key=prompt \
        data.n_samples=6 \
        data.output_path=$save_path \
        data.batch_size=8192 \
        model.path=$BASE_MODEL\
        +model.trust_remote_code=True \
        rollout.name=vllm \
        rollout.temperature=0.6 \
        rollout.top_k=20 \
        rollout.top_p=0.95 \
        rollout.prompt_length=2048 \
        rollout.response_length=32768 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.95 \
        rollout.max_num_seqs=128 \
        rollout.max_num_batched_tokens=34816 2>&1 | tee /checkpoints/yibiaoy-sandbox/skywork-or1/verl_terminal_log_$EXPERIMENT_NAME.log
else
    sleep infinity
fi
