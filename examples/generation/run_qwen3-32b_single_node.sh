source "/code/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate verl
export HF_HOME="/checkpoints/yibiaoy-sandbox/HF"
export HF_HUB_OFFLINE=1

export BASE_MODEL=Qwen/Qwen3-32B
export DATA_DIR=/data/yibiaoy-sandbox/skywork-or1
export EXPERIMENT_NAME=skywork-or1-qwen3-32b-generation-single-node


save_path=/data/yibiaoy-sandbox/skywork-or1/qwen3-32b_generation_test.parquet
train_1p5b_math_file=/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math.parquet
train_files="['$train_1p5b_math_file']"

filter_path=/code/yibiaoy-sandbox/verl/outputs/evaluation/offline_generation/evaluation_counts_combined.csv

python3 -m verl.trainer.main_generation_dynamics \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$train_1p5b_math_file \
    data.prompt_key=prompt \
    data.n_samples=2 \
    data.filter_path=$filter_path \
    data.output_path=$save_path \
    data.batch_size=8 \
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

