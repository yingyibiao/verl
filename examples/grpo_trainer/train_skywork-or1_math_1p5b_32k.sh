source "/code/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate verl
export HF_HOME="/checkpoints/yibiaoy-sandbox/HF"
export HF_HUB_OFFLINE=1
export DIST_NNODES="${REPLICA}"

export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export DATA_DIR=/data/yibiaoy-sandbox/skywork-or1
export MLFLOW_TRACKING_URI=file:///data/yibiaoy-sandbox/skywork-or1/mlflow_logs
export EXPERIMENT_NAME=skywork-or1-1p5b-baseline-32k

export VERL_PPO_LOGGING_LEVEL=DEBUG

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


train_1p5b_math_file=/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math.parquet
aime24_file=/data/yibiaoy-sandbox/skywork-or1/aime24_modified.parquet
aime25_file=/data/yibiaoy-sandbox/skywork-or1/aime25.parquet
train_files="['$train_1p5b_math_file']"
test_files="['$aime24_file', '$aime25_file']"

# Entropy Config
ENTROPY_COEFF=0.0

ROLLOUT_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048
RES_LENGTH=32768
GROUP_SIZE=16
N_VAL_SAMPLES=8

TRAIN_TEMPERATURE=1.0

TP=1
SP=1
MAX_TOKEN_LEN=$((2*(RES_LENGTH + MAX_PROMPT_LENGTH) / SP))


if [ "${HOSTNAME##*-}" -eq 0 ]; then
    # Command 1
    echo "Executing command 1 because DIST_NODE_RANK is 0"
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=$ROLLOUT_BATCH_SIZE \
        data.val_batch_size=512 \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$RES_LENGTH \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=8 \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
        actor_rollout_ref.rollout.n=$GROUP_SIZE \
        actor_rollout_ref.rollout.val_kwargs.n=$N_VAL_SAMPLES \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=/checkpoints/yibiaoy-sandbox/skywork-or1/$EXPERIMENT_NAME \
        trainer.logger=['console','mlflow'] \
        trainer.project_name='verl-skywork-or1' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$DIST_NNODES \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path='/checkpoints/yibiaoy-sandbox/skywork-or1/skywork-or1-1p5b-baseline/global_step_800' \
        trainer.total_epochs=30 2>&1 | tee /checkpoints/yibiaoy-sandbox/skywork-or1/verl_terminal_log_$EXPERIMENT_NAME.log
else
    sleep infinity
fi
