#!/usr/bin/env bash
source "/code/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME="/checkpoints/yibiaoy-sandbox/HF"
export HF_HUB_OFFLINE=1

dir1='/data/yibiaoy-sandbox/skywork-or1/qwen3-32b_generation_shards/'
dir2='/data/yibiaoy-sandbox/skywork-or1/qwen3-32b_generation_part1_shards//'
dirs=( "$dir1" "$dir2" )
files_array=()
for d in "${dirs[@]}"; do
  files_array+=( "$d"* )
done
formatted_files=$(printf ",'%s'" "${files_array[@]}")
formatted_files="[${formatted_files#?}]"
offline_rollout_files_list="$formatted_files"
echo "Rollout Files: $offline_rollout_files_list"



# offline_rollout_files_list="/data/yibiaoy-sandbox/skywork-or1/qwen3-32b_generation_shards/shard_00000.parquet"
python3 -m verl.trainer.main_eval \
    data.path=$offline_rollout_files_list \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.output_path=/code/yibiaoy-sandbox/verl/outputs/evaluation/offline_generation/evaluation_correct_counts_combined.csv \
    data.reward_model_key=reward_model  2>&1 | tee /code/yibiaoy-sandbox/verl/logs/offline_generation/qwen3-32b_generation.log
