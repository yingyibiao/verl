# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
from omegaconf import ListConfig

# from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.reward_score import default_compute_score


@ray.remote
def process_item(reward_fn, data_source, index, response_lst, reward_data):
    """
    为单个项目处理，计算正确响应的数量和总响应的数量。
    """
    ground_truth = reward_data["ground_truth"]
    correct_count = sum(1 for r in response_lst if reward_fn(data_source, r, ground_truth) == 1.0)
    total_count = len(response_lst)
    return index, correct_count, total_count


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    # 检查 config.data.path 的类型，并创建文件路径列表
    if isinstance(config.data.path, (ListConfig, list)):
        file_paths = list(config.data.path)
    else:
        file_paths = [config.data.path]

    print(f"Reading {len(file_paths)} parquet file(s) into a single DataFrame...")
    # --- 核心修改：一次性读取所有文件 ---
    # 由于不需要 copy_to_local，可以直接将路径列表传递给 read_parquet
    dataset = pd.read_parquet(file_paths)
    print("All files have been loaded successfully.")

    # 读取所有需要的列
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    extra_info_key = config.data.get('extra_info_key', 'extra_info')
    extra_infos = dataset[extra_info_key]

    total = len(dataset)
    print(f"Total rows to process across all files: {total}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # 初始化计数器
    index_correct_counts = defaultdict(int)
    index_total_counts = defaultdict(int)
    
    # compute_score = get_custom_reward_fn(config)
    compute_score = default_compute_score

    # 为 DataFrame 中的每一行创建远程任务
    remote_tasks = []
    for i in range(total):
        index = extra_infos[i]['index']
        data_source = data_sources[i]
        task = process_item.remote(compute_score, data_source, index, responses[i], reward_model_data[i])
        remote_tasks.append(task)

    # Process results as they come in
    with tqdm(total=total, desc="Evaluating all rows") as pbar:
        while len(remote_tasks) > 0:
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                index, c_count, t_count = ray.get(result_id)
                index_correct_counts[index] += c_count
                index_total_counts[index] += t_count
                pbar.update(1)

    # 构建最终的 DataFrame
    print("\nProcessing results into final format...")
    correct_df = pd.DataFrame(list(index_correct_counts.items()), columns=['index', 'correct_count'])
    total_df = pd.DataFrame(list(index_total_counts.items()), columns=['index', 'total_count'])

    if correct_df.empty and total_df.empty:
        print("No results were generated.")
        return
        
    if correct_df.empty:
        results_df = total_df
        results_df['correct_count'] = 0
    elif total_df.empty:
        results_df = correct_df
        results_df['total_count'] = 0
    else:
        results_df = pd.merge(correct_df, total_df, on='index', how='outer').fillna(0)
    
    # 转换列类型为整数
    results_df['correct_count'] = results_df['correct_count'].astype(int)
    results_df['total_count'] = results_df['total_count'].astype(int)
    
    results_df = results_df.sort_values(by='index').reset_index(drop=True)

    print("\n--- Final Combined Results ---")
    print(results_df)

    # 将结果保存到 CSV 文件
    output_path = config.data.get('output_path', 'evaluation_counts_combined.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults have been successfully saved to {output_path}")


if __name__ == "__main__":
    main()
