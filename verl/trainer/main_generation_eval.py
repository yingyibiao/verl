# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv
import os
import json
import math
import random
import concurrent.futures
from functools import partial
from multiprocessing import get_context   # <<<<<< 新增

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tabulate import tabulate
from transformers import AutoTokenizer  # noqa
from timeout_decorator import timeout, TimeoutError as TimeoutException

from verl import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score.math_verify import compute_score

# 环境变量
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'


# --------------------------------------------------------------------------- #
#                                 入口函数                                    #
# --------------------------------------------------------------------------- #
@hydra.main(config_path="config", config_name="generation_eval", version_base=None)
def main(config):
    from pprint import pprint

    # 展示 Hydra 配置
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # ---------- 模型与分词器 ----------
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # ---------- 若结果已存在则跳过生成 ----------
    if os.path.exists(config.data.output_path):
        print(
            f"Output file {config.data.output_path} already exists. "
            "Skipping generation and proceeding to evaluation."
        )
        if config.data.output_path.endswith(".pkl"):
            dataset = pd.read_pickle(config.data.output_path)
            if not isinstance(dataset, pd.DataFrame):
                dataset = pd.DataFrame(dataset)
        else:
            dataset = pd.read_parquet(config.data.output_path)
    else:
        # ----------- 读取数据集 (JSONL / Parquet / Pickle) -----------
        if config.data.path.endswith(".pkl"):
            dataset = pd.read_pickle(config.data.path)
            if not isinstance(dataset, pd.DataFrame):
                dataset = pd.DataFrame(dataset)
        elif config.data.path.endswith(".jsonl"):
            dataset = [json.loads(x) for x in open(config.data.path)]
            if not isinstance(dataset, pd.DataFrame):
                dataset = pd.DataFrame(dataset)
        else:
            dataset = pd.read_parquet(config.data.path)

        if config.rollout.temperature == 0.0:
            assert (
                config.data.n_samples == 1
            ), "When temperature=0, n_samples must be 1."

        chat_lst = dataset[config.data.prompt_key].tolist()
        chat_lst = [
            (chat.tolist() if not isinstance(chat, list) else chat)
            for chat in chat_lst
        ]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ---------- Ray 初始化 ----------
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout"
        )
        resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
        )
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        # ---------- 逐 batch 生成 ----------
        total_samples = len(dataset)
        cfg_bs = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // cfg_bs) + 1
        output_lst = []

        for batch_idx in range(num_batch):
            print(f"[{batch_idx+1}/{num_batch}] Start to process.")

            batch_chat_lst = chat_lst[batch_idx * cfg_bs : (batch_idx + 1) * cfg_bs]
            if not batch_chat_lst:
                break

            # 每条 prompt 重复 n_samples 次
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)

            inputs = tokenizer.apply_chat_template(
                repeated_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = compute_position_id_with_mask(attention_mask)
            data = DataProto.from_dict(
                {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
            )

            real_batch_size = data.batch["input_ids"].shape[0]
            if real_batch_size % dp_size != 0 or real_batch_size % wg.world_size != 0:
                lcm_value = math.lcm(dp_size, wg.world_size)
                adjusted_bs = (real_batch_size // lcm_value + 1) * lcm_value
                dummy_sz = adjusted_bs - real_batch_size
                data = DataProto.concat([data, data[:dummy_sz]])
                print(
                    f"dp_size {dp_size} | real_bs {real_batch_size} -> pad {dummy_sz} dummy rows"
                )

            batch_size = data.batch["input_ids"].shape[0]
            assert (
                batch_size % dp_size == 0
            ), f"batch_size {batch_size} not divisible by dp_size"

            # ---------- 生成 ----------
            print(f"[{batch_idx+1}/{num_batch}] Start to generate.")
            output = wg.generate_sequences(data)[:real_batch_size]

            decoded = tokenizer.batch_decode(
                output.batch["input_ids"][:, -config.rollout.response_length :],
                skip_special_tokens=False,
            )

            # 去除 pad_token
            pad_token = tokenizer.pad_token
            output_lst.extend([t.replace(pad_token, "") for t in decoded])

        # ---------- 结果重排并写回 ----------
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()
        dataset["responses"] = output_lst

        # 保存
        makedirs(os.path.dirname(config.data.output_path), exist_ok=True)
        dataset.to_pickle(config.data.output_path)

    # ------------------------------------------------------------------ #
    #                           评分阶段                                 #
    # ------------------------------------------------------------------ #
    print("Start evaluation ...")
    reward_model_data = dataset[config.data.reward_model_key]
    reward_model_data = [
        x["ground_truth"] for x in reward_model_data for _ in range(config.data.n_samples)
    ]
    flat_outputs = [x for xx in dataset["responses"] for x in xx]

    # ---------------------- 并行评测（多进程） -------------------------
    print("Scoring outputs ...")
    ctx = get_context("fork")       # Linux 下可用；若在 mac/Windows 请改成 "spawn"
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=48, mp_context=ctx) as executor:
        scores = list(executor.map(
            compute_score,
            flat_outputs,
            reward_model_data,
            [0] * len(flat_outputs)           # 第 3 个参数 timeout=0
        ))

    scores = np.array(scores).reshape(-1, config.data.n_samples)

    pass_at_n = (scores.max(-1) == 1).mean()
    pass_at_1 = (scores[:, 0] == 1).mean()
    pass_at_1_avg_sample = (scores == 1).mean()

    # --------- 汇总与存盘 ---------
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        "model_path": config.model.path,
        "dataset": dataset_name,
        "ex_name": os.path.basename(config.data.output_path),
        f"{config.rollout.response_length//1024}K_Pass@1": pass_at_1,
        f"{config.rollout.response_length//1024}K_Pass@1(avg_{config.data.n_samples})": pass_at_1_avg_sample,
        f"{config.rollout.response_length//1024}K_Pass@{config.data.n_samples}": pass_at_n,
    }

    # 将分数写回 dataset
    dataset["score"] = scores.tolist()
    dataset.to_pickle(config.data.output_path)

    # 追加 CSV
    csv_path = os.path.join(os.path.dirname(config.data.output_path), "pass.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # 终端打印表格
    print(tabulate([[k, v] for k, v in row_data.items()], headers=["Metric", "Value"], tablefmt="grid"))


if __name__ == "__main__":
    main()
