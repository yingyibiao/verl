# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0.

"""
Parallel generation on a Ray cluster.
Each batch is written to its own Parquet file shard (shard_00000.parquet, …).
Any shard is a *complete* Parquet file, so you can load it at any time.

读取示例：
    from glob import glob
    from datasets import load_dataset
    ds = load_dataset(
        "parquet",
        data_files={"train": glob("/.../qwen3-32b_generation_test/shard_*.parquet")},
    )
"""

import os
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.device import is_cuda_available     # bool 或函数
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ---------- 用 Ray Actor 保证全局 shard 计数器（并发安全） ----------
@ray.remote
class ShardCounter:
    def __init__(self):
        self.i = 0

    def next(self):
        cur = self.i
        self.i += 1
        return cur


# ---------- Hydra 入口 ----------
@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(cfg):
    run_generation(cfg)


def run_generation(cfg):
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=cfg.ray_init.num_cpus,
        )
    ray.get(main_task.remote(cfg))


# ---------- 主任务 ----------
@ray.remote(num_cpus=1)
def main_task(cfg):
    pprint(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.resolve(cfg)

    # -------- tokenizer --------
    local_path = copy_to_local(cfg.model.path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=cfg.data.get("trust_remote_code", False))

    if cfg.rollout.temperature == 0.0:
        assert cfg.data.n_samples == 1, "temperature=0 → n_samples 必须为 1"
    assert cfg.data.n_samples >= 1

    data_df = pd.read_parquet(cfg.data.path)
    all_chats = [row.tolist() for row in data_df[cfg.data.prompt_key]]

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # -------- Generation worker group --------
    _cuda = is_cuda_available() if callable(is_cuda_available) else is_cuda_available
    wg = RayWorkerGroup(
        resource_pool=RayResourcePool([cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes),
        ray_cls_with_init=RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker), config=cfg, role="rollout"
        ),
        device_name="cuda" if _cuda else "npu",
    )
    wg.init_model()

    # -------- 输出目录准备 --------
    # 如果 cfg.data.output_path 形如 ".../xxx.parquet"，去掉扩展名并 + "_shards"
    out_base = cfg.data.output_path
    if out_base.endswith(".parquet"):
        out_base = os.path.splitext(out_base)[0] + "_shards"
    makedirs(out_base, exist_ok=True)
    print(f"→ Shards will be written to dir: {out_base}")

    counter = ShardCounter.remote()   # 全局 shard 计数

    # -------- batch loop --------
    total, bs = len(data_df), cfg.data.batch_size
    num_batches = -(-total // bs)  # ceil

    for b in range(num_batches):
        print(f"\n[Batch {b+1}/{num_batches}]")
        s, e = b * bs, min((b + 1) * bs, total)
        batch_chats = all_chats[s:e]

        # --- Tokenize ---
        inputs = tokenizer.apply_chat_template(
            batch_chats,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=cfg.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        ids, att = inputs["input_ids"], inputs["attention_mask"]
        pos = compute_position_id_with_mask(att)

        data_proto = DataProto.from_dict(
            {"input_ids": ids, "attention_mask": att, "position_ids": pos}
        )
        pad_proto, pad_sz = pad_dataproto_to_divisor(data_proto, wg.world_size)

        # --- n_samples generation ---
        sample_texts, sample_logps = [], []   # n_samples × batch_len
        for _ in range(cfg.data.n_samples):
            out = unpad_dataproto(wg.generate_sequences(pad_proto), pad_sz)
            texts_i, logps_i = [], []
            for item in out:
                p_len = item.batch["prompts"].shape[-1]
                r_len = int(item.batch["attention_mask"][p_len:].sum().item())
                texts_i.append(tokenizer.decode(item.batch["responses"][:r_len], skip_special_tokens=True))
                logps_i.append(item.batch["rollout_log_probs"][:r_len].cpu().tolist())
            sample_texts.append(texts_i)
            sample_logps.append(logps_i)

        # --- Build batch df ---
        df_batch = data_df.iloc[s:e].copy().reset_index(drop=True)
        df_batch["responses"]         = [list(x) for x in zip(*sample_texts)]
        df_batch["rollout_log_probs"] = [list(x) for x in zip(*sample_logps)]

        # --- Write shard (each shard is a complete parquet file) ---
        shard_id = ray.get(counter.next.remote())
        shard_path = os.path.join(out_base, f"shard_{shard_id:05d}.parquet")

        # 直接用 pyarrow 写，一次完成 → 文件立即可读
        pq.write_table(pa.Table.from_pandas(df_batch, preserve_index=False),
                       shard_path,
                       compression="snappy")
        print(f"✓ rows {s}-{e-1} saved → {shard_path}")

    print(f"\nAll shards done. Directory: {out_base}\n")
    print("Example to load later:")
    print("  from glob import glob; from datasets import load_dataset")
    print(f"  ds = load_dataset('parquet', data_files={{'train': glob('{out_base}/shard_*.parquet')}})")
    print("Enjoy!\n")


if __name__ == "__main__":
    main()
