# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0.

"""
Parallel generation on a Ray cluster.
Each batch is written to its own Parquet file shard (shard_00000.parquet, …).
Any shard is a *complete* Parquet file, so you can load it at any time.

This script supports two modes:
1. Fixed Sampling: Generates a fixed `n_samples` for every prompt.
   (Activated when `data.filter_path` is not provided in the config).
2. Dynamic Sampling: Uses a robust, "with insurance" strategy to generate
   a variable number of samples for each prompt based on its `correct_count`.
   This is controlled by `target_total_samples`, `dynamic_sampling_min_samples`,
   and `dynamic_sampling_multiplier`.
   (Activated when `data.filter_path` is provided).

Example to read the output:
    from glob import glob
    from datasets import load_dataset
    ds = load_dataset(
        "parquet",
        data_files={"train": glob("/.../my_generation_output_shards/shard_*.parquet")},
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
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ---------- Ray Actor for a thread-safe global shard counter ----------
@ray.remote
class ShardCounter:
    def __init__(self):
        self.i = 0

    def next(self):
        cur = self.i
        self.i += 1
        return cur


# ---------- Hydra entry point ----------
@hydra.main(config_path="config", config_name="generation_dynamics", version_base=None)
def main(cfg):
    run_generation(cfg)


def run_generation(cfg):
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=cfg.ray_init.num_cpus,
        )
    ray.get(main_task.remote(cfg))


# ---------- Main task running on Ray ----------
@ray.remote(num_cpus=1)
def main_task(cfg):
    pprint(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.resolve(cfg)

    # -------- Tokenizer setup --------
    local_path = copy_to_local(cfg.model.path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=cfg.data.get("trust_remote_code", False))
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # -------- Load data --------
    data_df = pd.read_parquet(cfg.data.path)
    data_df['extracted_index'] = data_df['extra_info'].apply(lambda x: x.get('index'))

    # -------- Prepare output directory --------
    out_base = cfg.data.output_path
    if out_base.endswith(".parquet"):
        out_base = os.path.splitext(out_base)[0] + "_shards"
    makedirs(out_base, exist_ok=True)
    print(f"→ Shards will be written to dir: {out_base}")
    counter = ShardCounter.remote()

    # -------- Generation worker group setup --------
    _cuda = is_cuda_available() if callable(is_cuda_available) else is_cuda_available
    wg = RayWorkerGroup(
        resource_pool=RayResourcePool([cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes),
        ray_cls_with_init=RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker), config=cfg, role="rollout"
        ),
        device_name="cuda" if _cuda else "npu",
    )
    wg.init_model()

    # -------- Main logic dispatcher --------
    if cfg.data.get("filter_path"):
        print(f"INFO: filter_path '{cfg.data.filter_path}' found. Running in DYNAMIC sampling mode.")
        run_dynamic_sampling_generation(cfg, data_df, tokenizer, wg, counter, out_base)
    else:
        print("INFO: filter_path not found. Running in FIXED sampling mode.")
        run_fixed_sampling_generation(cfg, data_df, tokenizer, wg, counter, out_base)

    print(f"\nAll shards done. Directory: {out_base}\n")
    print("Example to load later:")
    print("  from glob import glob; from datasets import load_dataset")
    print(f"  ds = load_dataset('parquet', data_files={{'train': glob('{out_base}/shard_*.parquet')}})")
    print("Enjoy!\n")


def run_fixed_sampling_generation(cfg, data_df, tokenizer, wg, counter, out_base):
    """
    Runs generation with a fixed number of samples for every prompt.
    """
    bs = cfg.data.batch_size
    total = len(data_df)
    num_batches = -(-total // bs)  # Ceil division

    for b in range(num_batches):
        print(f"\n[Batch {b+1}/{num_batches}]")
        s, e = b * bs, min((b + 1) * bs, total)
        df_slice = data_df.iloc[s:e]

        df_batch = generate_for_batch(
            cfg=cfg,
            df_slice=df_slice,
            n_samples=cfg.data.n_samples,
            tokenizer=tokenizer,
            wg=wg
        )

        shard_id = ray.get(counter.next.remote())
        shard_path = os.path.join(out_base, f"shard_{shard_id:05d}.parquet")
        pq.write_table(pa.Table.from_pandas(df_batch, preserve_index=False), shard_path, compression="snappy")
        print(f"✓ Rows {s}-{e-1} saved → {shard_path}")


def run_dynamic_sampling_generation(cfg, data_df, tokenizer, wg, counter, out_base):
    """
    Runs generation with a dynamic number of samples based on a robust, "with insurance" logic.
    """
    filter_df = pd.read_csv(cfg.data.filter_path)

    # 1. Merge DataFrame to get correct_count for each prompt
    merged_df = pd.merge(data_df, filter_df, left_on='extracted_index', right_on='index', how='left')
    merged_df['correct_count'].fillna(0, inplace=True)
    merged_df['correct_count'] = merged_df['correct_count'].astype(int)

    # 2. Calculate samples to generate using the robust "with insurance" logic
    target_total = cfg.data.get("target_total_samples", 6)
    min_samples = cfg.data.get("dynamic_sampling_min_samples", 3)
    multiplier = cfg.data.get("dynamic_sampling_multiplier", 1.5)

    print(f"INFO: Target total samples: {target_total}")
    print(f"INFO: Dynamic sampling minimum: {min_samples}, Multiplier: {multiplier}")

    # Step 1: Calculate the base number of samples needed
    needed_samples = target_total - merged_df['correct_count']
    
    # Step 2: Apply the multiplier and take the ceiling
    multiplied_samples = np.ceil(needed_samples * multiplier)

    # Step 3: Ensure the number of samples is not less than the minimum safety net
    final_samples_to_gen = np.maximum(multiplied_samples, min_samples)

    # Assign final value, but filter based on the original need
    merged_df['samples_to_generate'] = final_samples_to_gen.astype(int)
    prompts_to_process_df = merged_df[needed_samples > 0].copy()

    if prompts_to_process_df.empty:
        print("INFO: All prompts have reached the target sample count. Nothing to generate.")
        return

    # 3. Group prompts by the number of samples they need
    unique_sample_counts = sorted(prompts_to_process_df['samples_to_generate'].unique(), reverse=True)
    bs = cfg.data.batch_size

    # 4. Iterate through each group and generate
    for n_samples in unique_sample_counts:
        group_df = prompts_to_process_df[prompts_to_process_df['samples_to_generate'] == n_samples]
        total_group = len(group_df)
        num_batches_group = -(-total_group // bs)

        print(f"\n---------- Processing group needing {n_samples} sample(s) each ({total_group} prompts) ----------")

        for b in range(num_batches_group):
            print(f"\n[Group(n={n_samples}), Batch {b+1}/{num_batches_group}]")
            s, e = b * bs, min((b + 1) * bs, total_group)
            df_slice = group_df.iloc[s:e]

            df_batch = generate_for_batch(
                cfg=cfg,
                df_slice=df_slice,
                n_samples=n_samples,
                tokenizer=tokenizer,
                wg=wg
            )

            shard_id = ray.get(counter.next.remote())
            original_indices = df_slice.index.to_list()
            shard_path = os.path.join(out_base, f"shard_{shard_id:05d}.parquet")
            pq.write_table(pa.Table.from_pandas(df_batch, preserve_index=False), shard_path, compression="snappy")
            print(f"✓ Prompts with original indices {original_indices} saved → {shard_path}")


def generate_for_batch(cfg, df_slice, n_samples, tokenizer, wg):
    """
    Core generation logic for a given slice of a DataFrame (a batch).
    """
    if n_samples <= 0 or df_slice.empty:
        return pd.DataFrame()

    batch_chats = [row.tolist() for row in df_slice[cfg.data.prompt_key]]

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

    data_proto = DataProto.from_dict({"input_ids": ids, "attention_mask": att, "position_ids": pos})
    pad_proto, pad_sz = pad_dataproto_to_divisor(data_proto, wg.world_size)

    # --- n_samples generation loop ---
    sample_texts, sample_logps = [], []
    for i in range(n_samples):
        print(f"  Generating sample {i+1}/{n_samples}...")
        out = unpad_dataproto(wg.generate_sequences(pad_proto), pad_sz)
        texts_i, logps_i = [], []
        for item in out:
            p_len = item.batch["prompts"].shape[-1]
            r_len = int(item.batch["attention_mask"][p_len:].sum().item())
            texts_i.append(tokenizer.decode(item.batch["responses"][:r_len], skip_special_tokens=True))
            logps_i.append(item.batch["rollout_log_probs"][:r_len].cpu().tolist())
        sample_texts.append(texts_i)
        sample_logps.append(logps_i)

    # --- Build batch DataFrame ---
    df_batch = df_slice.copy().reset_index(drop=True)
    df_batch["responses"] = [list(x) for x in zip(*sample_texts)]
    df_batch["rollout_log_probs"] = [list(x) for x in zip(*sample_logps)]

    return df_batch


if __name__ == "__main__":
    main()