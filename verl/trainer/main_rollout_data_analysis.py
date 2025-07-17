from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor
import gc
import os

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
from omegaconf import ListConfig

from tokenizers import Tokenizer
from transformers import AutoTokenizer

try:
    from tokenizers import Tokenizer
    FAST_TOKENIZER_AVAILABLE = True
    print("Using fast tokenizers library (Rust implementation)")
except ImportError:
    from transformers import AutoTokenizer
    FAST_TOKENIZER_AVAILABLE = False
    print("Using transformers library (fallback)")

FAST_TOKENIZER_AVAILABLE = False
from verl.utils.reward_score import default_compute_score

# 固定参数
BATCH_SIZE = 50          # 每批处理的index数量
MAX_IO_WORKERS = 20       # 并行读取文件的线程数
TOKENIZER_BATCH_SIZE = 200  # tokenizer批量处理大小


@ray.remote
class FastTokenizerActor:
    """Ray Actor使用快速tokenizer，避免重复加载"""
    
    def __init__(self, model_name="Qwen/Qwen3-32B"):
        print(f"Loading tokenizer in actor: {model_name}...")
        
        if FAST_TOKENIZER_AVAILABLE:
            # 尝试使用fast tokenizer
            try:
                # 首先尝试从HuggingFace hub加载tokenizer.json
                from huggingface_hub import hf_hub_download
                tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                self.is_fast = True
                print("Loaded fast tokenizer from tokenizer.json")
            except Exception as e:
                print(f"Failed to load fast tokenizer: {e}")
                # 回退到transformers
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.is_fast = False
                print("Fallback to transformers tokenizer")
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.is_fast = False
        
        print("Tokenizer loaded successfully in actor")
    
    def batch_encode(self, texts, batch_size=TOKENIZER_BATCH_SIZE):
        """批量编码文本，返回token长度列表"""
        if not texts:
            return []
        
        lengths = []
        
        if self.is_fast:
            # 使用fast tokenizer批量处理
            try:
                # 分批处理避免内存问题
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    # 使用fast tokenizer的批量编码
                    encodings = self.tokenizer.encode_batch(batch_texts)
                    for encoding in encodings:
                        lengths.append(len(encoding.ids))
                return lengths
            except Exception as e:
                print(f"Fast tokenizer batch encoding failed: {e}")
                # 回退到单个编码
                for text in texts:
                    encoding = self.tokenizer.encode(text)
                    lengths.append(len(encoding.ids))
                return lengths
        else:
            # 使用transformers tokenizer
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                # 使用批量编码
                batch_encodings = self.tokenizer(
                    batch_texts, 
                    truncation=False, 
                    padding=False, 
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )
                # 获取每个文本的长度
                for encoding in batch_encodings['input_ids']:
                    lengths.append(len(encoding))
            
            return lengths


@ray.remote
def process_responses_batch(reward_fn, batch_data, tokenizer_actor):
    """
    批量处理多个index的responses
    """
    results = {}
    
    for index, aggregated_data in batch_data.items():
        # 合并所有responses
        responses = []
        rollout_log_probs = []
        data_source = aggregated_data['data_source']
        prompt = aggregated_data['prompt']
        ability = aggregated_data['ability']
        reward_model = aggregated_data['reward_model']
        ground_truth = reward_model["ground_truth"]
        extra_info = aggregated_data['extra_info']
        
        # 收集所有responses
        for responses_list in aggregated_data['responses_lists']:
            responses.extend(responses_list)
        
        for log_probs_list in aggregated_data['rollout_log_probs_lists']:
            rollout_log_probs.extend(log_probs_list)
        
        assert len(responses) == len(rollout_log_probs)

        response_lengths = ray.get(tokenizer_actor.batch_encode.remote(responses))
        
        valid_responses = []
        valid_rollout_log_probs = []
        valid_response_lengths = []
        for response, rollout_log_prob, response_length in zip(responses, rollout_log_probs, response_lengths):
            if len(rollout_log_prob) == response_length + 1 and response.startswith("<think>\n"):
                valid_responses.append(response)
                valid_rollout_log_probs.append(rollout_log_prob)
                valid_response_lengths.append(response_length)

        # 批量计算正确性
        valid_response_correctness = []
        for response in valid_responses:
            is_correct = reward_fn(data_source, response, ground_truth) == 1.0
            valid_response_correctness.append(is_correct)

        results[index] = {
            'responses': valid_responses,
            'rollout_log_probs': valid_rollout_log_probs,
            'response_lengths': valid_response_lengths,
            'response_correctness': valid_response_correctness,
            'data_source': data_source,
            'prompt': prompt,
            'ability': ability,
            'reward_model': reward_model,
            'extra_info': extra_info,
        }
    
    return results


def read_parquet_file(file_path):
    """读取单个parquet文件"""
    try:
        df = pd.read_parquet(file_path)
        print(f"✓ Read {file_path}: {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        return file_path, df
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
        return file_path, None


def parallel_read_files(file_paths):
    """并行读取多个文件"""
    max_workers = min(len(file_paths), MAX_IO_WORKERS)
    print(f"Reading {len(file_paths)} files in parallel (max_workers={max_workers})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_parquet_file, path) for path in file_paths]
        results = []
        
        for future in tqdm(futures, desc="Reading files"):
            file_path, df = future.result()
            if df is not None:
                results.append((file_path, df))
            else:
                print(f"Skipping failed file: {file_path}")
    
    return results


def optimized_aggregate_dataframes(file_results):
    """
    优化的聚合函数，减少内存拷贝
    """
    print("Aggregating data by index...")
    
    aggregated_data = defaultdict(lambda: {
        'responses_lists': [],
        'rollout_log_probs_lists': [],
        'data_source': None,
        'prompt': None,
        'ability': None,
        'reward_model': None,
        'extra_info': None
    })
    
    total_rows_processed = 0
    duplicate_indices = set()
    
    for file_path, df in file_results:
        print(f"Processing {file_path} with {len(df)} rows...")
        
        # 批量处理行，减少循环开销
        for i in range(len(df)):
            row = df.iloc[i]
            
            # 提取index
            if isinstance(row['extra_info'], dict) and 'index' in row['extra_info']:
                index = row['extra_info']['index']
            else:
                print(f"Warning: No index found in extra_info for row {i} in {file_path}")
                continue
            
            # 检查是否已经处理过这个index
            if aggregated_data[index]['data_source'] is not None:
                duplicate_indices.add(index)
            
            # 聚合数据
            agg_data = aggregated_data[index]
            agg_data['responses_lists'].append(row['responses'])
            agg_data['rollout_log_probs_lists'].append(row['rollout_log_probs'])
            
            # 设置元数据（只在第一次遇到时设置）
            if agg_data['data_source'] is None:
                agg_data['data_source'] = row['data_source']
                agg_data['prompt'] = row['prompt']
                agg_data['ability'] = row['ability']
                agg_data['reward_model'] = row['reward_model']
                agg_data['extra_info'] = row['extra_info']
            
            total_rows_processed += 1
        
        # 释放内存
        del df
        gc.collect()
    
    print(f"Processed {total_rows_processed} total rows")
    print(f"Aggregated {len(aggregated_data)} unique indices")
    if duplicate_indices:
        print(f"Found {len(duplicate_indices)} indices with data from multiple files")
    
    return dict(aggregated_data)


def create_batches(data_dict, batch_size=BATCH_SIZE):
    """将数据分批，用于并行处理"""
    items = list(data_dict.items())
    batches = []
    
    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i + batch_size])
        batches.append(batch)
    
    print(f"Created {len(batches)} batches (batch_size={batch_size})")
    return batches


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    # 解析文件路径
    if isinstance(config.data.path, (ListConfig, list)):
        file_paths = list(config.data.path)
    else:
        file_paths = [config.data.path]

    print(f"🚀 Processing {len(file_paths)} parquet file(s)...")
    print(f"📊 Configuration: batch_size={BATCH_SIZE}, max_io_workers={MAX_IO_WORKERS}")
    
    # 并行读取文件
    file_results = parallel_read_files(file_paths)
    
    if not file_results:
        print("❌ No files successfully read!")
        return
    
    # 优化的数据聚合
    aggregated_data = optimized_aggregate_dataframes(file_results)
    print(f"📊 Aggregated data contains {len(aggregated_data)} unique indices\n")

    # prompt_df = pd.read_parquet("/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_modified.parquet")
    prompt_df = pd.read_parquet("/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_modified_only_hard.parquet")
    valid_indices = set(prompt_df['extracted_index'])
    aggregated_data = {key: value for key, value in aggregated_data.items() if key in valid_indices}
    print(f"📊 After filtering, aggregated data contains {len(aggregated_data)} unique indices\n")

    reward_model_mapping = prompt_df.set_index('extracted_index')['reward_model'].to_dict()
    for index, data_dict in aggregated_data.items():
        # 检查 key 是否在映射中，然后更新
        if index in reward_model_mapping:
            data_dict['reward_model'] = reward_model_mapping[index]

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # 创建tokenizer actors
    num_tokenizer_actors = 20  # 减少actor数量，给worker留更多资源
    print(f"🤖 Creating {num_tokenizer_actors} tokenizer actors...")
    
    tokenizer_actors = [FastTokenizerActor.remote() for _ in range(num_tokenizer_actors)]
    
    # 等待所有tokenizer加载完成
    ray.get([actor.batch_encode.remote([]) for actor in tokenizer_actors])
    print("✅ All tokenizer actors ready!")

    # compute_score函数
    compute_score = default_compute_score

    # 将数据分批处理
    data_batches = create_batches(aggregated_data)

    # 创建远程任务
    remote_tasks = []
    actor_idx = 0
    
    for batch in data_batches:
        # 循环使用tokenizer actors
        tokenizer_actor = tokenizer_actors[actor_idx % len(tokenizer_actors)]
        task = process_responses_batch.remote(compute_score, batch, tokenizer_actor)
        remote_tasks.append(task)
        actor_idx += 1

    # 收集结果
    all_results = {}
    
    with tqdm(total=len(data_batches), desc="🔄 Processing batches") as pbar:
        while remote_tasks:
            done_ids, remote_tasks = ray.wait(remote_tasks, num_returns=min(len(remote_tasks), 3))
            
            for result_id in done_ids:
                batch_results = ray.get(result_id)
                all_results.update(batch_results)
                pbar.update(1)

    print("\n📝 Creating final unified dataset...")
    
    # 创建最终的DataFrame
    final_data = []
    
    for index in sorted(all_results.keys()):
        result = all_results[index]

        row_data = {
            'extracted_index': index,
            'data_source': result['data_source'],
            'prompt': result['prompt'],
            'ability': result['ability'],
            'reward_model': result['reward_model'],
            'extra_info': result['extra_info'],
            'responses': result['responses'],
            'rollout_log_probs': result['rollout_log_probs'],
            'response_lengths': result['response_lengths'],
            'response_correctness': result['response_correctness']
        }
        
        # 验证长度一致性
        responses_len = len(row_data['responses'])
        lengths_len = len(row_data['response_lengths'])
        correctness_len = len(row_data['response_correctness'])
        rollout_len = len(row_data['rollout_log_probs'])
        assert responses_len == lengths_len == correctness_len == rollout_len
        final_data.append(row_data)
    
    # 创建DataFrame
    final_columns = ['extracted_index', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info',
                    'responses', 'rollout_log_probs', 'response_lengths', 'response_correctness']
    
    final_dataset = pd.DataFrame(final_data, columns=final_columns)
    
    # 打印统计信息
    print(f"\n📊 Final dataset shape: {final_dataset.shape}")
    print(f"📋 Final columns: {list(final_dataset.columns)}")
    
    # 计算和打印各种统计信息
    print("\n--- 📈 Processing Statistics ---")
    total_responses = sum(len(row['responses']) for _, row in final_dataset.iterrows())
    
    print(f"🎯 Total unique indices: {len(final_dataset)}")
    print(f"💬 Total responses: {total_responses}")
    
    # 计算token长度统计
    all_lengths = []
    for _, row in final_dataset.iterrows():
        all_lengths.extend(row['response_lengths'])
    
    if all_lengths:
        print(f"📏 Average token length: {np.mean(all_lengths):.2f}")
        print(f"📏 Min token length: {min(all_lengths)}")
        print(f"📏 Max token length: {max(all_lengths)}")
    
    # 保存结果
    output_path = config.data.get('output_path', 'unified_aggregated_dataset.parquet')
    print(f"\n💾 Saving unified and aggregated dataset to {output_path}...")
    final_dataset.to_parquet(output_path, index=False)
    print(f"✅ Unified dataset saved successfully!")
    
    # 创建汇总统计文件
    print("📊 Creating summary statistics...")
    summary_data = []
    for _, row in final_dataset.iterrows():
        row_lengths = row['response_lengths']
        row_correctness = row['response_correctness']
        
        row_data = {
            'extracted_index': row['extracted_index'],
            'total_responses': len(row_lengths),
            'correct_responses': sum(row_correctness),
            'avg_token_length': np.mean(row_lengths) if row_lengths else 0,
            'min_token_length': min(row_lengths) if row_lengths else 0,
            'max_token_length': max(row_lengths) if row_lengths else 0,
        }
        
        # 计算正确和错误回答的平均token长度
        correct_lengths_row = [row_lengths[j] for j in range(len(row_lengths)) 
                              if j < len(row_correctness) and row_correctness[j]]
        wrong_lengths_row = [row_lengths[j] for j in range(len(row_lengths))
                            if j < len(row_correctness) and not row_correctness[j]]
        row_data['correct_avg_token_length'] = np.mean(correct_lengths_row) if correct_lengths_row else 0
        row_data['wrong_avg_token_length'] = np.mean(wrong_lengths_row) if wrong_lengths_row else 0
        summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = output_path.replace('.parquet', '_summary.csv')
    summary_df.to_csv(summary_output_path, index=False)
    print(f"📊 Summary statistics saved to {summary_output_path}")

    # 清理资源
    ray.shutdown()
    
    print(f"\n🎉 Processing completed!")
    print(f"📁 Unified dataset: {output_path}")
    print(f"📊 Summary statistics: {summary_output_path}")
    
    # 性能提示
    if FAST_TOKENIZER_AVAILABLE:
        print("🚀 Used fast tokenizers (Rust implementation) for better performance!")
    else:
        print("💡 Install tokenizers library for even better performance: pip install tokenizers")


if __name__ == "__main__":
    main()