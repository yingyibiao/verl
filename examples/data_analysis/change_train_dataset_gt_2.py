import pandas as pd
import logging
import os  # 导入os模块来处理文件路径和目录
from typing import Optional

# --- 设置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_examples_by_responses(
    original_df_path: str,
    summary_df_path: str,
    min_correct_responses: int,
) -> Optional[pd.DataFrame]:
    """
    根据 summary 文件中的 correct_responses 数量来过滤原始数据。

    这个函数会加载原始数据和摘要数据，然后移除那些在摘要中
    'correct_responses' 值小于指定阈值的样本。

    Args:
        original_df_path (str): 原始 Parquet 文件的路径。
        summary_df_path (str): 包含 'correct_responses' 和 'extracted_index' 的 CSV 摘要文件路径。
        min_correct_responses (int): 'correct_responses' 的最小允许数量。
                                     任何小于此值的样本都将被过滤掉。

    Returns:
        Optional[pd.DataFrame]: 一个被过滤后的新 DataFrame。如果发生错误则返回 None。
    """
    try:
        # 加载数据
        original_df = pd.read_parquet(original_df_path)
        summary_df = pd.read_csv(summary_df_path)
        
        # 提取 index
        if 'extra_info' in original_df.columns and 'extracted_index' not in original_df.columns:
            original_df['extracted_index'] = original_df['extra_info'].apply(
                lambda x: x.get('index') if isinstance(x, dict) else None
            )

    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
        return None
    except Exception as e:
        logging.error(f"加载数据时发生错误: {e}")
        return None

    # 识别需要移除的索引
    indices_to_remove = set(summary_df[summary_df['correct_responses'] < min_correct_responses]['extracted_index'])
    
    if not indices_to_remove:
        logging.warning(f"对于阈值 {min_correct_responses}, 没有找到需要移除的样本。将返回完整的DataFrame。")
        return original_df
        
    logging.info(f"对于阈值 {min_correct_responses}, 找到了 {len(indices_to_remove)} 个样本需要移除。")

    # 过滤
    logging.info(f"过滤前, 原始 DataFrame 中有 {len(original_df)} 行。")
    filtered_df = original_df[~original_df['extracted_index'].isin(indices_to_remove)].copy()
    logging.info(f"过滤后, DataFrame 中还剩下 {len(filtered_df)} 行。")

    return filtered_df


# --- 主执行逻辑 ---
if __name__ == '__main__':
    # --- 1. 配置您的参数 ---
    
    # 输入文件路径
    original_path = "/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_modified_no_index.parquet"
    summary_path = "/data/yibiaoy-sandbox/skywork-or1/qwen3-32b_generation_detailed_summary.csv"
    
    # 指定一个目录来存放所有输出文件
    output_directory = "/data/yibiaoy-sandbox/skywork-or1/train_files"
    
    # ***************************************************************
    # *** 在这里定义您所有想要测试的 correct_responses 阈值 ***
    # 例如，[1, 2, 3] 会生成三个文件：
    # - 一个包含 correct_responses >= 1 的所有样本
    # - 一个包含 correct_responses >= 2 的所有样本
    # - 一个包含 correct_responses >= 3 的所有样本
    thresholds_to_process = [1, 2, 3, 4, 5, 6]
    # ***************************************************************

    # --- 2. 执行处理循环 ---

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_directory, exist_ok=True)
    logging.info(f"所有过滤后的文件将保存在: {output_directory}")

    # 循环处理每个阈值
    for threshold in thresholds_to_process:
        print("-" * 60)
        logging.info(f"正在处理阈值: min_correct_responses = {threshold}")

        # 调用核心函数进行过滤
        filtered_df = filter_examples_by_responses(
            original_df_path=original_path,
            summary_df_path=summary_path,
            min_correct_responses=threshold
        )
        
        # 如果过滤成功且结果不为空，则保存文件
        if filtered_df is not None and not filtered_df.empty:
            # 构建动态的文件名
            # 例如: filtered_data_min_resp_2.parquet
            output_filename = f"train_1p5b_math_modified_gt_{threshold}.parquet"
            output_path = os.path.join(output_directory, output_filename)
            
            # 保存到 Parquet 文件
            filtered_df.drop(columns=['extracted_index'], inplace=True, errors='ignore')
            filtered_df.to_parquet(output_path, index=False)
            logging.info(f"成功保存文件 -> {output_path} (共 {len(filtered_df)} 行)")
        else:
            logging.warning(f"对于阈值 {threshold}, 没有生成有效的 DataFrame，跳过保存。")

    print("-" * 60)
    logging.info("所有处理任务已完成！")
