import pandas as pd
import json

data_df = pd.read_parquet("/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math.parquet")
data_df['extracted_index'] = data_df['extra_info'].apply(lambda x: x.get('index'))


csv_df = pd.read_csv("/code/yibiaoy-sandbox/verl/outputs/offline_generation/evaluation_correct_counts_combined.csv")
indices_of_interest = csv_df[
    (csv_df['correct_count'] == 0) | (csv_df['correct_count'] == 1) | (csv_df['correct_count'] == 2) | (csv_df['correct_count'] == 3)
]['index']
print("hard question indices:", len(indices_of_interest))


classification_df = pd.read_parquet("/data/yibiaoy-sandbox/skywork-or1/Qwen3-4B_hard_question_classification_shards/shard_00000.parquet")
classification_df["responses"] = classification_df["responses"].apply(lambda x: x.tolist() if isinstance(x, (list, pd.Series)) else [str(x)])
is_mcq_mask = classification_df["responses"].apply(
    lambda res_list: isinstance(res_list, list) and all("True" in item or "true" in item for item in res_list)
)
mcq_indices = set(classification_df.loc[is_mcq_mask, 'extracted_index'])
print("MCQ indices from classification:", len(mcq_indices))


data_df_filtered = data_df[data_df["extracted_index"].isin(indices_of_interest)].copy()


def modify_prompt_with_reward(row):
    """
    修改 prompt 的函数，会同时使用原始的 prompt 和 reward_model 列的内容。
    
    Args:
        row (pd.Series): DataFrame 的一行数据。
        
    Returns:
        list: 修改后的 prompt 列表。
    """
    # 从行中获取 'prompt' 和 'reward_model' 的内容
    prompt_obj = row['prompt']
    reward_model_output = row['reward_model']['ground_truth']

    # 步骤 1: 统一转换为 list
    prompt_list = list(prompt_obj)

    # 步骤 2: 在 list 的基础上进行判断和操作
    if not prompt_list:
        return prompt_list  # 返回一个空的 list

    first_item = prompt_list[0]
    if isinstance(first_item, dict) and 'content' in first_item:
        first_item_copy = first_item.copy()
        
        # 清理原始的 prompt 内容
        original_content = first_item_copy['content']
        cleaned_content = original_content.replace("Let's think step by step and output the final answer within \\boxed{}.", "").strip()
        
        # --- 这是核心修改部分 ---
        # 定义新的指示语
        new_instruction = ("I'll give you a multiple-choice question and the corresponding reference answer. "
                            "Your task is to output a json object with the following 3 keys: \n"
                            "1. is_mcp_question (boolean): True if the question is a multiple-choice question, False otherwise. \n"
                            "2. is_answer_options (boolean): True for reference answer being choice option letter(s) or option numbers, False for being the actual solution text. \n"
                            "3. all_choice_options (list): a list of choice letters of all options in the question. \n"
                            "4. answer_letters (list): a list of choice letters corresponding to the reference answer. \n"
                            "Only output the required json.\n\n")
        
        new_instruction = ("I will provide you with a Question and its corresponding Reference Answer.\n"
                            "Your task is to precisely analyze this information and generate a single JSON object with the following schema.\n"
                            "JSON Output Schema:\n"
                            "You must output a JSON object containing these keys:\n"
                            "1. is_question_mcp (boolean): true if the 'Question' is a multiple-choice question (i.e., it presents distinct options like A, B, C, 1, 2, 3), otherwise false.\n"
                            "2. question_parts_num (integer): The number of distinct parts in the 'Question'. If the question is a single part, return 1.\n"
                            "3. answer_parts_num (integer): The number of distinct parts in the 'Reference Answer'. If the answer is a single part, return 1.\n"
                            "4. is_answer_options (boolean): true if the 'Reference Answer' consists of option(s) (e.g., A, B, C, 1, 2, 3) from the 'Question'. false if the 'Reference Answer' is the actual solution text (e.g., Paris or $x=5$).\n"
                            "5. all_choice_letters (list of strings): A list containing all the choice option letters available in the question. Only choice letters, don't include actual solution texts. If it is not a multiple-choice question, return an empty list [].\n"
                            "6. all_choices (list of strings): A list containing all the choice options available in the question. Include actual solution texts after each choice letters. If it is not a multiple-choice question, return an empty list [].\n"
                            "7. answer_choice_letters (list of strings): A list containing the choice option letter(s) that correspond to the 'Reference Answer'. Don't include actual solution texts. If the 'Question' is not a multiple-choice question, return an empty list [].\n"
                            "8. answer_choices (list of strings): A list containing the choice option(s) that correspond to the 'Reference Answer'. Include actual solution texts after each choice letters. If the 'Question' is not a multiple-choice question, return an empty list [].\n\n")
        
        
        # 将指示语、清理后的问题、以及 reward_model 的输出组合成新的内容
        # 你可以根据你的具体需求调整下面的 f-string 格式
        first_item_copy['content'] = (
            f"{new_instruction}"
            f"Question: {cleaned_content}\n\n"
            f"Reference Answer: {reward_model_output}"
        )

        # 返回包含修改后内容的新的 list
        return [first_item_copy] + prompt_list[1:]
    else:
        # 如果不满足条件，返回原始 list
        return prompt_list


data_df_filtered['prompt'] = data_df_filtered.apply(modify_prompt_with_reward, axis=1)

# 保存修改后的 DataFrame
output_parquet_path = "/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_hard_json.parquet"
data_df_filtered.to_parquet(output_parquet_path, index=False)

print(f"处理完成，过滤后的数据已保存到: {output_parquet_path}")
print(f"过滤后的数据框大小: {data_df_filtered.shape}")
# 打印一个样本查看结果
print("\n一个修改后的 prompt 示例:")
print(data_df_filtered['prompt'].iloc[0])
print(data_df_filtered['prompt'].iloc[1])
print(data_df_filtered['prompt'].iloc[2])
