import pandas as pd
import re
import json
import logging
import random
from typing import Optional, Dict, Any, List, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(original_path, generated_path):
    print("--- 正在加载数据... ---")
    try:
        original_df = pd.read_parquet(original_path)
        original_df['extracted_index'] = original_df['extra_info'].apply(lambda x: x.get('index'))

        generated_df = pd.read_parquet(generated_path)
        print("--- 数据加载完毕 ---")
        return original_df, generated_df
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 请检查路径。将使用空DataFrame继续。")
        return pd.DataFrame(), pd.DataFrame()


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    从Markdown格式的文本中提取JSON字符串。

    这个函数会查找以 "```json" 开头的代码块，并提取其中的内容。
    结尾的 "```" 是可选的。

    Args:
        text: 包含JSON的原始字符串。

    Returns:
        提取并清理过的JSON字符串，如果未找到匹配项，则返回原始文本。
    """
    # 正则表达式解释:
    # ```json      -> 匹配文字 '```json'
    # \s* -> 匹配零个或多个空白字符（包括空格、制表符、换行符）
    # (.*?)        -> 非贪婪捕获组:
    #                 .   -> 匹配任何字符
    #                 * -> 零次或多次
    #                 ?   -> 非贪婪模式，即匹配尽可能少的字符
    # (?:\s*```)?   -> 可选的非捕获组:
    #                 (?:...) -> 非捕获组，用于分组而不创建反向引用
    #                 \s*```  -> 匹配零个或多个空白字符后跟 '```'
    #                 ?       -> 使整个组成为可选，即出现零次或一次
    # $            -> 匹配字符串的末尾
    #
    # re.DOTALL 标志让 '.' 可以匹配包括换行符在内的所有字符。
    pattern = r"```json\s*(.*?)(?:\s*```)?$"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # group(1) 返回第一个捕获组的内容，也就是我们想要提取的JSON字符串
        return match.group(1).strip()
    else:
        return text


def extract_and_load_json_from_markdown(text: str) -> Union[Dict[str, Any], List[Any], None]:
    """
    一个组合函数：先提取，再加载为Python对象。

    Args:
        text: 包含 Markdown JSON 代码块的原始字符串。

    Returns:
        一个Python字典或列表（如果提取和解析都成功）。
        None（如果找不到或无法解析）。
    """
    json_string = extract_json_from_markdown(text).strip()
    
    try:
        data = json.loads(json_string)
        return data
    except json.JSONDecodeError as e:
        data = repair_and_load_json(json_string)
        if data is not None:
            return data
        print("修复后的JSON字符串仍然无法解析。")
        print("原始JSON字符串:")
        print(json_string)
        print("\n")
        return None


def repair_and_load_json(malformed_string: str) -> Union[Dict[str, Any], List[Any], None]:
    """
    修复并解析包含非法反斜杠转义（如LaTeX）的JSON字符串的最终版本。

    此版本采用“白名单”策略，能精确区分应保留的合法JSON转义和
    应修复的来自LaTeX等的孤立反斜杠，解决了先前版本的所有已知问题。

    Args:
        malformed_string: 包含潜在非法转义字符的原始JSON字符串。

    Returns:
        一个Python字典或列表（如果修复和解析成功），否则为None。
    """
    # 定义一个替换函数，用于 re.sub
    def replacer(match: re.Match) -> str:
        # 正则表达式 r'(\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})|(\\)' 有两个捕获组:
        # 组1: `(\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})` -> 匹配一个完整的、合法的JSON转义序列。
        # 组2: `(\\)`                               -> 匹配一个孤立的、需要被修复的反斜杠。
        # re.sub 会对每一个匹配项调用此函数。

        # 检查组1是否捕获到内容
        if match.group(1):
            # 如果是，说明匹配到了一个合法的转义序列，将其原样返回。
            return match.group(1)
        else:
            # 否则，说明匹配到了一个孤立的反斜杠（在组2中），
            # 需要将其替换为JSON中合法的双反斜杠。
            return '\\\\'

    try:
        # 核心修复步骤：
        # 1. 优先匹配并捕获一个完整的合法转义序列 (\\["\\/bfnrt] 或 \\uXXXX)。
        # 2. 如果匹配不到，则匹配并捕获一个孤立的反斜杠。
        # 3. 将所有匹配项传入 replacer 函数进行处理。
        repaired_string = re.sub(
            r'(\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})|(\\)',
            replacer,
            malformed_string
        )

        return json.loads(repaired_string)

    except json.JSONDecodeError as e:
        # print(f"错误：即使在修复后，字符串仍然无法解析。错误信息: {e}")
        return None
    except Exception as e:
        # print(f"发生未知错误: {e}")
        return None


def analyze_generated_responses(generated_df: pd.DataFrame) -> dict:
    """
    Analyzes the generated responses and categorizes row indices based on content.

    Args:
        generated_df: DataFrame containing the LLM-generated responses.
                       It must have 'extracted_index' and 'responses' columns.

    Returns:
        A dictionary containing sets of indices for each category.
    """
    indices = {
        "incomplete": set(),
        "over_answered": set(),
        "answer_options": {},
        "answer_not_option": {},
        "not_mcp": set(),
        "no_valid_response": set()
    }

    for _, row in generated_df.iterrows():
        # Attempt to parse a valid JSON response from the 5 attempts
        response = None
        for i in range(5):
            res_text = row.get("responses", [])[i]
            if "</think>" in res_text:
                # Extract content after the thinking tag
                content = res_text.split("</think>", 1)[1].strip()
                parsed_json = extract_and_load_json_from_markdown(content)
                if parsed_json:
                    response = parsed_json
                    break  # Use the first valid JSON found
        
        index = row['extracted_index']

        if response is None:
            indices["no_valid_response"].add(index)
            continue
        
        # print(f"Processing index: {index} with response: {response}")
        # Extract details from the valid response
        is_mcp = response["is_question_mcp"]
        q_parts = response["question_parts_num"]
        a_parts = response["answer_parts_num"]
        is_answer_options = response["is_answer_options"]
        all_choice_letters = response["all_choice_letters"]
        all_choices = response["all_choices"]
        answer_choice_letters = response["answer_choice_letters"]
        answer_choices = response["answer_choices"]
        ground_truth = row['reward_model']['ground_truth']

        # --- Categorization Logic ---
        if a_parts < q_parts:
            indices["incomplete"].add(index)
            continue
        
        if a_parts > q_parts:
            indices["over_answered"].add(index)
            continue

        if not is_mcp:
            indices["not_mcp"].add(index)
        else:
            answer_choice_letters = " ".join(answer_choice_letters) if isinstance(answer_choice_letters, list) else answer_choice_letters
            answer_choices = " ".join(answer_choices) if isinstance(answer_choices, list) else answer_choices
            if answer_choice_letters == "" and answer_choices == "":
                continue
            
            if is_answer_options:
                ground_truths = [answer_choice_letters, answer_choices, ground_truth]
                ground_truths = [item for item in ground_truths if item != ""]
                ground_truths = list(dict.fromkeys(ground_truths))
                indices["answer_options"][index] = {'ground_truth': ground_truths, 'style': 'rule'}
            else:
                if len(answer_choice_letters) == 1:
                    answer_choice_letter_append_gt = answer_choice_letters + " " + ground_truth
                    ground_truths = [answer_choice_letters, answer_choices, answer_choice_letter_append_gt, ground_truth]
                else:
                    ground_truths = [answer_choice_letters, answer_choices, ground_truth]
                ground_truths = [item for item in ground_truths if item != ""]
                ground_truths = list(dict.fromkeys(ground_truths))
                indices["answer_not_option"][index] = {'ground_truth': ground_truths, 'style': 'rule'}
            
    # --- Print Summary ---
    print("--- Analysis Summary ---")
    print(f"Total rows analyzed in generated_df: {len(generated_df)}")
    for key, value in indices.items():
        print(f"Indices in category '{key}': {len(value)}")
    print("------------------------\n")
    print()
    return indices


def create_modified_dataframe(original_df: pd.DataFrame, generated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new DataFrame by modifying original_df based on analysis of generated_df.

    This version correctly maps analysis results to the original DataFrame using
    the 'extracted_index' column.

    Rules:
    1. Rows from original_df not present in generated_df are kept.
    2. Rows categorized as 'incomplete' or 'over_answered' are dropped.
    3. Rows categorized as 'not_mcp' are kept without changes.
    4. Rows in 'answer_options' get their 'reward_model' column modified (Rule A).
    5. Rows in 'answer_not_option' get their 'reward_model' column modified (Rule B).
    
    Args:
        original_df: The original DataFrame. It MUST contain an 'extracted_index' column
                     that holds the unique identifiers used in the analysis.
        generated_df: The DataFrame with LLM analysis results.

    Returns:
        A new DataFrame with the rules applied.
    """
    # 1. Analyze the generated responses. The keys in the returned dictionaries
    #    (e.g., in 'incomplete', 'answer_options') correspond to values in
    #    the 'extracted_index' column of original_df.
    analyzed_indices = analyze_generated_responses(generated_df)

    # 2. Create a copy to avoid modifying the original DataFrame in place.
    modified_df = original_df.copy()
    modified_df = modified_df[modified_df['extracted_index'].isin(generated_df['extracted_index'])]
    
    # 3. Identify rows to be dropped based on their 'extracted_index'.
    extracted_indices_to_drop = analyzed_indices["incomplete"].union(analyzed_indices["over_answered"])
    
    # Find the actual DataFrame indices that correspond to the extracted_indices to drop.
    # This is the key change: we use .isin() on the 'extracted_index' column.
    df_indices_to_drop = modified_df[modified_df['extracted_index'].isin(extracted_indices_to_drop)].index
    
    # Drop the identified rows from the DataFrame.
    modified_df.drop(index=df_indices_to_drop, inplace=True)
    modified_df.reset_index(drop=True, inplace=True)

    print(f"Dropped {len(df_indices_to_drop)} rows due to being incomplete or over-answered.\n\n")

    modified_df["reward_model"] = modified_df["reward_model"].apply(
    lambda d: {**d, "ground_truth": [d["ground_truth"]]}
)

    # 4. Modify the 'reward_model' column for the relevant rows.
    modified_count_A = 0
    for extracted_idx, reward_model in analyzed_indices["answer_options"].items():
        target_idx = modified_df.index[modified_df["extracted_index"] == extracted_idx].item()
        # print(target_idx)
        old_reward_model = modified_df.at[target_idx, 'reward_model'].copy()
        modified_df.at[target_idx, 'reward_model'] = reward_model
        modified_count_A += 1
        # sample print
        if random.random() < 0:
            print(f"Modified 'reward_model' for extracted index {extracted_idx}:\r\n"
                    f"from: {old_reward_model}\nto: {reward_model}\n\n")
    
    modified_count_B = 0
    for extracted_idx, reward_model in analyzed_indices["answer_not_option"].items():
        target_idx = modified_df.index[modified_df["extracted_index"] == extracted_idx].item()
        # print(target_idx)
        old_reward_model = modified_df.at[target_idx, 'reward_model'].copy()
        modified_df.at[target_idx, 'reward_model'] = reward_model
        modified_count_B += 1
        # sample print
        if random.random() < 0:
            print(f"Modified 'reward_model' for extracted index {extracted_idx}:\r\n"
                    f"from: {old_reward_model}\nto: {reward_model}\n\n")
    
    print(f"Modified 'reward_model' for {modified_count_A} rows (answer is an option).")
    print(f"Modified 'reward_model' for {modified_count_B} rows (answer not an option).\n\n")
    return modified_df


def main():
    original_path = "/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math.parquet"
    generated_path = "/data/yibiaoy-sandbox/skywork-or1/Qwen3-32B_hard_question_json_shards/"

    original_df, generated_df = load_data(original_path, generated_path)
    print(f"Original DataFrame loaded with {len(original_df)} rows.")
    print(f"Generated DataFrame loaded with {len(generated_df)} rows.\n\n")

    modified_df = create_modified_dataframe(original_df, generated_df)
    print(f"Modified DataFrame created with {len(modified_df)} rows.")
    # drop the 'extracted_index' column as it is no longer needed
    modified_df.drop(columns=['extracted_index'], inplace=True, errors='ignore')
    output_path = "/data/yibiaoy-sandbox/skywork-or1/train_1p5b_math_modified_only_hard_no_index.parquet"
    print(modified_df.loc[5, 'reward_model'])
    modified_df.to_parquet(output_path, index=False)


if __name__ == '__main__':
    main()
