import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 文件路径和对应的标签
files_info = {
    '/data/yibiaoy-sandbox/skywork-or1/qwen3-32b_generation_detailed_summary.csv': 'Qwen3-32B',
    '/data/yibiaoy-sandbox/skywork-or1/distill_qwen_1p5b_generation_detailed_summary.csv': 'Distill Qwen 1.5B',
    '/data/yibiaoy-sandbox/skywork-or1/distill_qwen_1p5b_800steps_generation_detailed_summary.csv': 'Distill Qwen 1.5B (800 steps)'
}

# 读取数据
data_dict = {}
for file_path, label in files_info.items():
    try:
        df = pd.read_csv(file_path)
        data_dict[label] = df['avg_token_length'].values
        print(f"成功读取 {label}: {len(df)} 条记录")
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"读取 {file_path} 时出错: {e}")

# 创建输出目录
output_dir = "/data/yibiaoy-sandbox/skywork-or1/plots"
os.makedirs(output_dir, exist_ok=True)

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 颜色设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 图1: 直方图对比
for i, (label, data) in enumerate(data_dict.items()):
    ax1.hist(data, bins=30, alpha=0.7, label=label, color=colors[i], density=True)

ax1.set_xlabel('Average Token Length')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Average Token Length - Histogram')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 核密度估计曲线
for i, (label, data) in enumerate(data_dict.items()):
    sns.kdeplot(data=data, label=label, ax=ax2, color=colors[i], linewidth=2)

ax2.set_xlabel('Average Token Length')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of Average Token Length - Density Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片 - 多种格式
plt.savefig(f"{output_dir}/avg_token_length_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/avg_token_length_comparison.pdf", bbox_inches='tight')
plt.savefig(f"{output_dir}/avg_token_length_comparison.svg", bbox_inches='tight')

print(f"\n图片已保存到 {output_dir}/ 目录下")
print("保存格式: PNG (高清), PDF (矢量), SVG (矢量)")

plt.show()

# === 单独的密度曲线图 ===
plt.figure(figsize=(10, 6))

for i, (label, data) in enumerate(data_dict.items()):
    sns.kdeplot(data=data, label=label, color=colors[i], linewidth=2.5)

plt.xlabel('Average Token Length', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution Comparison of Average Token Length', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存单独的密度图
plt.savefig(f"{output_dir}/avg_token_length_density_only.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/avg_token_length_density_only.pdf", bbox_inches='tight')

print("密度曲线图也已单独保存")
plt.show()

# 打印统计信息
print("\n=== 统计信息 ===")
for label, data in data_dict.items():
    print(f"\n{label}:")
    print(f"  样本数量: {len(data)}")
    print(f"  平均值: {np.mean(data):.2f}")
    print(f"  标准差: {np.std(data):.2f}")
    print(f"  最小值: {np.min(data):.2f}")
    print(f"  最大值: {np.max(data):.2f}")
    print(f"  中位数: {np.median(data):.2f}")