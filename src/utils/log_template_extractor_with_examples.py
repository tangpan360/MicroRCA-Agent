import os
import sys
import pandas as pd
import json
import random
from drain.drain_template_extractor import *
from tqdm import tqdm  # 用于显示进度条

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from utils.io_util import load, save
from utils import io_util


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取当前文件的上层目录
error_logs_path = os.path.join(base_dir, 'data', 'phaseone', 'processed', 'error_logs.parquet')
print(f"项目根目录: {base_dir}")
print(f"错误日志路径: {error_logs_path}")

# 读取错误日志数据
print("读取错误日志数据...")
df_error_logs = pd.read_parquet(error_logs_path)
print(f"共读取 {len(df_error_logs)} 条错误日志")

# 收集日志消息
error_logs = df_error_logs['message'].values.tolist()

# 加载已训练的Drain模型
print("加载Drain模型...")
miner = io_util.load('drain/error_log-drain.pkl')

# 对聚类结果按大小降序排序
sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)
print(f"共有 {len(sorted_clusters)} 个模板")

# 准备模板统计信息
template_ids = []      # 存储模板ID
template_counts = []   # 存储模板出现次数  
templates = []         # 存储模板内容
template_examples = [] # 存储每个模板的实例示例

# 创建一个字典，用于存储每个日志消息对应的模板ID
message_to_template = {}
for log_message in tqdm(error_logs, desc="匹配日志模板"):
    cluster = miner.match(log_message)
    if cluster:
        message_to_template[log_message] = cluster.cluster_id

# 遍历排序后的聚类结果
print("处理聚类结果...")
for cluster in tqdm(sorted_clusters, desc="处理模板"):
    template = cluster.get_template()  # 获取模板字符串
    cluster_id = cluster.cluster_id    # 模板ID
    count = cluster.size               # 模板出现次数
    
    templates.append(template)
    template_ids.append(cluster_id)
    template_counts.append(count)
    
    # 为每个模板收集实例示例
    examples = []
    for message, template_id in message_to_template.items():
        if template_id == cluster_id:
            examples.append(message)
            if len(examples) >= 1000:  # 每个模板最多收集5个示例
                break
    
    template_examples.append(examples)

# 创建DataFrame保存模板信息
template_df = pd.DataFrame(data={
    'template': templates,       # 模板内容列
    'count': template_counts,    # 模板出现次数列
    'examples': template_examples # 模板实例示例列
})

# 将模板信息保存到CSV文件
print("保存模板信息到CSV文件...")
template_df.to_csv('./drain/error_log-template-with-examples.csv', index=True)

# 将模板信息保存到JSON文件（更易于阅读实例）
print("保存模板信息到JSON文件...")
template_json = []
for i, row in template_df.iterrows():
    template_json.append({
        'id': i,
        'template': row['template'],
        'count': row['count'],
        'examples': row['examples']
    })

with open('./drain/error_log-template-with-examples.json', 'w') as f:
    json.dump(template_json, f, indent=2)

print("处理完成！模板及实例已保存到 ./drain/error_log-template-with-examples.csv 和 ./drain/error_log-template-with-examples.json")

# 输出前1000个模板的示例
print("\n前1000个模板的示例:")
for i in range(min(5, len(template_df))):
    print(f"\n模板 {i}:")
    print(f"内容: {template_df.iloc[i]['template'][:100]}..." if len(template_df.iloc[i]['template']) > 100 else f"内容: {template_df.iloc[i]['template']}")
    print(f"出现次数: {template_df.iloc[i]['count']}")
    print("示例:")
    for j, example in enumerate(template_df.iloc[i]['examples']):
        print(f"  示例 {j+1}: {example[:150]}..." if len(example) > 150 else f"  示例 {j+1}: {example}") 