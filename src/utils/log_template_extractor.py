import os
import sys
import pandas as pd
from drain.drain_template_extractor import *
from tqdm import tqdm  # 用于显示进度条

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from utils.io_util import load, save
from utils import io_util


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取当前文件的上层目录
error_logs_path = os.path.join(base_dir, 'data', 'phaseone', 'processed', 'error_logs.parquet')
print(base_dir)
print(error_logs_path)

df_error_logs = pd.read_parquet(error_logs_path)

error_logs = df_error_logs['message'].values.tolist()  # 收集日志消息

# 使用Drain算法提取日志模板
miner = extract_templates(
    log_list=error_logs,  # 传入所有日志消息列表
    save_pth='drain/error_log-drain.pkl'  # 保存训练好的Drain模型路径
)
# 也可以直接加载已训练的模型:
# miner = io_util.load('drain/error_log-drain.pkl')

# 对聚类结果按大小降序排序
sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)

# 准备模板统计信息
template_ids = []    # 存储模板ID
template_counts = [] # 存储模板出现次数  
templates = []       # 存储模板内容

# 遍历排序后的聚类结果
for cluster in sorted_clusters:
    templates.append(cluster.get_template())  # 获取模板字符串
    template_ids.append(cluster.cluster_id)   # 记录模板ID
    template_counts.append(cluster.size)      # 记录模板出现次数

# 创建DataFrame保存模板信息
template_df = pd.DataFrame(data={
    # 'id': template_ids,      # 模板ID列
    'template': templates,   # 模板内容列
    'count': template_counts # 模板出现次数列
})

# 将模板信息保存到CSV文件
template_df.to_csv('./drain/error_log-template.csv', index=True)
