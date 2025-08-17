import pandas as pd
import os
import sys
from typing import Optional, List, Tuple, Dict
import json

# 添加项目根目录到系统路径，确保可以导入utils.io_util
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import utils.io_util as io

# 导入大模型相关模块
from agent.agents import create_llm


def get_fault_period_info(df_fault_timestamps: pd.DataFrame, row_index: int) -> Tuple[List[str], str, str, str]:
    """
    获取指定行的故障时间段信息

    参数:
        df_fault_timestamps: 包含故障起止时间戳的DataFrame
        row_index: 指定要查询的行索引

    返回:
        匹配的Pod文件列表, 日期, 开始时间, 结束时间
    """
    row = df_fault_timestamps.iloc[row_index]
    date = row['date']
    start_time = row['start_timestamp']
    end_time = row['end_timestamp']

    # 构建Pod数据目录路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pod_dir = os.path.join(project_root, 'data', 'processed', f'{date}', 'metric-parquet', 'apm', 'pod')
    matching_files = os.listdir(pod_dir)

    return matching_files, date, start_time, end_time


def extract_service_name_from_pod(pod_name: str) -> str:
    """
    从pod名称中提取service名称

    参数:
        pod_name: pod名称，如 "redis-cart-0"

    返回:
        service名称，如 "redis"
    """
    # 提取用-分割后的第一项作为服务名
    if '-' in pod_name:
        return pod_name.split('-')[0]
    return pod_name


def get_normal_time_periods(df_fault_timestamps: pd.DataFrame, current_index: int) -> List[Tuple[str, str]]:
    """
    获取正常时间段（当前故障前后的正常时间段）

    参数:
        df_fault_timestamps: 故障时间戳DataFrame
        current_index: 当前故障索引

    返回:
        正常时间段列表 [(start_time, end_time), ...]
    """
    normal_periods = []
    current_row = df_fault_timestamps.iloc[current_index]
    current_start = current_row['start_timestamp']
    current_end = current_row['end_timestamp']

    # 获取当前故障前的正常时间段（上一个故障结束到当前故障开始）
    if current_index > 0:
        prev_row = df_fault_timestamps.iloc[current_index - 1]
        prev_end = prev_row['end_timestamp']
        # 正常时间段：上一个故障结束后10分钟 到 当前故障开始
        normal_periods.append((prev_end + 10 * 60 * 1_000_000_000, current_start))
        # normal_periods.append((prev_end , current_start))

    # 获取当前故障后的正常时间段（当前故障结束到下一个故障开始）
    if current_index < len(df_fault_timestamps) - 1:
        next_row = df_fault_timestamps.iloc[current_index + 1]
        next_start = next_row['start_timestamp']
        # 正常时间段：当前故障结束 到 下一个故障开始
        normal_periods.append((current_end + 10 * 60 * 1_000_000_000, next_start))
        # normal_periods.append((current_end , next_start))

    return normal_periods


def get_metrics_description_from_dataframe(df_pod: pd.DataFrame, columns: List[str] = None) -> Dict[str, pd.Series]:
    """
    获取DataFrame指定列的统计描述信息

    参数:
        df_pod: Pod指标数据的DataFrame
        columns: 需要获取描述统计的列名列表，如果为None则使用数值型列

    返回:
        包含每列描述统计信息的字典
    """
    if columns is None:
        # 默认选择数值型列
        numeric_columns = ['client_error_ratio', 'error_ratio', 'request', 'response', 'rrt', 'server_error_ratio',
                           'timeout']
        # 过滤出实际存在的列
        columns = [col for col in numeric_columns if col in df_pod.columns]

    descriptions = {}
    for column in columns:
        if column in df_pod.columns:
            # 描述统计（含 0.25、0.5、0.75、0.95、0.99）
            desc = df_pod[column].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])

            # 计算非零比例
            col_data = df_pod[column].dropna()
            non_zero_ratio = (col_data != 0).sum() / len(col_data) if len(col_data) > 0 else 0
            desc['non_zero_ratio'] = round(non_zero_ratio, 3)  # 保留三位小数

            descriptions[column] = desc
        else:
            print(f"警告: 列 '{column}' 不存在于DataFrame中")

    return descriptions


def get_filtered_metrics_description_with_outlier_removal(df_pod: pd.DataFrame, start_time: str, end_time: str,
                                                          target_columns: List[str] = None,
                                                          remove_outliers: bool = False) -> Dict[str, pd.Series]:
    """
    获取指定时间范围内的指标描述统计，可选择是否移除异常值

    参数:
        df_pod: Pod指标数据的DataFrame
        start_time: 开始时间戳
        end_time: 结束时间戳
        target_columns: 需要分析的列名列表
        remove_outliers: 是否移除异常值（最小2个和最大2个值）

    返回:
        指标描述统计信息字典
    """
    if 'timestamp_ns' in df_pod.columns:
        # 将时间戳转换为整数进行比较
        start_ts = int(start_time)
        end_ts = int(end_time)
        df_filtered = df_pod[(df_pod['timestamp_ns'] >= start_ts) & (df_pod['timestamp_ns'] <= end_ts)]
    else:
        print("警告: 未找到timestamp_ns列，无法进行时间过滤")
        df_filtered = df_pod

    if len(df_filtered) == 0:
        print("指定时间范围内无数据")
        return {}

    # 如果需要移除异常值且数据量足够
    if remove_outliers and len(df_filtered) > 4:  # 至少需要5个数据点才能移除4个
        return get_metrics_description_from_dataframe_without_outliers(df_filtered, target_columns)
    else:
        return get_metrics_description_from_dataframe(df_filtered, target_columns)


def get_metrics_description_from_dataframe_without_outliers(df_pod: pd.DataFrame, columns: List[str] = None) -> Dict[
    str, pd.Series]:
    """
    获取DataFrame指定列的统计描述信息，移除最小2个和最大2个值

    参数:
        df_pod: Pod指标数据的DataFrame
        columns: 需要获取描述统计的列名列表，如果为None则使用数值型列

    返回:
        包含每列描述统计信息的字典
    """
    if columns is None:
        # 默认选择数值型列
        numeric_columns = ['client_error_ratio', 'error_ratio', 'request', 'response', 'rrt', 'server_error_ratio',
                           'timeout']
        # 过滤出实际存在的列
        columns = [col for col in numeric_columns if col in df_pod.columns]

    descriptions = {}
    for column in columns:
        if column in df_pod.columns:
            # 获取该列的数据并排序
            col_data = df_pod[column].dropna().sort_values()

            if len(col_data) <= 4:
                # 数据点太少，直接用原始数据描述
                desc = col_data.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
                print(f"警告: 列 '{column}' 数据点不足({len(col_data)}个)，未移除异常值")
            else:
                # 去掉最小2个和最大2个
                trimmed_data = col_data.iloc[2:-2]
                desc = trimmed_data.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
                print(f"列 '{column}': 原始数据{len(col_data)}个，移除最大和最小两个值后{len(trimmed_data)}个")

            # 计算非零比例（基于去除异常值后的数据）
            non_zero_ratio = (trimmed_data != 0).sum() / len(trimmed_data) if len(col_data) > 4 else (col_data != 0).sum() / len(col_data)
            desc['non_zero_ratio'] = round(non_zero_ratio, 3)

            descriptions[column] = desc
        else:
            print(f"警告: 列 '{column}' 不存在于DataFrame中")

    return descriptions


def analyze_fault_vs_normal_metrics_by_service(df_fault_timestamps: pd.DataFrame, index: int,
                                               target_columns: List[str] = None) -> Optional[Dict]:
    """
    按Service级别分析故障时间段与正常时间段的指标对比
    结构：service → pod → metrics (normal_periods_combined, fault_period)

    参数:
        df_fault_timestamps: 故障时间戳DataFrame
        index: 要分析的故障索引
        target_columns: 需要分析的指标列名列表

    返回:
        按Service组织的包含故障和正常时间段指标对比的字典
    """
    pod_files, date, fault_start, fault_end = get_fault_period_info(df_fault_timestamps, index)

    if not pod_files:
        print("未找到匹配的Pod文件")
        return None

    normal_periods = get_normal_time_periods(df_fault_timestamps, index)

    print(f"故障日期: {date}")
    print(f"故障时间段: {fault_start} 至 {fault_end}")
    print(f"正常时间段数量: {len(normal_periods)}")
    print(f"匹配的Pod文件数量: {len(pod_files)}")

    # 按Service → Pod → Metrics 结构组织分析结果
    service_analysis = {}

    for pod_file in pod_files:
        pod_path = os.path.join(project_root, 'data', 'processed', f'{date}', 'metric-parquet', 'apm', 'pod', pod_file)
        pod_name = pod_file.split('_')[1] if '_' in pod_file else pod_file.split('.')[0]
        service_name = extract_service_name_from_pod(pod_name)

        try:
            df_pod = pd.read_parquet(pod_path)

            if len(df_pod) == 0:
                print(f"Pod {pod_name} 无数据")
                continue

            # 如果service不存在，初始化
            if service_name not in service_analysis:
                service_analysis[service_name] = {}

            # 如果pod不存在，初始化
            if pod_name not in service_analysis[service_name]:
                service_analysis[service_name][pod_name] = {
                    'normal_periods_combined': {},  # 合并的正常数据统计
                    'fault_period': {}  # 故障数据统计
                }

            print(f"\n=== Service: {service_name} | Pod: {pod_name} ===")

            # 1. 先合并所有正常时间段的数据进行统计
            print(f"\n📈 正常时间段合并分析（已移除异常值）:")

            # 收集所有正常时间段的数据
            all_normal_data = []
            total_normal_count = 0

            for i, (normal_start, normal_end) in enumerate(normal_periods):
                print(f"  包含正常时间段 {i + 1}: {normal_start} 至 {normal_end}")

                # 过滤当前正常时间段的数据
                start_ts = int(normal_start)
                end_ts = int(normal_end)
                normal_data = df_pod[(df_pod['timestamp_ns'] >= start_ts) & (df_pod['timestamp_ns'] <= end_ts)]

                if len(normal_data) > 0:
                    all_normal_data.append(normal_data)
                    total_normal_count += len(normal_data)
                    print(f"    时间段 {i + 1} 数据行数: {len(normal_data)}")

            # 合并所有正常时间段的数据
            if all_normal_data:
                combined_normal_data = pd.concat(all_normal_data, ignore_index=True)
                print(f"  合并后正常时间段总数据行数: {len(combined_normal_data)}")

                # 对合并的正常数据进行统计（移除异常值）
                if len(combined_normal_data) > 4:  # 至少需要5个数据点才能移除4个
                    normal_desc = get_metrics_description_from_dataframe_without_outliers(combined_normal_data,
                                                                                          target_columns)
                else:
                    normal_desc = get_metrics_description_from_dataframe(combined_normal_data, target_columns)

                service_analysis[service_name][pod_name]['normal_periods_combined'] = normal_desc

                if normal_desc:
                    print("  合并正常期间指标统计:")
                    for col_name, desc in normal_desc.items():
                        print(f"    {col_name}: mean={desc['mean']:.2f}, std={desc['std']:.2f}")
            else:
                print("  未找到正常时间段数据")

            # 2. 再获取故障时间段的统计（不移除异常值）
            print(f"\n📊 故障时间段分析:")
            fault_desc = get_filtered_metrics_description_with_outlier_removal(
                df_pod, fault_start, fault_end, target_columns, remove_outliers=False
            )

            service_analysis[service_name][pod_name]['fault_period'] = fault_desc

            fault_data_count = len(df_pod[(df_pod['timestamp_ns'] >= int(fault_start)) &
                                          (df_pod['timestamp_ns'] <= int(fault_end))])
            print(f"  故障时间段数据行数: {fault_data_count}")

            if fault_desc:
                print("  故障期间指标统计:")
                for col_name, desc in fault_desc.items():
                    print(f"    {col_name}: mean={desc['mean']:.2f}, std={desc['std']:.2f}")

        except Exception as e:
            print(f"处理Pod文件 {pod_file} 时出错: {e}")

    return service_analysis if service_analysis else None


def get_node_metrics_files_mapping(date: str) -> Dict[str, str]:
    """
    获取节点指标文件名映射，返回指标名称到文件名的映射关系

    参数:
        date: 日期，格式如 "2025-06-06"

    返回:
        指标名到文件名的映射字典
    """
    return {
        'node_cpu_usage_rate': f'infra_node_node_cpu_usage_rate_{date}.parquet',
        'node_disk_read_bytes_total': f'infra_node_node_disk_read_bytes_total_{date}.parquet',
        'node_disk_read_time_seconds_total': f'infra_node_node_disk_read_time_seconds_total_{date}.parquet',
        'node_disk_write_time_seconds_total': f'infra_node_node_disk_write_time_seconds_total_{date}.parquet',
        'node_disk_written_bytes_total': f'infra_node_node_disk_written_bytes_total_{date}.parquet',
        'node_filesystem_free_bytes': f'infra_node_node_filesystem_free_bytes_{date}.parquet',
        'node_filesystem_size_bytes': f'infra_node_node_filesystem_size_bytes_{date}.parquet',
        'node_filesystem_usage_rate': f'infra_node_node_filesystem_usage_rate_{date}.parquet',
        'node_memory_MemAvailable_bytes': f'infra_node_node_memory_MemAvailable_bytes_{date}.parquet',
        'node_memory_MemTotal_bytes': f'infra_node_node_memory_MemTotal_bytes_{date}.parquet',
        'node_memory_usage_rate': f'infra_node_node_memory_usage_rate_{date}.parquet',
        'node_network_receive_bytes_total': f'infra_node_node_network_receive_bytes_total_{date}.parquet',
        'node_network_receive_packets_total': f'infra_node_node_network_receive_packets_total_{date}.parquet',
        'node_network_transmit_bytes_total': f'infra_node_node_network_transmit_bytes_total_{date}.parquet',
        'node_network_transmit_packets_total': f'infra_node_node_network_transmit_packets_total_{date}.parquet',
        'node_sockstat_TCP_inuse': f'infra_node_node_sockstat_TCP_inuse_{date}.parquet'
    }


def get_target_nodes() -> List[str]:
    """
    获取目标分析节点列表（只分析aiops-k8s-01到aiops-k8s-08这8个节点）

    返回:
        目标节点名称列表
    """
    return [f'aiops-k8s-{i:02d}' for i in range(1, 9)]  # aiops-k8s-01 到 aiops-k8s-08


def load_node_metric_data(date: str, metric_name: str) -> Optional[pd.DataFrame]:
    """
    加载指定日期和指标的节点数据

    参数:
        date: 日期，格式如 "2025-06-06"
        metric_name: 指标名称，如 "node_cpu_usage_rate"

    返回:
        节点指标数据DataFrame，如果文件不存在则返回None
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    node_dir = os.path.join(project_root, 'data', 'processed', f'{date}', 'metric-parquet', 'infra', 'infra_node')

    file_mapping = get_node_metrics_files_mapping(date)

    if metric_name not in file_mapping:
        print(f"故障的指标名称: {metric_name}")
        return None

    file_path = os.path.join(node_dir, file_mapping[metric_name])

    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        df = pd.read_parquet(file_path)

        # 只保留目标节点数据
        target_nodes = get_target_nodes()
        df_filtered = df[df['kubernetes_node'].isin(target_nodes)]

        if len(df_filtered) == 0:
            print(f"文件 {file_path} 中未找到目标节点数据")
            return None

        return df_filtered

    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None


def get_node_metrics_description_with_time_filter(df_node: pd.DataFrame, start_time: str, end_time: str,
                                                  metric_column: str, remove_outliers: bool = False) -> Optional[
    pd.Series]:
    """
    获取指定时间范围内节点指标的描述统计

    参数:
        df_node: 节点指标数据DataFrame
        start_time: 开始时间戳
        end_time: 结束时间戳
        metric_column: 指标列名（实际数值列）
        remove_outliers: 是否移除异常值

    返回:
        指标描述统计信息，如果无数据则返回None
    """
    if 'timestamp_ns' not in df_node.columns:
        print("警告: 未找到timestamp_ns列，无法进行时间过滤")
        return None

    # 时间过滤
    start_ts = int(start_time)
    end_ts = int(end_time)
    df_filtered = df_node[(df_node['timestamp_ns'] >= start_ts) & (df_node['timestamp_ns'] <= end_ts)]

    if len(df_filtered) == 0:
        print("指定时间范围内无数据")
        return None

    # 获取指标数据
    if metric_column not in df_filtered.columns:
        print(f"警告: 列 '{metric_column}' 不存在于DataFrame中")
        return None

    metric_data = df_filtered[metric_column].dropna()

    if len(metric_data) == 0:
        print(f"指标 '{metric_column}' 无有效数据")
        return None

    # 是否移除异常值
    if remove_outliers and len(metric_data) > 4:
        metric_data_sorted = metric_data.sort_values()
        metric_data = metric_data_sorted.iloc[2:-2]  # 去掉最小2个和最大2个
        print(f"移除异常值后数据点数量: {len(metric_data)}")
     # 描述统计 + 百分位
    desc = metric_data.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])

    # **新增：非零比例**
    non_zero_ratio = (metric_data != 0).sum() / len(metric_data)
    desc['non_zero_ratio'] = round(non_zero_ratio, 3)

    return desc


def analyze_node_metrics_by_node(df_fault_timestamps: pd.DataFrame, index: int,
                                 target_metrics: List[str] = None) -> Optional[Dict]:
    """
    分析指定故障时间段与正常时间段的节点指标对比
    结构：node → metric → {normal_periods_combined, fault_period}

    参数:
        df_fault_timestamps: 故障时间戳DataFrame
        index: 要分析的故障索引
        target_metrics: 需要分析的指标列表，如果为None则使用全部10个指标

    返回:
        按节点组织的包含故障和正常时间段指标对比的字典
    """
    if target_metrics is None:
        target_metrics = ['node_cpu_usage_rate',
                          'node_disk_read_bytes_total',
                          'node_disk_read_time_seconds_total',
                          'node_disk_write_time_seconds_total',
                          'node_disk_written_bytes_total',
                          'node_filesystem_free_bytes',
                          'node_filesystem_usage_rate',
                          'node_filesystem_usage_rate',
                          'node_memory_MemAvailable_bytes',
                          'node_memory_MemTotal_bytes',
                          'node_memory_usage_rate',
                          'node_network_receive_bytes_total',
                          'node_network_receive_packets_total',
                          'node_network_transmit_bytes_total',
                          'node_network_transmit_packets_total',
                          'node_sockstat_TCP_inuse', ]

    # 获取故障时间信息
    _, date, fault_start, fault_end = get_fault_period_info(df_fault_timestamps, index)
    normal_periods = get_normal_time_periods(df_fault_timestamps, index)
    target_nodes = get_target_nodes()

    print(f"节点分析 - 故障日期: {date}")
    print(f"节点分析 - 故障时间段: {fault_start} 至 {fault_end}")
    print(f"节点分析 - 正常时间段数量: {len(normal_periods)}")
    print(f"节点分析 - 目标节点数量: {len(target_nodes)}")
    print(f"节点分析 - 分析指标数量: {len(target_metrics)}")

    # 按 节点 → 指标 → 时间段 结构组织分析结果
    nodes_analysis = {}

    for node_name in target_nodes:
        print(f"\n=== 处理节点: {node_name} ===")

        # 初始化节点结构
        nodes_analysis[node_name] = {}

        # 为当前节点分析所有指标
        for metric_name in target_metrics:
            print(f"  处理指标: {metric_name}")

            # 加载该指标的数据
            df_metric = load_node_metric_data(date, metric_name)

            if df_metric is None:
                print(f"    无法加载指标 {metric_name} 的数据，跳过")
                continue

            # 过滤当前节点的数据
            df_node = df_metric[df_metric['kubernetes_node'] == node_name]

            if len(df_node) == 0:
                print(f"    节点 {node_name} 无 {metric_name} 数据")
                continue

            # 初始化指标结构
            nodes_analysis[node_name][metric_name] = {
                'normal_periods_combined': None,
                'fault_period': None
            }

            # 1. 合并所有正常时间段数据进行统计
            print(f"    正常时间段分析:")
            all_normal_data = []

            for i, (normal_start, normal_end) in enumerate(normal_periods):
                start_ts = int(normal_start)
                # start_ts = int(normal_start)
                end_ts = int(normal_end)
                normal_data = df_node[(df_node['timestamp_ns'] >= start_ts) & (df_node['timestamp_ns'] <= end_ts)]

                if len(normal_data) > 0:
                    all_normal_data.append(normal_data)
                    print(f"      时间段 {i + 1} 数据行数: {len(normal_data)}")

            # 合并正常时间段数据并统计
            if all_normal_data:
                combined_normal_data = pd.concat(all_normal_data, ignore_index=True)
                print(f"    合并后正常时间段总数据行数: {len(combined_normal_data)}")

                # 获取统计（移除异常值）
                normal_desc = get_node_metrics_description_with_time_filter(
                    combined_normal_data,
                    str(combined_normal_data['timestamp_ns'].min()),
                    str(combined_normal_data['timestamp_ns'].max()),
                    metric_name,
                    remove_outliers=(len(combined_normal_data) > 4)
                )

                nodes_analysis[node_name][metric_name]['normal_periods_combined'] = normal_desc

                if normal_desc is not None:
                    print(f"    正常期间 {metric_name}: mean={normal_desc['mean']:.2f}, std={normal_desc['std']:.2f}")

            # 2. 故障时间段统计
            print(f"    故障时间段分析:")
            fault_desc = get_node_metrics_description_with_time_filter(
                df_node, fault_start, fault_end, metric_name, remove_outliers=False
            )

            nodes_analysis[node_name][metric_name]['fault_period'] = fault_desc

            if fault_desc is not None:
                fault_data_count = len(df_node[(df_node['timestamp_ns'] >= int(fault_start)) &
                                               (df_node['timestamp_ns'] <= int(fault_end))])
                print(f"    故障时间段数据行数: {fault_data_count}")
                print(f"    故障期间 {metric_name}: mean={fault_desc['mean']:.2f}, std={fault_desc['std']:.2f}")

    return nodes_analysis if nodes_analysis else None


# ==================== 1. Pod 指标文件映射 ====================

def get_pod_metrics_files_mapping(date: str) -> Dict[str, str]:
    """
    获取 Pod 指标文件名映射，返回指标名称到文件名的映射关系

    参数:
        date: 日期，格式如 "2025-06-06"

    返回:
        指标名到文件名的映射字典
    """
    return {
        'pod_cpu_usage': f'infra_pod_pod_cpu_usage_{date}.parquet',
        'pod_fs_reads_bytes': f'infra_pod_pod_fs_reads_bytes_{date}.parquet',
        'pod_fs_writes_bytes': f'infra_pod_pod_fs_writes_bytes_{date}.parquet',
        'pod_memory_working_set_bytes': f'infra_pod_pod_memory_working_set_bytes_{date}.parquet',
        'pod_network_receive_bytes': f'infra_pod_pod_network_receive_bytes_{date}.parquet',
        'pod_network_receive_packets': f'infra_pod_pod_network_receive_packets_{date}.parquet',
        'pod_network_transmit_bytes': f'infra_pod_pod_network_transmit_bytes_{date}.parquet',
        'pod_network_transmit_packets': f'infra_pod_pod_network_transmit_packets_{date}.parquet',
        'pod_processes': f'infra_pod_pod_processes_{date}.parquet'
    }


# ==================== 2. 目标 Pod 列表 ====================

def get_target_pods() -> List[str]:
    """
    获取目标分析 Pod 列表
    """
    services = [
        "adservice-0", "adservice-1", "adservice-2",
        "cartservice-0", "cartservice-1", "cartservice-2",
        "checkoutservice-0", "checkoutservice-1", "checkoutservice-2",
        "currencyservice-0", "currencyservice-1", "currencyservice-2",
        "emailservice-0", "emailservice-1", "emailservice-2",
        "frontend-0", "frontend-1", "frontend-2",
        "paymentservice-0", "paymentservice-1", "paymentservice-2",
        "productcatalogservice-0", "productcatalogservice-1", "productcatalogservice-2",
        "recommendationservice-0", "recommendationservice-1", "recommendationservice-2",
        "redis-cart-0",
        "shippingservice-0", "shippingservice-1", "shippingservice-2"
    ]
    return services


# ==================== 3. 加载 Pod 指标数据 ====================

def load_pod_metric_data(date: str, metric_name: str) -> Optional[pd.DataFrame]:
    """
    加载指定日期和指标的 Pod 数据

    参数:
        date: 日期，格式如 "2025-06-06"
        metric_name: 指标名称，如 "pod_cpu_usage"

    返回:
        Pod 指标数据 DataFrame，如果文件不存在则返回 None
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pod_dir = os.path.join(project_root, 'data', 'processed', f'{date}', 'metric-parquet', 'infra', 'infra_pod')

    file_mapping = get_pod_metrics_files_mapping(date)

    if metric_name not in file_mapping:
        print(f"故障的指标名称: {metric_name}")
        return None

    file_path = os.path.join(pod_dir, file_mapping[metric_name])

    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        df = pd.read_parquet(file_path)

        # 只保留目标 pod 数据
        target_pods = get_target_pods()
        df_filtered = df[df['pod'].isin(target_pods)]

        if len(df_filtered) == 0:
            print(f"文件 {file_path} 中未找到目标 pod 数据")
            return None

        return df_filtered

    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None


# ==================== 4. 时间过滤统计 ====================

def get_pod_metrics_description_with_time_filter(df_pod: pd.DataFrame, start_time: str, end_time: str,
                                                 metric_column: str, remove_outliers: bool = False) -> Optional[
    pd.Series]:
    """
    获取指定时间范围内 Pod 指标的描述统计
    """
    if 'timestamp_ns' not in df_pod.columns:
        print("警告: 未找到 timestamp_ns 列，无法进行时间过滤")
        return None

    # 时间过滤
    start_ts = int(start_time)
    end_ts = int(end_time)
    df_filtered = df_pod[(df_pod['timestamp_ns'] >= start_ts) & (df_pod['timestamp_ns'] <= end_ts)]

    if len(df_filtered) == 0:
        print("指定时间范围内无数据")
        return None

    # 获取指标数据
    if metric_column not in df_filtered.columns:
        print(f"警告: 列 '{metric_column}' 不存在于 DataFrame 中")
        return None

    metric_data = df_filtered[metric_column].dropna()

    if len(metric_data) == 0:
        print(f"指标 '{metric_column}' 无有效数据")
        return None

    # 是否移除异常值
    if remove_outliers and len(metric_data) > 4:
        metric_data_sorted = metric_data.sort_values()
        metric_data = metric_data_sorted.iloc[2:-2]  # 去掉最小2个和最大2个
        print(f"移除异常值后数据点数量: {len(metric_data)}")
    # 生成描述统计
    desc = metric_data.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])

    # **新增非零比例**
    desc['non_zero_ratio'] = round((metric_data != 0).sum() / len(metric_data), 3)

    return desc


# ==================== 5. 按 Pod 分析故障 vs 正常 ====================

def analyze_pod_metrics_by_pod(df_fault_timestamps: pd.DataFrame, index: int,
                               target_metrics: List[str] = None) -> Optional[Dict]:
    """
    分析指定故障时间段与正常时间段的 Pod 指标对比
    结构：pod → metric → {normal_periods_combined, fault_period}
    """
    if target_metrics is None:
        target_metrics = [
            'pod_cpu_usage', 'pod_fs_reads_bytes', 'pod_fs_writes_bytes',
            'pod_memory_working_set_bytes', 'pod_network_receive_bytes',
            'pod_network_receive_packets', 'pod_network_transmit_bytes',
            'pod_network_transmit_packets', 'pod_processes'
        ]

    # 获取故障时间信息
    _, date, fault_start, fault_end = get_fault_period_info(df_fault_timestamps, index)
    normal_periods = get_normal_time_periods(df_fault_timestamps, index)
    target_pods = get_target_pods()

    print(f"Pod 分析 - 故障日期: {date}")
    print(f"Pod 分析 - 故障时间段: {fault_start} 至 {fault_end}")
    print(f"Pod 分析 - 正常时间段数量: {len(normal_periods)}")
    print(f"Pod 分析 - 目标 Pod 数量: {len(target_pods)}")
    print(f"Pod 分析 - 分析指标数量: {len(target_metrics)}")

    # 按 Pod → 指标 → 时间段 结构组织分析结果
    pods_analysis = {}

    for pod_name in target_pods:
        print(f"\n=== 处理 Pod: {pod_name} ===")

        pods_analysis[pod_name] = {}

        for metric_name in target_metrics:
            print(f"  处理指标: {metric_name}")

            # 加载该指标的数据
            df_metric = load_pod_metric_data(date, metric_name)

            if df_metric is None:
                print(f"    无法加载指标 {metric_name} 的数据，跳过")
                continue

            # 过滤当前 Pod 的数据
            df_pod = df_metric[df_metric['pod'] == pod_name]
            # 删除 device 列为 /dev/vdb 的行
            if 'device' in df_pod.columns:
                df_pod = df_pod[df_pod['device'] != '/dev/dmb']

            if len(df_pod) == 0:
                print(f"    Pod {pod_name} 无 {metric_name} 数据")
                continue

            # 初始化指标结构
            pods_analysis[pod_name][metric_name] = {
                'normal_periods_combined': None,
                'fault_period': None
            }

            # 1. 合并所有正常时间段数据
            print(f"    正常时间段分析:")
            all_normal_data = []

            for i, (normal_start, normal_end) in enumerate(normal_periods):
                start_ts = int(normal_start)
                end_ts = int(normal_end)
                normal_data = df_pod[(df_pod['timestamp_ns'] >= start_ts) & (df_pod['timestamp_ns'] <= end_ts)]

                if len(normal_data) > 0:
                    all_normal_data.append(normal_data)
                    print(f"      时间段 {i + 1} 数据行数: {len(normal_data)}")

            # 合并正常时间段数据并统计
            if all_normal_data:
                combined_normal_data = pd.concat(all_normal_data, ignore_index=True)
                print(f"    合并后正常时间段总数据行数: {len(combined_normal_data)}")

                normal_desc = get_pod_metrics_description_with_time_filter(
                    combined_normal_data,
                    str(combined_normal_data['timestamp_ns'].min()),
                    str(combined_normal_data['timestamp_ns'].max()),
                    metric_name,
                    remove_outliers=(len(combined_normal_data) > 4)
                )
                if normal_desc is not None:
                    print(f"    正常期间 {metric_name}: mean={normal_desc['mean']:.2f}, std={normal_desc['std']:.2f}")

            # 2. 故障时间段统计
            print(f"    故障时间段分析:")
            fault_desc = get_pod_metrics_description_with_time_filter(
                df_pod, fault_start, fault_end, metric_name, remove_outliers=False
            )
            # if normal_desc is not None and fault_desc is not None:#过滤掉变化倍数在 0.95 到 1.05 之间的指标
            #     normal_mean = normal_desc['mean']
            #     fault_mean = fault_desc['mean']
            #     epsilon = 1e-9  # 极小数，防止除零
            #     ratio = (fault_mean + epsilon) / (normal_mean + epsilon)
            #
            #     if 0.95 <= ratio <= 1.05:
            #         print(f"    指标 {metric_name} 变化倍数 {ratio:.2f} 在 0.95~1.05 之间，跳过保存")
            #         continue
            pods_analysis[pod_name][metric_name]['fault_period'] = fault_desc
            pods_analysis[pod_name][metric_name]['normal_periods_combined'] = normal_desc
            if fault_desc is not None:
                fault_data_count = len(df_pod[(df_pod['timestamp_ns'] >= int(fault_start)) &
                                              (df_pod['timestamp_ns'] <= int(fault_end))])
                print(f"    故障时间段数据行数: {fault_data_count}")
                print(f"    故障期间 {metric_name}: mean={fault_desc['mean']:.2f}, std={fault_desc['std']:.2f}")

    return pods_analysis if pods_analysis else None


def call_llm_analysis(prompt: str, uuid: str = None, call_type: str = "未知类型") -> str:
    """
    调用大模型进行分析

    参数:
        prompt: 分析prompt
        uuid: 样本UUID，用于记录
        call_type: 调用类型，用于记录

    返回:
        大模型的分析结果
    """
    try:
        llm_agent = create_llm()
        messages = [
            {
                "content": prompt,
                "role": "user",
            }
        ]
        reply = llm_agent.generate_reply(messages)

        response_content = reply['content'] if isinstance(reply, dict) and 'content' in reply else reply
        # 记录大模型调用
        if uuid:
            try:
                from utils.llm_record_utils import record_llm_call
                record_llm_call(uuid, call_type, prompt, response_content)
            except Exception as record_e:
                print(f"记录大模型调用失败: {record_e}")

        return response_content
    except Exception as e:
        print(f"调用大模型时出错: {e}")
        error_msg = f"大模型调用失败: {str(e)}"

        # 即使出错也记录
        if uuid:
            try:
                from utils.llm_record_utils import record_llm_call
                record_llm_call(uuid, f"{call_type}(失败)", prompt, error_msg)
            except Exception as record_e:
                print(f"记录大模型调用失败: {record_e}")

        return error_msg


def get_node_pod_mapping(date: str) -> Dict[str, List[str]]:
    """
    获取每个节点上部署的Pod列表

    参数:
        date: 日期，格式如 "2025-06-06"

    返回:
        节点到Pod列表的映射字典 {node_name: [pod1, pod2, ...]}
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    infra_pod_dir = os.path.join(project_root, 'data', 'processed', f'{date}', 'metric-parquet', 'infra', 'infra_pod')

    # 优先尝试读取CPU使用率文件
    target_file = f'infra_pod_pod_cpu_usage_{date}.parquet'
    target_file_path = os.path.join(infra_pod_dir, target_file)

    df_pod_info = None

    try:
        if os.path.exists(target_file_path):
            print(f"使用目标文件获取Pod部署信息: {target_file}")
            df_pod_info = pd.read_parquet(target_file_path)
        else:
            # 如果目标文件不存在，随机选择一个文件
            if os.path.exists(infra_pod_dir):
                available_files = [f for f in os.listdir(infra_pod_dir) if f.endswith('.parquet')]
                if available_files:
                    selected_file = available_files[0]  # 选择第一个文件
                    selected_file_path = os.path.join(infra_pod_dir, selected_file)
                    print(f"目标文件不存在，使用备选文件获取Pod部署信息: {selected_file}")
                    df_pod_info = pd.read_parquet(selected_file_path)
                else:
                    print("infra_pod目录中没有可用的parquet文件")
                    return {}
            else:
                print(f"infra_pod目录不存在: {infra_pod_dir}")
                return {}

        if df_pod_info is None or len(df_pod_info) == 0:
            print("无法读取Pod部署信息")
            return {}

        # 获取目标节点列表
        target_nodes = get_target_nodes()
        node_pod_mapping = {}

        for node_name in target_nodes:
            # 筛选该节点的数据
            node_data = df_pod_info[df_pod_info['instance'] == node_name]
            if len(node_data) > 0:
                # 获取该节点上的唯一Pod列表
                pods_on_node = node_data['pod'].unique().tolist()
                node_pod_mapping[node_name] = pods_on_node
                print(f"节点 {node_name} 部署的Pod数量: {len(pods_on_node)}")
            else:
                print(f"节点 {node_name} 未找到Pod部署信息")
                node_pod_mapping[node_name] = []

        return node_pod_mapping

    except Exception as e:
        print(f"获取Pod部署信息时出错: {e}")
        return {}


def create_combined_node_prompt_with_service_analysis(node_analysis_results: Dict, pod_result: Dict,
                                                      service_analysis_result: str,
                                                      node_pod_mapping: Dict[str, List[str]]) -> str:
    """
    合并所有节点的prompt为一个综合prompt，包含service分析结果和pod部署信息

    参数:
        node_analysis_results: 节点分析结果
        service_analysis_result: service级别的LLM分析结果
        node_pod_mapping: 节点到Pod列表的映射

    返回:
        合并后的综合prompt
    """
    if not node_analysis_results:
        return ""

    # 获取所有节点信息
    all_nodes = list(node_analysis_results.keys())
    all_metrics = set()
    for node_data in node_analysis_results.values():
        all_metrics.update(node_data.keys())
    all_metrics = list(all_metrics)

    # 指标中文名称映射
    metric_chinese_names = {
        'node_cpu_usage_rate': 'CPU使用率',
        'node_disk_read_bytes_total': '磁盘读取字节数',
        'node_disk_read_time_seconds_total': '磁盘读取时间',
        'node_disk_write_time_seconds_total': '磁盘写入时间',
        'node_disk_written_bytes_total': '磁盘写入字节数',
        'node_filesystem_free_bytes': '空闲磁盘大小',
        'node_filesystem_size_bytes': '磁盘总大小',
        'node_filesystem_usage_rate': '文件系统使用率',
        'node_memory_MemAvailable_bytes': '空闲内存大小',
        'node_memory_MemTotal_bytes': '内存总大小',
        'node_memory_usage_rate': '内存使用率',
        'node_network_receive_bytes_total': '网络接收字节数',
        'node_network_receive_packets_total': 'Receive 各个接口每秒接收的数据包总数',
        'node_network_transmit_bytes_total': 'Transmit 各个网络接口发送速率',
        'node_network_transmit_packets_total': 'Transmit 各个接口每秒发送的数据包总数',
        'node_sockstat_TCP_inuse': 'TCP连接数',
    }

    # 构建节点Pod部署信息
    node_deployment_info = []
    total_pods = 0

    # for node_name in all_nodes:
    #     pods_on_node = node_pod_mapping.get(node_name, [])
    #     total_pods += len(pods_on_node)
    #     node_deployment_info.append(f"\n### 节点: {node_name}")
    #     node_deployment_info.append(f"- **部署Pod数量**: {len(pods_on_node)}")
    #     if pods_on_node:
    #         node_deployment_info.append(f"- **Pod列表**: {pods_on_node}")
    #     else:
    #         node_deployment_info.append(f"- **Pod列表**: 无数据")

    # deployment_info_text = "\n".join(node_deployment_info)

    # 构建合并的数据对比表格
    combined_json = {}

    # 提取全局一致的 metric_count 和 metric_list
    # 假设所有 node_data 的指标一致，用第一个节点的数据即可
    first_node_data = next(iter(node_analysis_results.values()))
    combined_json["metric_count"] = len(first_node_data)
    combined_json["metric_list"] = list(first_node_data.keys())

    # 用 nodes 存储每个节点的独立数据
    combined_json["nodes"] = {}

    for node_name, node_data in node_analysis_results.items():
        pods_on_node = node_pod_mapping.get(node_name, [])

        # 初始化节点信息
        node_info = {
            "node_name": node_name,
            "pod_count": len(pods_on_node),
            "pods": pods_on_node,  # Pod名称列表
            "metrics": {},  # 节点指标信息
            "pods_detail": {}  # 该节点下Pod的详细指标
        }

        # ---- 处理节点指标 ----
        for metric_name, metric_stats in node_data.items():
            normal_stats = metric_stats.get('normal_periods_combined')
            fault_stats = metric_stats.get('fault_period')

            if normal_stats is not None and fault_stats is not None:
                normal_mean = normal_stats.get('mean', 0)
                normal_std = normal_stats.get('std', 0)
                normal_max = normal_stats.get('max', 0)
                normal_p99 = normal_stats.get('99%', 0)
                normal_p50 = normal_stats.get('50%', 0)
                fault_mean = fault_stats.get('mean', 0)
                fault_std = fault_stats.get('std', 0)
                fault_max = fault_stats.get('max', 0)
                fault_nzr = fault_stats.get('non_zero_ratio',0)
                normal_nzr = normal_stats.get('non_zero_ratio', 0)
                fault_p99 = fault_stats.get('99%', 0)
                fault_p50 = fault_stats.get('50%', 0)
                normal_p75 = normal_stats.get('75%', 0)
                normal_p25 = normal_stats.get('25%', 0)
                normal_iqr = normal_p75 - normal_p25

                fault_p75 = fault_stats.get('75%', 0)
                fault_p25 = fault_stats.get('25%', 0)
                fault_iqr = fault_p75 - fault_p25
                # 计算对称比率
                p50_symmetric_ratio = abs(fault_p50 - normal_p50) / (
                        (fault_p50 + normal_p50) / 2 + 1e-9
                )
                p99_symmetric_ratio = abs(fault_p99 - normal_p99) / (
                        (fault_p99 + normal_p99) / 2 + 1e-9
                )
                # 过滤阈值（建议 0.5 或 1，根据需求调整）
                if p50_symmetric_ratio < 0.05 and p99_symmetric_ratio <0.05:
                    continue

                node_info["metrics"][metric_name] = {
                    "正常期间中位数": round(normal_p50, 2),
                    "正常期间四分位距": round(normal_iqr, 2),
                    "正常期间99分位数": round(normal_p99, 2),
                    "故障期间中位数": round(fault_p50, 2),
                    "故障期间四分位距": round(fault_iqr, 2),
                    "故障期间99分位数": round(fault_p99, 2)
                }
            else:
                node_info["metrics"][metric_name] = "缺失数据"

        # ---- 处理节点下的 Pods 指标 ----
        for pod_name in pods_on_node:
            pod_metrics = pod_result.get(pod_name, {})  # 获取当前 Pod 的指标数据

            pod_detail = {}
            for metric_name, metric_stats in pod_metrics.items():
                normal_stats = metric_stats.get('normal_periods_combined')
                fault_stats = metric_stats.get('fault_period')

                if normal_stats is not None and fault_stats is not None:
                    normal_mean = normal_stats.get('mean', 0)
                    normal_std = normal_stats.get('std', 0)
                    normal_max = normal_stats.get('max', 0)
                    normal_p99 = normal_stats.get('99%', 0)
                    normal_p50 = normal_stats.get('50%', 0)
                    normal_p25 = normal_stats.get('25%', 0)
                    normal_p75 = normal_stats.get('75%', 0)
                    fault_mean = fault_stats.get('mean', 0)
                    fault_std = fault_stats.get('std', 0)
                    fault_max = fault_stats.get('max', 0)
                    fault_p99 = fault_stats.get('99%', 0)
                    fault_p50 = fault_stats.get('50%', 0)
                    fault_p25 = fault_stats.get('25%', 0)
                    fault_p75 = fault_stats.get('75%', 0)
                    normal_iqr = normal_p75 - normal_p25
                    fault_iqr = fault_p75 - fault_p25
                    fault_nzr = fault_stats.get('non_zero_ratio', 0)
                    normal_nzr = normal_stats.get('non_zero_ratio', 0)
                    # 计算对称比率
                    p50_symmetric_ratio = abs(fault_p50 - normal_p50) / (
                            (fault_p50 + normal_p50) / 2 + 1e-9
                    )
                    p99_symmetric_ratio = abs(fault_p99 - normal_p99) / (
                            (fault_p99 + normal_p99) / 2 + 1e-9
                    )
                    # 过滤阈值（建议 0.5 或 1，根据需求调整）
                    if p50_symmetric_ratio < 0.05 and p99_symmetric_ratio < 0.05:
                        continue

                    pod_detail[metric_name] = {
                        "正常期间中位数": round(normal_p50, 2),
                        "正常期间四分位距": round(normal_iqr, 2),
                        "正常期间99分位数": round(normal_p99, 2),
                        "故障期间中位数": round(fault_p50, 2),
                        "故障期间四分位距": round(fault_iqr, 2),
                        "故障期间99分位数": round(fault_p99, 2)
                    }
                else:
                    pod_detail[metric_name] = "缺失数据"

            # 将该 Pod 详细指标写入节点下
            node_info["pods_detail"][pod_name] = pod_detail

        # 将节点信息放入总 JSON
        combined_json["nodes"][node_name] = node_info
        # combined_json= compress_combined_json(combined_json)
    return f"""
请基于提供的Service级别分析结果、Pod部署信息和基础设施监控指标数据，进行全局的现象总结分析。

## Service级别分析结果回顾
{service_analysis_result}

## 集群基础设施信息
- **节点总数**: {len(all_nodes)}
- **监控指标总数**: {len(all_metrics)}
- **集群总Pod数**: {total_pods}
- **节点列表**: {all_nodes}
- **监控指标**: {[metric_chinese_names.get(m, m) for m in all_metrics]}

## 使用规范说明
• 所有监控指标均为 `kpi_key` 指标（如 `node_cpu_usage_rate`），请始终使用这些原始英文名称进行分析与输出；
• 严禁使用中文或缩写形式（如"CPU"、"CPU使用率"、"磁盘读写"等）代替；
• 必须显式包含对应的 `kpi_key`， 每次提及`kpi_key`指标如( `node_cpu_usage_rate`）时，必须显式包含对应的 `kpi_key`，必须显式指出这是一个 `kpi_key` 指标如：
  • 错误示例：节点级指标变化幅度（CPU ↑23%）
  • 正确示例：kpi_key指标`node_cpu_usage_rate` 在节点 aiops-k8s-08 上上升了 23%

## 基础设施指标分类说明

### 计算资源类指标(kpi_key)：
- **node_cpu_usage_rate**: CPU使用率, 反映节点CPU使用率  
- **node_memory_usage_rate**: 内存使用率, 反映节点内存使用率  
- **pod_cpu_usage**: Pod CPU使用率  
- **pod_memory_working_set_bytes**: Pod工作集内存使用量  
- **pod_processes**: Pod内运行进程数量  

### 存储资源类指标(kpi_key)：
- **node_filesystem_usage_rate**: 文件系统使用率, 反映节点存储空间使用率  
- **node_disk_read_bytes_total / node_disk_read_time_seconds_total**: 磁盘读取字节数/时间, 反映磁盘读取性能  
- **node_disk_written_bytes_total / node_disk_write_time_seconds_total**: 磁盘写入字节数/时间, 反映磁盘写入性能  
- **pod_fs_reads_bytes**: Pod 文件系统读取字节数  
- **pod_fs_writes_bytes**: Pod 文件系统写入字节数  

### 网络资源类指标(kpi_key)：
- **node_network_receive_bytes_total**: 网络接收字节数, 反映节点网络接收流量  
- **node_network_transmit_bytes_total**: 网络发送字节数, 反映节点网络发送流量  
- **node_network_receive_packets_total**: 各接口每秒接收的数据包总数  
- **node_network_transmit_packets_total**: 各接口每秒发送的数据包总数  
- **node_sockstat_TCP_inuse**: TCP连接数, 反映节点TCP连接活跃度  
- **pod_network_receive_bytes**: Pod 网络接收字节数  
- **pod_network_receive_packets**: Pod 网络接收数据包数  
- **pod_network_transmit_bytes**: Pod 网络发送字节数  
- **pod_network_transmit_packets**: Pod 网络发送数据包数  

## 基础设施指标数据对比表格,特别注意**缺失数据和空数据,代表数据波动极小,通常情况默认正常**:
{json.dumps(combined_json, ensure_ascii=False, indent=2)}

## 综合现象分析要求
请从以下维度进行全局现象描述，**仅描述观察到的现象，不做异常判断或结论**：

### 1. Node级别现象观察
基于Node的正常时间段与异常时间段的数据对比，描述：
- 同一Node的正常时间段和异常时间段，显著的指标异常变化或显著的 `kpi_key` 指标异常变化
- 不同Node之间比较表现出的异常差异

### 2. Service级别现象观察
- 集中在同一类型的服务中出现的问题，比如emailservice-0, emailservice-1, emailservice-2, 都存在相似的异常数据变化，则描述具体变化现象，因为这可能是emailservice存在潜在问题

### 3. Pod级别现象观察
- 个别Pod的异常表现特征
- 如 cartservice-0 中存在异常的数据变化，而cartservice-1, cartservice-2以及其他pod正常，则可能是单独的pod级别的异常现象
- 大多数异常的Pod是否部署在同一个Node，还是分散在不同的Node，是否存在Node级别的异常现象

## 重要提示
**这是为后期多模态综合决策分析提供的全局现象总结，请控制总结内容在2000字左右，重点突出主要变化现象。**

**总结要求：**
- 必须在输出中包含原始 `kpi_key` 指标名称，如 `node_cpu_usage_rate`
- 分析原因时必须明确指出该原因属于哪个 `metric`（其他指标 "以及/或者" 哪个`kpi_key`）下的观察
- 如果出现明显变化，请重点描述变化显著的服务、指标和现象
- 采用概括性语言，重点关注对业务影响较大的指标变化
- 关注异常现象倾向于node，service还是pod级别的异常
- 提供系统级的综合现象描述，为后续决策提供全面视角
- 采用客观、描述性的语言，避免主观判断
- 缺失或为空的数据表示波动极小，可视为正常，无需描述

请基于Service分析、Pod部署信息和基础设施监控数据，提供全局的综合现象总结，控制在2000字以内。
"""


# ==================== TiDB 服务相关函数 ====================

def get_tidb_services_files_mapping(date: str) -> Dict[str, Dict[str, str]]:
    """
    获取TiDB服务的文件名映射，返回服务名到指标文件的映射关系

    参数:
        date: 日期，格式如 "2025-06-06"

    返回:
        服务名到指标文件映射的字典 {service_name: {metric_name: file_name}}
    """
    return {
        'tidb-tidb': {
            'failed_query_ops': f'infra_tidb_failed_query_ops_{date}.parquet',
            'duration_99th': f'infra_tidb_duration_99th_{date}.parquet',
            'connection_count': f'infra_tidb_connection_count_{date}.parquet',
            'server_is_up': f'infra_tidb_server_is_up_{date}.parquet',
            'cpu_usage': f'infra_tidb_cpu_usage_{date}.parquet',
            'memory_usage': f'infra_tidb_memory_usage_{date}.parquet'
        },
        'tidb-pd': {
            'store_up_count': f'infra_pd_store_up_count_{date}.parquet',
            'store_down_count': f'infra_pd_store_down_count_{date}.parquet',
            'cpu_usage': f'infra_pd_cpu_usage_{date}.parquet',
            'memory_usage': f'infra_pd_memory_usage_{date}.parquet',
            'storage_used_ratio': f'infra_pd_storage_used_ratio_{date}.parquet',
            'store_unhealth_count': f'infra_pd_store_unhealth_count_{date}.parquet'
        },
        'tidb-tikv': {
            'cpu_usage': f'infra_tikv_cpu_usage_{date}.parquet',
            'memory_usage': f'infra_tikv_memory_usage_{date}.parquet',
            'server_is_up': f'infra_tikv_server_is_up_{date}.parquet',
            'available_size': f'infra_tikv_available_size_{date}.parquet',
            'raft_propose_wait': f'infra_tikv_raft_propose_wait_{date}.parquet',
            'raft_apply_wait': f'infra_tikv_raft_apply_wait_{date}.parquet',
            'rocksdb_write_stall': f'infra_tikv_rocksdb_write_stall_{date}.parquet'
        }
    }


def get_tidb_services_directories() -> Dict[str, str]:
    """
    获取TiDB服务的数据目录映射

    返回:
        服务名到目录路径的映射字典
    """
    return {
        'tidb-tidb': 'infra/infra_tidb',
        'tidb-pd': 'other',
        'tidb-tikv': 'other'
    }


def get_tidb_core_metrics() -> Dict[str, List[str]]:
    """
    获取TiDB服务的核心指标列表（基于您的筛选建议）

    返回:
        服务名到核心指标列表的映射字典
    """
    return {
        'tidb-tidb': [
            'failed_query_ops',  # 失败请求数 - 错误率指标
            'duration_99th',  # 99分位请求延迟 - 关键性能指标
            'connection_count',  # 连接数 - 负载指标
            'server_is_up',  # 服务存活节点数 - 可用性指标
            'cpu_usage',  # CPU使用率 - 资源饱和度
            'memory_usage'  # 内存使用量 - 资源使用
        ],
        'tidb-pd': [
            'store_up_count',  # 健康Store数量 - 集群健康度
            'store_down_count',  # Down Store数量 - 故障指标
            'store_unhealth_count',  # Unhealth Store数量 - 异常指标
            'storage_used_ratio',  # 已用容量比 - 容量指标
            'cpu_usage',  # CPU使用率 - 资源使用
            'memory_usage'  # 内存使用量 - 资源使用
        ],
        'tidb-tikv': [
            'cpu_usage',  # CPU使用率 - 资源使用
            'memory_usage',  # 内存使用量 - 资源使用
            'server_is_up',  # 服务存活节点数 - 可用性
            'available_size',  # 可用存储容量 - 容量预警
            'raft_propose_wait',  # RaftPropose等待延迟P99 - 性能指标
            'raft_apply_wait',  # RaftApply等待延迟P99 - 性能指标
            'rocksdb_write_stall'  # RocksDB写阻塞次数 - 关键异常指标
        ]
    }


def load_tidb_service_data(date: str, service_name: str, metric_name: str) -> Optional[pd.DataFrame]:
    """
    加载指定TiDB服务的指标数据

    参数:
        date: 日期，格式如 "2025-06-06"
        service_name: 服务名称，如 "tidb-tidb"
        metric_name: 指标名称，如 "cpu_usage"

    返回:
        TiDB服务指标数据DataFrame，如果文件不存在则返回None
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 获取目录映射
    directories = get_tidb_services_directories()
    if service_name not in directories:
        print(f"未知的TiDB服务名称: {service_name}")
        return None

    # 构建数据目录路径
    data_dir = os.path.join(project_root, 'data', 'processed', f'{date}', 'metric-parquet', directories[service_name])

    # 获取文件映射
    file_mapping = get_tidb_services_files_mapping(date)
    if service_name not in file_mapping or metric_name not in file_mapping[service_name]:
        print(f"未找到服务 {service_name} 的指标 {metric_name} 的文件映射")
        return None

    file_path = os.path.join(data_dir, file_mapping[service_name][metric_name])

    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        df = pd.read_parquet(file_path)

        if len(df) == 0:
            print(f"文件 {file_path} 中无数据")
            return None

        return df

    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None


def get_tidb_metrics_description_with_time_filter(df_tidb: pd.DataFrame, start_time: str, end_time: str,
                                                  metric_column: str, remove_outliers: bool = False) -> Optional[
    pd.Series]:
    """
    获取指定时间范围内TiDB指标的描述统计

    参数:
        df_tidb: TiDB指标数据DataFrame
        start_time: 开始时间戳
        end_time: 结束时间戳
        metric_column: 指标列名（实际数值列）
        remove_outliers: 是否移除异常值

    返回:
        指标描述统计信息，如果无数据则返回None
    """
    if 'timestamp_ns' not in df_tidb.columns:
        print("警告: 未找到timestamp_ns列，无法进行时间过滤")
        return None

    # 时间过滤
    start_ts = int(start_time)
    end_ts = int(end_time)
    df_filtered = df_tidb[(df_tidb['timestamp_ns'] >= start_ts) & (df_tidb['timestamp_ns'] <= end_ts)]

    if len(df_filtered) == 0:
        print("指定时间范围内无数据")
        return None

    # 获取指标数据
    if metric_column not in df_filtered.columns:
        print(f"警告: 列 '{metric_column}' 不存在于DataFrame中")
        return None

    metric_data = df_filtered[metric_column].dropna()

    if len(metric_data) == 0:
        print(f"指标 '{metric_column}' 无有效数据")
        return None

    # 是否移除异常值
    if remove_outliers and len(metric_data) > 4:
        metric_data_sorted = metric_data.sort_values()
        metric_data = metric_data_sorted.iloc[2:-2]  # 去掉最小2个和最大2个
        print(f"移除异常值后数据点数量: {len(metric_data)}")
    desc = metric_data.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])

    # **新增非零比例**
    desc['non_zero_ratio'] = round((metric_data != 0).sum() / len(metric_data), 3)

    return desc


def analyze_tidb_services_metrics(df_fault_timestamps: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    分析TiDB服务在故障时间段与正常时间段的指标对比
    结构：service → metric → {normal_periods_combined, fault_period}

    参数:
        df_fault_timestamps: 故障时间戳DataFrame
        index: 要分析的故障索引

    返回:
        按TiDB服务组织的包含故障和正常时间段指标对比的字典
    """
    # 获取故障时间信息
    _, date, fault_start, fault_end = get_fault_period_info(df_fault_timestamps, index)
    normal_periods = get_normal_time_periods(df_fault_timestamps, index)

    # 获取TiDB服务和核心指标
    core_metrics = get_tidb_core_metrics()

    print(f"TiDB服务分析 - 故障日期: {date}")
    print(f"TiDB服务分析 - 故障时间段: {fault_start} 至 {fault_end}")
    print(f"TiDB服务分析 - 正常时间段数量: {len(normal_periods)}")
    print(f"TiDB服务分析 - 分析服务数量: {len(core_metrics)}")

    # 按 服务 → 指标 → 时间段 结构组织分析结果
    tidb_analysis = {}

    for service_name, metrics_list in core_metrics.items():
        print(f"\n=== 处理TiDB服务: {service_name} ===")

        # 初始化服务结构
        tidb_analysis[service_name] = {}

        for metric_name in metrics_list:
            print(f"  处理指标: {metric_name}")

            # 加载该指标的数据
            df_metric = load_tidb_service_data(date, service_name, metric_name)

            if df_metric is None:
                print(f"    无法加载指标 {metric_name} 的数据，跳过")
                continue

            # 初始化指标结构
            tidb_analysis[service_name][metric_name] = {
                'normal_periods_combined': None,
                'fault_period': None
            }

            # 1. 合并所有正常时间段数据进行统计
            print(f"    正常时间段分析:")
            all_normal_data = []

            for i, (normal_start, normal_end) in enumerate(normal_periods):
                start_ts = int(normal_start)
                end_ts = int(normal_end)
                normal_data = df_metric[(df_metric['timestamp_ns'] >= start_ts) & (df_metric['timestamp_ns'] <= end_ts)]

                if len(normal_data) > 0:
                    all_normal_data.append(normal_data)
                    print(f"      时间段 {i + 1} 数据行数: {len(normal_data)}")

            # 合并正常时间段数据并统计
            if all_normal_data:
                combined_normal_data = pd.concat(all_normal_data, ignore_index=True)
                print(f"    合并后正常时间段总数据行数: {len(combined_normal_data)}")

                # 获取统计（移除异常值）
                normal_desc = get_tidb_metrics_description_with_time_filter(
                    combined_normal_data,
                    str(combined_normal_data['timestamp_ns'].min()),
                    str(combined_normal_data['timestamp_ns'].max()),
                    metric_name,
                    remove_outliers=(len(combined_normal_data) > 4)
                )

                tidb_analysis[service_name][metric_name]['normal_periods_combined'] = normal_desc

                if normal_desc is not None:
                    print(f"    正常期间 {metric_name}: mean={normal_desc['mean']:.2f}, std={normal_desc['std']:.2f}")

            # 2. 故障时间段统计
            print(f"    故障时间段分析:")
            fault_desc = get_tidb_metrics_description_with_time_filter(
                df_metric, fault_start, fault_end, metric_name, remove_outliers=False
            )

            tidb_analysis[service_name][metric_name]['fault_period'] = fault_desc

            if fault_desc is not None:
                fault_data_count = len(df_metric[(df_metric['timestamp_ns'] >= int(fault_start)) &
                                                 (df_metric['timestamp_ns'] <= int(fault_end))])
                print(f"    故障时间段数据行数: {fault_data_count}")
                print(f"    故障期间 {metric_name}: mean={fault_desc['mean']:.2f}, std={fault_desc['std']:.2f}")

    return tidb_analysis if tidb_analysis else None


def create_combined_service_prompt_with_tidb(service_analysis_results: Dict, tidb_analysis_results: Dict = None) -> str:
    """
    合并所有服务（包括TiDB服务）的prompt为一个综合prompt

    参数:
        service_analysis_results: 普通服务分析结果
        tidb_analysis_results: TiDB服务分析结果

    返回:
        合并后的综合prompt
    """
    if not service_analysis_results and not tidb_analysis_results:
        return ""

    # 构建合并的数据对比表格
    combined_json = {}

    # 1. 处理普通服务
    if service_analysis_results:
        for service_name, service_data in service_analysis_results.items():
            combined_json[service_name] = {
                "service_name": service_name,
                "service_type": "microservice",  # 标记为微服务
                "pod_count": len(service_data),
                "pod_list": list(service_data.keys()),
                "pods": {}
            }

            for pod_name, pod_metrics in service_data.items():
                normal_stats = pod_metrics.get('normal_periods_combined', {})
                fault_stats = pod_metrics.get('fault_period', {})

                pod_json = {}
                target_columns = ['client_error_ratio', 'error_ratio', 'request', 'response',
                                  'rrt', 'server_error_ratio', 'timeout']

                for metric in target_columns:
                    normal_desc = normal_stats.get(metric)
                    fault_desc = fault_stats.get(metric)

                    if normal_desc is not None and fault_desc is not None:
                        normal_mean = normal_desc.get('mean', 0)
                        normal_std = normal_desc.get('std', 0)
                        normal_max = normal_desc.get('max', 0)
                        normal_p99 = normal_desc.get('99%', 0)
                        normal_p50 = normal_desc.get('50%', 0)
                        normal_p25 = normal_desc.get('25%', 0)
                        normal_p75 = normal_desc.get('75%', 0)
                        fault_mean = fault_desc.get('mean', 0)
                        fault_std = fault_desc.get('std', 0)
                        fault_max = fault_desc.get('max', 0)
                        fault_p99 = fault_desc.get('99%', 0)
                        fault_p50 = fault_desc.get('50%', 0)
                        fault_p25 = fault_desc.get('25%', 0)
                        fault_p75 = fault_desc.get('75%', 0)
                        normal_iqr = normal_p75 - normal_p25
                        fault_iqr = fault_p75 - fault_p25
                        fault_nzr = fault_desc.get('non_zero_ratio', 0)
                        normal_nzr = normal_desc.get('non_zero_ratio', 0)
                        # 计算对称比率
                        p50_symmetric_ratio = abs(fault_p50 - normal_p50) / (
                                (fault_p50 + normal_p50) / 2 + 1e-9
                        )
                        p99_symmetric_ratio = abs(fault_p99 - normal_p99) / (
                                (fault_p99 + normal_p99) / 2 + 1e-9
                        )
                        # 过滤阈值（建议 0.5 或 1，根据需求调整）
                        if p50_symmetric_ratio < 0.05 and p99_symmetric_ratio < 0.05:
                            continue

                        pod_json[metric] = {
                            "正常期间中位数": round(normal_p50, 2),
                            "正常期间四分位距": round(normal_iqr, 2),
                            "正常期间99分位数": round(normal_p99, 2),
                            "故障期间中位数": round(fault_p50, 2),
                            "故障期间四分位距": round(fault_iqr, 2),
                            "故障期间99分位数": round(fault_p99, 2)
                        }

                combined_json[service_name]["pods"][pod_name] = pod_json

    # 2. 处理TiDB服务
    if tidb_analysis_results:
        for service_name, service_metrics in tidb_analysis_results.items():
            combined_json[service_name] = {
                "service_name": service_name,
                "service_type": "tidb_component",  # 标记为TiDB组件
                "metrics": {}  # TiDB服务直接存储指标，没有Pod概念
            }

            for metric_name, metric_stats in service_metrics.items():
                normal_stats = metric_stats.get('normal_periods_combined')
                fault_stats = metric_stats.get('fault_period')

                if normal_stats is not None and fault_stats is not None:
                    normal_mean = normal_stats.get('mean', 0)
                    normal_std = normal_stats.get('std', 0)
                    normal_max = normal_stats.get('max', 0)
                    normal_p99 = normal_stats.get('99%', 0)
                    normal_p50 = normal_stats.get('50%', 0)
                    normal_p25 = normal_stats.get('25%', 0)
                    normal_p75 = normal_stats.get('75%', 0)

                    fault_mean = fault_stats.get('mean', 0)
                    fault_std = fault_stats.get('std', 0)
                    fault_max = fault_stats.get('max', 0)
                    fault_p99 = fault_stats.get('99%', 0)
                    fault_p50 = fault_stats.get('50%', 0)
                    fault_p25 = fault_stats.get('25%', 0)
                    fault_p75 = fault_stats.get('75%', 0)
                    normal_iqr = normal_p75 - normal_p25
                    fault_iqr = fault_p75 - fault_p25
                    fault_nzr = fault_stats.get('non_zero_ratio', 0)
                    normal_nzr = normal_stats.get('non_zero_ratio', 0)
                    # 计算对称比率
                    p50_symmetric_ratio = abs(fault_p50 - normal_p50) / (
                            (fault_p50 + normal_p50) / 2 + 1e-9
                    )
                    p99_symmetric_ratio = abs(fault_p99 - normal_p99) / (
                            (fault_p99 + normal_p99) / 2 + 1e-9
                    )
                    # 过滤阈值（建议 0.5 或 1，根据需求调整）
                    if p50_symmetric_ratio < 0.05 and p99_symmetric_ratio < 0.05:
                        continue

                    combined_json[service_name]["metrics"][metric_name] = {
                        "正常期间中位数": round(normal_p50, 2),
                        "正常期间四分位距": round(normal_iqr, 2),
                        "正常期间99分位数": round(normal_p99, 2),
                        "故障期间中位数": round(fault_p50, 2),
                        "故障期间四分位距": round(fault_iqr, 2),
                        "故障期间99分位数": round(fault_p99, 2)
                    }
                else:
                    combined_json[service_name]["metrics"][metric_name] = "缺失数据"

    # 统计信息
    all_services = list(combined_json.keys())
    microservice_count = sum(1 for s in combined_json.values() if s.get("service_type") == "microservice")
    tidb_service_count = sum(1 for s in combined_json.values() if s.get("service_type") == "tidb_component")
    total_pods = sum(s.get("pod_count", 0) for s in combined_json.values() if "pod_count" in s)

    return f"""
请根据提供的APM（应用性能监控）指标数据和TiDB分布式数据库指标数据，描述所有服务在正常期间和故障期间的业务服务性能表现差异现象。

## 系统整体信息
- **微服务总数**: {microservice_count}
- **TiDB组件数**: {tidb_service_count}
- **服务总数**: {len(all_services)}
- **Pod总数**: {total_pods}
- **服务列表**: {all_services}

## APM关键指标说明（微服务）
### 请求响应类指标：
- `request`: 请求数量 - 反映服务接收到的业务请求总数
- `response`: 响应数量 - 反映服务成功处理并响应的请求总数
- `rrt`: 平均时延 - 反映服务处理请求的平均响应时间

### 异常类指标：
- `timeout`: 超时数量 - 反映服务处理请求超时的次数
- `error_ratio`: 异常比例 - 异常请求占总请求的比例
- `client_error_ratio`: 客户端异常比例 - 客户端异常占总请求的比例
- `server_error_ratio`: 服务端异常比例 - 服务端异常占总请求的比例

## TiDB关键指标说明（数据库组件）
### TiDB组件指标：
- `failed_query_ops`: 失败请求数 - 数据库请求错误率指标
- `duration_99th`: 99分位请求延迟 - 数据库关键性能指标
- `connection_count`: 连接数 - 数据库负载指标
- `server_is_up`: 服务存活节点数 - 数据库可用性指标
- `cpu_usage`: CPU使用率 - 数据库资源饱和度
- `memory_usage`: 内存使用量 - 数据库资源使用

### TiKV组件指标：
- `cpu_usage`: CPU使用率 - 存储层资源使用
- `memory_usage`: 内存使用量 - 存储层资源使用
- `server_is_up`: 服务存活节点数 - 存储层可用性
- `available_size`: 可用存储容量 - 存储容量预警
- `raft_propose_wait`: RaftPropose等待延迟P99 - 分布式一致性性能
- `raft_apply_wait`: RaftApply等待延迟P99 - 分布式一致性性能
- `rocksdb_write_stall`: RocksDB写阻塞次数 - 存储引擎异常指标

### PD组件指标：
- `store_up_count`: 健康Store数量 - 集群健康度
- `store_down_count`: Down Store数量 - 集群故障指标
- `store_unhealth_count`: Unhealth Store数量 - 集群异常指标
- `storage_used_ratio`: 已用容量比 - 集群容量指标
- `cpu_usage`: CPU使用率 - 调度器资源使用
- `memory_usage`: 内存使用量 - 调度器资源使用

## 数据对比表格,特别注意**缺失数据和空数据,代表数据波动极小,通常情况默认正常**:
{json.dumps(combined_json, ensure_ascii=False, indent=2)}

## 现象描述要求
请从以下维度进行现象描述，**仅描述观察到的现象，不做异常判断或结论**：

### 微服务级别现象观察
- 集中在同一类型的微服务中出现的问题，比如emailservice-0, emailservice-1, emailservice-2, 都存在相似的异常数据变化
- 个别Pod的异常表现特征，比如cartservice-0中存在异常的数据变化，而cartservice-1, cartservice-2以及其他pod正常

### TiDB数据库组件现象观察
- TiDB组件（tidb-tidb）、TiKV组件（tidb-tikv）和 PD组件（tidb-pd）的异常数据变化

## 重要提示
**这是为后期综合决策分析提供的系统级现象总结，请控制总结内容在2000字左右，重点突出主要变化现象。**

**总结要求：**
- 如果整体表现正常稳定，请简要说明"系统各项指标表现稳定，未观察到显著变化现象"
- 如果出现明显变化，请重点描述变化显著的服务、指标和现象
- 采用概括性语言，重点关注对业务影响较大的指标变化
- 缺失或为空的数据表示波动极小，可视为正常，无需描述

请基于APM业务监控数据和TiDB数据库监控数据客观描述观察到的现象，控制在2000字以内，为后续综合分析提供简洁有效的现象总结。
"""


def analyze_fault_comprehensive(df_fault_timestamps: pd.DataFrame, index: int, uuid: str = None) -> str:
    """
    对指定故障进行综合分析，包括Service级别、TiDB服务和Node级别的完整分析流程
    注意：TiDB服务数据会直接整合到第一次Service分析中，总共只调用2次LLM

    参数:
        df_fault_timestamps: 故障时间戳DataFrame
        index: 要分析的故障索引
        uuid: 样本UUID，用于记录大模型调用

    返回:
        node_analysis_result: 综合Node级别分析结果（包含Service和TiDB分析结果）
    """
    # 定义要分析的关键指标列
    key_metrics = ['client_error_ratio', 'error_ratio', 'request', 'response', 'rrt', 'server_error_ratio', 'timeout']

    # 定义要分析的节点指标
    node_metrics = ['node_cpu_usage_rate',
                    'node_disk_read_bytes_total',
                    'node_disk_read_time_seconds_total',
                    'node_disk_write_time_seconds_total',
                    'node_disk_written_bytes_total',
                    'node_filesystem_free_bytes',
                    'node_filesystem_usage_rate',
                    'node_memory_MemAvailable_bytes',
                    'node_memory_MemTotal_bytes',
                    'node_memory_usage_rate',
                    'node_network_receive_bytes_total',
                    'node_network_receive_packets_total',
                    'node_network_transmit_bytes_total',
                    'node_network_transmit_packets_total',
                    'node_sockstat_TCP_inuse']

    pod_metrics = [
        'pod_cpu_usage', 'pod_fs_reads_bytes', 'pod_fs_writes_bytes',
        'pod_memory_working_set_bytes', 'pod_network_receive_bytes',
        'pod_network_receive_packets', 'pod_network_transmit_bytes',
        'pod_network_transmit_packets', 'pod_processes'
    ]

    try:
        # 获取当前故障的日期
        fault_date = df_fault_timestamps.iloc[index]['date']

        print(f"开始分析故障索引: {index}")
        print("=" * 80)

        # ==================== SERVICE级别分析（包含TiDB服务）====================
        print(f"\n{'🔹' * 40}")
        print("SERVICE和TiDB服务级别数据分析（一次性处理）")
        print(f"{'🔹' * 40}")

        # 1. 分析普通微服务
        service_result = analyze_fault_vs_normal_metrics_by_service(df_fault_timestamps, index, key_metrics)
        if service_result is None:
            print("未找到匹配的Service指标数据")
            service_result = {}
        else:
            print(f"成功分析了 {len(service_result)} 个Service")

        # 2. 分析TiDB服务（直接整合）
        tidb_result = analyze_tidb_services_metrics(df_fault_timestamps, index)
        if tidb_result is None:
            print("未找到匹配的TiDB服务指标数据")
            tidb_result = {}
        else:
            print(f"成功分析了 {len(tidb_result)} 个TiDB服务")

        # 3. 创建合并的Service+TiDB prompt并调用LLM（第1次调用）
        combined_service_prompt = create_combined_service_prompt_with_tidb(service_result, tidb_result)
        print("已生成合并的Service和TiDB分析prompt")

        print("正在调用大模型分析Service和TiDB数据（第1次调用）...")
        service_analysis_result = call_llm_analysis(combined_service_prompt, uuid,
                                                    "第1次调用-Service和TiDB级别综合分析")
        print("Service和TiDB级别分析完成")

        # ==================== 获取Pod部署信息 ====================
        print(f"\n{'📋' * 40}")
        print("获取Pod部署信息")
        print(f"{'📋' * 40}")

        node_pod_mapping = get_node_pod_mapping(fault_date)
        if node_pod_mapping:
            total_pods = sum(len(pods) for pods in node_pod_mapping.values())
            print(f"成功获取Pod部署信息，总Pod数: {total_pods}")
            for node_name, pods in node_pod_mapping.items():
                print(f"  {node_name}: {len(pods)} 个Pod")
        else:
            print("未能获取Pod部署信息，将使用空的部署映射")

        # ==================== NODE级别分析 ====================
        print(f"\n{'🔷' * 40}")
        print("NODE级别数据分析")
        print(f"{'🔷' * 40}")

        pod_result = analyze_pod_metrics_by_pod(df_fault_timestamps, index, pod_metrics)
        node_result = analyze_node_metrics_by_node(df_fault_timestamps, index, node_metrics)

        if node_result is None:
            print("未找到匹配的Node指标数据")
            node_result = {}
        else:
            print(f"成功分析了 {len(node_result)} 个Node")

        # 创建包含Service+TiDB分析结果的Node prompt并调用LLM（第2次调用）
        combined_node_prompt = create_combined_node_prompt_with_service_analysis(
            node_result, pod_result, service_analysis_result, node_pod_mapping)
        print("已生成包含Service和TiDB分析结果的综合Node分析prompt")

        print("正在调用大模型进行综合Node分析（第2次调用）...")
        node_analysis_result = call_llm_analysis(combined_node_prompt, uuid, "第2次调用-Node级别综合分析")
        print("综合Node级别分析完成")

        print("=" * 80)
        print(f"故障索引 {index} 综合分析完成（包含TiDB服务，总计2次LLM调用）")
        print("=" * 80)

        return node_analysis_result

    except Exception as e:
        error_msg = f"分析故障索引 {index} 时出错: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)


# ==================== 更新主函数以支持TiDB ====================

if __name__ == "__main__":
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 使用绝对路径构建input_timestamp.csv的路径
    input_path = os.path.join(project_root, 'input', 'input_timestamp.csv')
    df_fault_timestamps = pd.read_csv(input_path)

    # 创建output目录
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 创建时间戳
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 限制只处理前2组数据进行验证（包含TiDB服务分析）
    MAX_PROCESS_COUNT = 2
    total_count = min(len(df_fault_timestamps), MAX_PROCESS_COUNT)

    # 统计变量
    successful_analysis_count = 0
    failed_analysis_count = 0

    print(f"开始处理前{MAX_PROCESS_COUNT}组数据进行综合LLM分析（包含TiDB服务）")
    print("=" * 100)

    for index, row in df_fault_timestamps.iterrows():
        # 限制只处理前MAX_PROCESS_COUNT组数据
        if index >= MAX_PROCESS_COUNT:
            print(f"已处理前{MAX_PROCESS_COUNT}组数据，停止处理")
            break

        print("=" * 100)
        print(f"处理故障索引: {index} (验证模式: {index + 1}/{MAX_PROCESS_COUNT}) - 包含TiDB服务分析")
        print("=" * 100)

        try:
            # 使用更新的包含TiDB服务的综合分析函数
            node_analysis_result = analyze_fault_comprehensive(df_fault_timestamps, index, None)

            # ==================== 保存结果 ====================
            print(f"\n{'💾' * 50}")
            print("保存分析结果")
            print(f"{'💾' * 50}")

            # 创建分析结果目录
            analysis_output_dir = os.path.join(output_dir, 'llm_analysis')
            os.makedirs(analysis_output_dir, exist_ok=True)

            # 保存综合分析结果（包含TiDB）
            analysis_file_path = os.path.join(analysis_output_dir,
                                              f'fault_{index:03d}_comprehensive_with_tidb_analysis_{timestamp}.txt')
            with open(analysis_file_path, 'w', encoding='utf-8') as f:
                f.write(f"故障索引 {index} - 综合分析结果（包含TiDB服务）\n")
                f.write(f"生成时间: {timestamp}\n")
                f.write(f"分析内容: 微服务APM + TiDB数据库组件 + 基础设施监控\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(node_analysis_result)

            print(f"综合分析结果已保存: {analysis_file_path}")

            successful_analysis_count += 1
            print(f"故障索引 {index} 分析完成并保存（包含TiDB服务）")

        except Exception as e:
            error_msg = f"处理故障索引 {index} 时出错: {e}"
            print(error_msg)
            failed_analysis_count += 1

            # 保存错误信息
            error_file_path = os.path.join(output_dir, 'llm_analysis',
                                           f'fault_{index:03d}_error_with_tidb_{timestamp}.txt')
            os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
            with open(error_file_path, 'w', encoding='utf-8') as f:
                f.write(f"故障索引 {index} 处理错误（TiDB分析模式）\n")
                f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误信息: {str(e)}\n")

        print("=" * 100)
        print(f"故障索引 {index} 处理完成 ({index + 1}/{MAX_PROCESS_COUNT})")
        print("=" * 100)

    print("\n" + "=" * 100)
    print("📊 TiDB服务集成分析处理统计:")
    print(f"✅ 成功分析: {successful_analysis_count} 个")
    print(f"❌ 分析失败: {failed_analysis_count} 个")
    print(f"🎯 总处理数: {successful_analysis_count + failed_analysis_count} 个")
    print(f"🚀 LLM调用优化: 每个样本只调用2次（Service+TiDB一次，Node综合一次）")
    print("=" * 100)
