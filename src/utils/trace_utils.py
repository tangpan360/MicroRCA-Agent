import pandas as pd
import os
import glob
import sys
import random
import numpy as np
import time
import pickle
import re
from typing import Optional, List, Tuple, Dict, Set
from collections import defaultdict
from sklearn.ensemble import IsolationForest

# 添加项目根目录到系统路径，确保可以导入utils.io_util
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import utils.io_util as io

# ========== 超参数配置 ==========
# 训练相关参数
SAMPLE_SIZE = 50  # 抽样数量
RANDOM_SEED = 42  # 随机种子
MINUTES_AFTER = 40  # 异常结束后多少分钟的数据视为正常数据
N_ESTIMATORS = 100  # IsolationForest的估计器数量
CONTAMINATION = 0.01  # IsolationForest的污染率

# 滑动窗口参数
WIN_SIZE_SECONDS = 30  # 滑动窗口大小（秒）
WIN_SIZE_NS = WIN_SIZE_SECONDS * 1000000000  # 滑动窗口大小（纳秒）

# 统计分析参数
TOP_N_COMBINATIONS = 10  # 取前N种异常组合进行详细分析
BEIJING_TIMEZONE_OFFSET = 8  # 北京时间偏移（UTC+8）


def _get_period_info(df_input_timestamp: pd.DataFrame, row_index: int) -> tuple[list[str], int, int]:
    """
    获取指定行的匹配信息
    
    参数:
        df_input_timestamp: 包含故障起止时间戳的DataFrame, df_input_timestamp = pd.read_csv('input_timestamp.csv') 文件读取后的结果
        row_index: 指定要查询的行索引
        
    返回:
        匹配的文件列表, start_time, end_time
    """    
    import glob

    row = df_input_timestamp.iloc[row_index]
    start_time_hour = row['start_time_hour']
    start_time = row['start_timestamp']
    end_time = row['end_timestamp']
    
    # 使用os.path.join和项目根目录的相对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_pattern = os.path.join(project_root, 'data', 'processed', '*', 'trace-parquet', f'*{start_time_hour}*')
    matching_files = glob.glob(search_pattern, recursive=True)

    return matching_files, start_time, end_time


def _filter_traces_by_timerange(matching_files: list[str], start_time: int, end_time: int, df_trace: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    根据时间范围过滤trace数据
    
    参数:
        matching_files: 匹配的文件路径列表
        start_time: 开始时间戳
        end_time: 结束时间戳
        df_trace: 包含trace数据的DataFrame，如果为None则会尝试读取匹配的文件
        
    返回:
        DataFrame: 过滤后的trace数据，只包含时间范围内的行；如果没有匹配文件则返回None
    """
    import pandas as pd

    # 检查是否找到匹配的文件
    if not matching_files:
        print("未找到匹配的trace文件")
        return None

    # 使用时间戳过滤数据
    filtered_df = df_trace[(df_trace['timestamp_ns'] >= start_time) & (df_trace['timestamp_ns'] <= end_time)]

    return filtered_df


def _load_or_train_anomaly_detection_model() -> Optional[Dict[str, Dict[str, IsolationForest]]]:
    """
    加载或训练异常检测模型
    
    返回:
        Dict[str, Dict[str, IsolationForest]]: 异常检测模型字典，如果失败则返回None
    """
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detector_file = os.path.join(project_root, 'models', 'trace_detectors.pkl')
    
    # 如果模型文件已存在，直接加载
    if os.path.exists(detector_file):
        try:
            with open(detector_file, 'rb') as f:
                trace_detectors = pickle.load(f)
            print(f"成功加载现有异常检测模型，包含 {len(trace_detectors)} 个检测器")
            return trace_detectors
        except Exception as e:
            print(f"加载现有模型失败: {e}")
            return None
    
    # 如果模型文件不存在，进行训练
    print("异常检测模型文件不存在，开始训练...")
    try:
        # 处理抽样trace数据
        print("开始处理抽样trace数据...")
        merged_file = os.path.join(project_root, 'data', 'merged', 'merged_traces.parquet')
        
        # 调用主处理函数，抽取指定数量的trace文件，并将其合并保存为merged_traces.parquet并返回给变量merged_traces，从merged_traces中提取用于训练iforest的正常trace数据保存至train_iforest_normal_traces.pkl并返回给变量normal_traces
        merged_traces, normal_traces = _process_trace_samples(
            sample_size=SAMPLE_SIZE, random_seed=RANDOM_SEED, output_path=merged_file, minutes_after=MINUTES_AFTER
        )
        print(f"处理完成，合并后的数据包含 {len(merged_traces)} 行")
        print(f"用于训练iforest的正常trace数据包含 {len(normal_traces)} 组")
        
        # 训练异常检测模型
        trace_detectors, normal_stats = _train_anomaly_detection_model(normal_traces, output_path=detector_file)
        print(f"异常检测模型训练完成，包含 {len(trace_detectors)} 个检测器")
        
        return trace_detectors
        
    except Exception as e:
        print(f"训练异常检测模型失败: {e}")
        return None


def _extract_pod_name(process):
    """
    从process字典中提取'tags'列表里key为'name'或'podName'的value
    
    参数:
        process: 包含tags列表的字典
        
    返回:
        str: 提取的name或podName值，如果没有找到则返回None
    """
    if not isinstance(process, dict):
        return None
    
    tags = process.get('tags', [])
    for tag in tags:
        if tag.get('key') == 'name' or tag.get('key') == 'podName':
            return tag.get('value')
    
    return None


def _extract_service_name(process):
    """
    从process字典中提取'serviceName'字段

    参数:
        process: 包含serviceName的字典

    返回:
        str: 提取的serviceName值，如果没有找到则返回None
    """
    if not isinstance(process, dict):
        return None
    return process.get('serviceName', None)


def _extract_node_name(process):
    """
    从process字典中提取'tags'列表里key为'node_name'或'nodeName'的value

    参数:
        process: 包含tags列表的字典

    返回:
        str: 提取的node_name或nodeName值，如果没有找到则返回None
    """
    if not isinstance(process, dict):
        return None

    tags = process.get('tags', [])
    for tag in tags:
        if tag.get('key') == 'node_name' or tag.get('key') == 'nodeName':
            return tag.get('value')
    return None


def _extract_parent_spanid(ref):
    """
    从references中提取父spanID
    
    参数:
        ref: 包含父spanID的引用数组
        
    返回:
        str: 父spanID，如果没有找到则返回None
    """
    if isinstance(ref, np.ndarray) and ref.size == 1 and isinstance(ref[0], dict) and 'spanID' in ref[0]:
        return ref[0]['spanID']
    return None


def _extract_status_keys_and_values(tags_str: str) -> Tuple[Set[str], Dict[str, str]]:
    """
    从tags字符串中提取status相关的key和对应的value
    
    Returns:
        keys: status相关的key集合
        values: key到value的映射
    """
    try:
        # 提取所有包含status的key及其对应的value
        key_pattern = r"'key':\s*'([^']*status[^']*)'.*?'value':\s*'([^']*)'"
        matches = re.findall(key_pattern, tags_str, re.IGNORECASE)
        
        keys = set()
        values = {}
        
        for key, value in matches:
            keys.add(key)
            values[key] = value
            
        return keys, values
    except Exception as e:
        print(f"解析tags失败: {e}")
        return set(), {}


def _analyze_status_combinations_in_fault_period(df_filtered_traces: pd.DataFrame) -> str:
    """
    分析故障期间status.code和status.message的组合情况，包含详细的上下文信息
    
    参数:
        df_filtered_traces: 故障期间的trace数据（已经预处理过，包含pod信息）
        
    返回:
        status组合统计的CSV格式字符串，包含node_name, service_name, parent_pod, child_pod, operation_name等详细信息
    """
    print("开始分析故障期间的status组合...")
    
    # 筛选包含status的行
    status_logs = df_filtered_traces[df_filtered_traces['tags'].astype(str).str.contains("status", case=False, na=False)]
    
    if len(status_logs) == 0:
        print("故障期间没有包含status的记录")
        return ""
    
    print(f"故障期间找到 {len(status_logs)} 条包含status的记录")
    
    # 收集详细的status组合信息
    status_details = []
    
    # 处理每一行
    for _, row in status_logs.iterrows():
        keys, values = _extract_status_keys_and_values(str(row['tags']))
        
        # 检查是否包含status.code和status.message
        if 'status.code' in keys and 'status.message' in keys:
            status_code = values.get('status.code', 'N/A')
            status_message = values.get('status.message', 'N/A')
            
            # 过滤掉status.code为0的正常情况
            if status_code == '0':
                continue
            
            # 提取上下文信息
            node_name = row.get('node_name', 'N/A')
            service_name = row.get('service_name', 'N/A')
            parent_pod = row.get('parent_pod', 'N/A')
            child_pod = row.get('child_pod', 'N/A')
            operation_name = row.get('operationName', 'N/A')
            
            # 服务名替换：redis -> redis-cart
            if service_name == 'redis':
                service_name = 'redis-cart'
            
            # 处理None值
            node_name = str(node_name) if node_name is not None else "N/A"
            service_name = str(service_name) if service_name is not None else "N/A"
            parent_pod = str(parent_pod) if parent_pod is not None else "N/A"
            child_pod = str(child_pod) if child_pod is not None else "N/A"
            operation_name = str(operation_name) if operation_name is not None else "N/A"
            
            status_details.append({
                'status_code': status_code,
                'status_message': status_message,
                'node_name': node_name,
                'service_name': service_name,
                'parent_pod': parent_pod,
                'child_pod': child_pod,
                'operation_name': operation_name
            })
    
    if not status_details:
        print("没有找到非正常的status组合")
        return ""
    
    # 转换为DataFrame
    status_df = pd.DataFrame(status_details)
    
    # 统计相同组合的出现次数
    combination_columns = ['node_name', 'service_name', 'parent_pod', 'child_pod', 
                          'operation_name', 'status_code', 'status_message']
    
    # 分组统计
    grouped = status_df.groupby(combination_columns).size().reset_index(name='occurrence_count')
    
    # 按出现次数降序排列，取前20个（在添加文字之前排序）
    grouped = grouped.sort_values('occurrence_count', ascending=False).head(20)
    
    # 添加文字描述到次数列
    grouped['occurrence_count_display'] = grouped['occurrence_count'].apply(lambda x: f"出现次数:{x}")
    
    # 删除原始数字列，重命名显示列
    grouped = grouped.drop('occurrence_count', axis=1)
    grouped = grouped.rename(columns={'occurrence_count_display': 'occurrence_count'})
    
    # 调整列顺序为指定的顺序
    desired_column_order = ['node_name', 'service_name', 'parent_pod', 'child_pod', 
                           'operation_name', 'status_code', 'status_message', 'occurrence_count']
    
    # 确保只包含存在的列，并按指定顺序排列
    existing_columns = [col for col in desired_column_order if col in grouped.columns]
    grouped = grouped[existing_columns]
    
    print(f"找到 {len(grouped)} 种不同的status组合（包含上下文信息，显示前20个）")
    
    return grouped.to_csv(index=False)


def _sample_timestamp_data(sample_size: int = 50, random_seed: int = 42) -> pd.DataFrame:
    """
    从input_timestamp.csv中随机抽取指定数量的样本
    
    参数:
        sample_size: 要抽取的样本数量，默认为50
        random_seed: 随机种子，默认为42
        
    返回:
        DataFrame: 抽取的样本数据
    """
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 使用绝对路径构造input_timestamp.csv的路径
    input_path = os.path.join(project_root, 'input', 'input_timestamp.csv')
    df_input_timestamp = pd.read_csv(input_path)
    
    # 设置随机种子并抽样
    random.seed(random_seed)
    if sample_size >= len(df_input_timestamp):
        print(f"请求的样本数量({sample_size})大于或等于总样本数量({len(df_input_timestamp)})，返回全部数据")
        return df_input_timestamp
    
    sampled_df = df_input_timestamp.sample(n=sample_size, random_state=random_seed)
    print(f"从总共{len(df_input_timestamp)}个样本中随机抽取了{len(sampled_df)}个样本")
    
    return sampled_df


def _match_trace_files(sampled_df: pd.DataFrame) -> List[str]:
    """
    根据抽样数据匹配对应的trace文件
    
    参数:
        sampled_df: 抽样后的DataFrame
        
    返回:
        List[str]: 匹配到的文件路径列表
    """
    matched_trace_files = []
    
    for index, row in sampled_df.iterrows():
        start_time_hour = row['@start_time_hour']
        
        # 使用os.path.join和项目根目录的相对路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        search_pattern = os.path.join(project_root, 'data', 'phaseone', 'processed', '*', 'trace-parquet', f'*{start_time_hour}*')
        matching_file = glob.glob(search_pattern, recursive=True)
        
        if matching_file:
            matched_trace_files.append(matching_file[0])
            print(f"样本 {index}: 匹配到文件 {matching_file[0]}")
        else:
            print(f"样本 {index}: 未找到匹配文件")
    
    print(f"总共匹配到 {len(matched_trace_files)} 个文件")
    return matched_trace_files


def _merge_trace_files(matched_trace_files: List[str]) -> pd.DataFrame:
    """
    合并匹配到的trace文件
    
    参数:
        matched_trace_files: 匹配到的文件路径列表
        
    返回:
        DataFrame: 合并后的数据
    """
    all_traces = []
    
    for i, file_path in enumerate(matched_trace_files):
        try:
            df_trace = pd.read_parquet(file_path)
            
            # 添加来源文件信息
            df_trace['source_file'] = os.path.basename(file_path)
            all_traces.append(df_trace)
            print(f"文件 {i+1}/{len(matched_trace_files)}: 添加了 {len(df_trace)} 行数据")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    if not all_traces:
        print("没有有效的trace数据可合并")
        return pd.DataFrame()
    
    # 合并所有数据
    merged_df = pd.concat(all_traces, ignore_index=True)
    print(f"合并后的数据总行数: {len(merged_df)}")
    
    return merged_df


def _extract_normal_traces(sampled_df: pd.DataFrame, merged_df: pd.DataFrame, minutes_after: int = 40) -> Dict[str, List[pd.DataFrame]]:
    """
    从合并后的trace数据中提取正常时期的trace数据，并构建字典
    
    参数:
        sampled_df: 抽样后的DataFrame，包含end_time信息
        merged_df: 合并后的trace数据
        minutes_after: 异常结束后多少分钟的数据视为正常数据，默认40分钟
        
    返回:
        Dict[str, List[pd.DataFrame]]: 正常trace数据字典，key为parent_name-service_name-operationName，value为对应的duration列表
    """
    print(f"\n提取正常时期的trace数据（异常结束后{minutes_after}分钟）...")
    
    # 创建默认字典，值为列表
    normal_traces = defaultdict(list)
    
    # 纳秒转换为分钟的系数
    ns_to_min = 60 * 1000000000
    
    # 遍历每个样本
    for _, row in sampled_df.iterrows():
        end_time = row['end_time']
        normal_start_time = end_time  # 正常数据的开始时间是异常的结束时间
        normal_end_time = normal_start_time + minutes_after * ns_to_min  # 正常数据的结束时间
        
        print(f"处理样本: 正常时间范围 {pd.to_datetime(normal_start_time, unit='ns')} 到 {pd.to_datetime(normal_end_time, unit='ns')}")
        
        # 筛选正常时期的数据
        normal_df = merged_df[(merged_df['timestamp_ns'] >= normal_start_time) & (merged_df['timestamp_ns'] <= normal_end_time)]
        
        if normal_df.empty:
            print(f"警告: 在正常时期未找到数据")
            continue
            
        print(f"找到 {len(normal_df)} 条正常时期的数据")
        
        # 按parent_pod, child_pod, operationName分组
        trace_gp = normal_df.groupby(['parent_pod', 'child_pod', 'operationName'])
        
        # 遍历每个组，构建字典
        for (src, dst, op), call_df in trace_gp:
            # 处理None值
            src_str = str(src) if src is not None else "None"
            dst_str = str(dst) if dst is not None else "None"
            op_str = str(op) if op is not None else "None"
            
            name = f"{src_str}_{dst_str}_{op_str}"
            normal_traces[name].append(call_df)
            print(f"添加组 {name}: {len(call_df)} 条数据")
    
    # 打印统计信息
    print(f"\n正常trace统计信息:")
    print(f"总组数: {len(normal_traces)}")
    total_calls = sum(len(df) for dfs in normal_traces.values() for df in dfs)

    
    return normal_traces


def _slide_window(df: pd.DataFrame, win_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用滑动窗口计算持续时间的均值
    
    参数:
        df: 包含时间戳和持续时间的DataFrame
        win_size: 窗口大小（纳秒）
        
    返回:
        Tuple[np.ndarray, np.ndarray]: 窗口开始时间和对应的持续时间均值
    """
    window_start_times, durations = [], []
    
    time_min, time_max = df['timestamp_ns'].min(), df['timestamp_ns'].max()
    
    # 滑动窗口
    i = time_min
    while i < time_max:
        temp_df = df[(df['timestamp_ns'] >= i) & (df['timestamp_ns'] <= i + win_size)]
        
        if temp_df.empty:
            i += win_size
            continue
        
        window_start_times.append(i)
        durations.append(temp_df['duration'].mean())
        i += win_size
    
    return np.array(window_start_times), np.array(durations)


def _train_anomaly_detection_model(normal_traces: Dict[str, List[pd.DataFrame]], output_path: Optional[str] = None) -> Tuple[Dict[str, Dict[str, IsolationForest]], Dict[str, Dict[str, float]]]:
    """
    训练异常检测模型，只使用duration字段
    
    参数:
        normal_traces: 正常trace数据字典，key为service_name，value为对应的DataFrame列表
        output_path: 输出文件路径，如果不为None则保存模型
        
    返回:
        Tuple[Dict[str, Dict[str, IsolationForest]], Dict[str, Dict[str, float]]]: 
            - 训练好的异常检测模型字典
            - 正常数据的统计信息字典
    """
    print("\n开始训练异常检测模型...")
    
    # 创建异常检测器字典和统计信息字典
    trace_detectors = {}
    normal_stats = {}
    
    # 遍历每个服务调用组
    for name, call_dfs in normal_traces.items():
        start_time = time.time()
        print(f"处理组 {name}...")
        
        # 为每个组创建异常检测器
        trace_detectors[name] = {
            'dur_detector': IsolationForest(random_state=RANDOM_SEED, n_estimators=N_ESTIMATORS, contamination=CONTAMINATION)
        }
        
        # 收集训练数据
        train_ds = []
        for call_df in call_dfs:
            # 使用滑动窗口提取持续时间特征
            _, durs = _slide_window(call_df, WIN_SIZE_NS)
            train_ds.extend(durs)
        
        # 如果没有足够的训练数据，跳过
        if len(train_ds) == 0:
            print(f"警告: 组 {name} 没有足够的训练数据")
            continue
        
        # 计算正常数据的统计信息
        train_ds_array = np.array(train_ds)
        normal_stats[name] = {
            'mean': float(np.mean(train_ds_array)),
            'std': float(np.std(train_ds_array)),
            'median': float(np.median(train_ds_array)),
            'min': float(np.min(train_ds_array)),
            'max': float(np.max(train_ds_array)),
            'count': len(train_ds_array)
        }
        
        # 训练持续时间异常检测器
        print(f"训练组 {name} 的异常检测器，使用 {len(train_ds)} 个样本")
        print(f"  正常数据统计: 平均值={normal_stats[name]['mean']:.2f}, 标准差={normal_stats[name]['std']:.2f}")
        
        # 设置[name]的['dur_detector']是为了保留其他可能的检测器，比如后期添加['another_detector]
        dur_clf = trace_detectors[name]['dur_detector']
        dur_clf.fit(train_ds_array.reshape(-1, 1))
        trace_detectors[name]['dur_detector'] = dur_clf
        end_time = time.time()
        print(f"训练组 {name} 的异常检测器耗时: {end_time - start_time:.2f}秒")
        
    # 保存模型和统计信息
    if output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存模型
        with open(output_path, 'wb') as f:
            pickle.dump(trace_detectors, f)
        print(f"异常棆测模型已保存至: {output_path}")
        
        # 保存统计信息
        stats_path = output_path.replace('.pkl', '_normal_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(normal_stats, f)
        print(f"正常数据统计信息已保存至: {stats_path}")
    
    # 打印统计信息
    print(f"\n异常检测模型统计信息:")
    print(f"总组数: {len(trace_detectors)}")
    print(f"正常数据统计组数: {len(normal_stats)}")
    
    return trace_detectors, normal_stats


def _detect_anomalies(df: pd.DataFrame, trace_detectors: Dict[str, Dict[str, IsolationForest]]) -> List[List[str]]:
    """
    使用训练好的模型检测异常
    
    参数:
        df: 待检测的trace数据
        trace_detectors: 训练好的异常检测模型字典
        
    返回:
        List[List[str]]: 检测到的异常事件列表
    """
    print("\n开始检测异常...")
    
    # 创建事件列表
    events = []
    
    # 确保数据按时间戳排序
    df = df.sort_values(by='timestamp_ns', ascending=True)
    
    # 按parent_pod, child_pod, operationName分组
    gp = df.groupby(['parent_pod', 'child_pod', 'operationName'])
    
    # 遍历每个组
    for (parent_pod, child_pod, operation_name), call_df in gp:
        # 处理None值
        parent_pod_str = str(parent_pod) if parent_pod is not None else "None"
        child_pod_str = str(child_pod) if child_pod is not None else "None"
        operation_name_str = str(operation_name) if operation_name is not None else "None"
        
        name = f"{parent_pod_str}_{child_pod_str}_{operation_name_str}"
        
        # 检查是否有对应的检测器
        if name not in trace_detectors:
            print(f"警告: 组 {name} 没有对应的异常检测器")
            continue
        
        # 使用滑动窗口提取特征
        test_window_start_times, test_durations = _slide_window(call_df, WIN_SIZE_NS)
        
        # 如果没有足够的测试数据，跳过
        if len(test_durations) == 0:
            print(f"警告: 组 {name} 没有足够的测试数据")
            continue
        
        # 检测持续时间异常
        dur_detector = trace_detectors[name]['dur_detector']
        labels = dur_detector.predict(test_durations.reshape(-1, 1)).tolist()
        
        # 找到所有异常点
        anomaly_indices = [i for i, label in enumerate(labels) if label == -1]
        
        if anomaly_indices:
            print(f"在组 {name} 中检测到 {len(anomaly_indices)} 个持续时间异常")
            service_name = call_df['service_name'].iloc[0] if not call_df.empty and 'service_name' in call_df.columns else None
            node_name = call_df['node_name'].iloc[0] if not call_df.empty and 'node_name' in call_df.columns else None
            for idx in anomaly_indices:
                timestamp = test_window_start_times[idx]
                duration = test_durations[idx]
                events.append([timestamp, parent_pod_str, child_pod_str, operation_name_str, 'Duration', duration, service_name, node_name])
                print(f"  异常时间戳: {pd.to_datetime(timestamp, unit='ns')}, duration: {duration}")
        else:
            print(f"组 {name} 中未检测到异常")    
    
    print(f"总共检测到 {len(events)} 个异常事件")
    return events


def _process_trace_samples(sample_size: int = 50, random_seed: int = 42, output_path: Optional[str] = None, minutes_after: int = 40) -> Tuple[pd.DataFrame, Dict[str, List[pd.DataFrame]]]:
    """
    处理trace样本的主函数，包括抽样、匹配和合并
    
    参数:
        sample_size: 要抽取的样本数量，默认为50
        random_seed: 随机种子，默认为42
        output_path: 输出文件路径，如果不为None则保存合并后的数据
        minutes_after: 异常结束后多少分钟的数据视为正常数据，默认40分钟
        
    返回:
        Tuple[pd.DataFrame, Dict[str, List[pd.DataFrame]]]: 
            - 合并后的数据
            - 正常trace数据字典
    """
    # 步骤1: 随机抽样
    sampled_df = _sample_timestamp_data(sample_size, random_seed)
    
    # 步骤2: 匹配文件
    matched_files = _match_trace_files(sampled_df)
    
    # 步骤3: 合并文件
    merged_df = _merge_trace_files(matched_files)
    
    # 步骤4: 提取pod_name, service_name, node_name
    print("提取pod_name...")
    start_time = time.time()
    merged_df['pod_name'] = merged_df['process'].apply(_extract_pod_name)
    end_time = time.time()
    print(f"提取pod_name耗时: {end_time - start_time}秒")
    
    print("提取service_name...")
    start_time = time.time()
    merged_df['service_name'] = merged_df['process'].apply(_extract_service_name)
    end_time = time.time()
    print(f"提取service_name耗时: {end_time - start_time}秒")
    
    print("提取node_name...")
    start_time = time.time()
    merged_df['node_name'] = merged_df['process'].apply(_extract_node_name)
    end_time = time.time()
    print(f"提取node_name耗时: {end_time - start_time}秒")
    
    # 步骤5: 提取父spanID
    print("提取parent_spanID...")
    start_time = time.time()
    merged_df['parent_spanID'] = merged_df['references'].apply(_extract_parent_spanid)
    end_time = time.time()
    print(f"提取parent_spanID耗时: {end_time - start_time}秒")
    
    # 步骤6: 创建spanID到pod_name的映射
    print("创建spanID到pod_name的映射...")
    start_time = time.time()
    span_to_pod = dict(zip(merged_df['spanID'].tolist(), merged_df['pod_name'].tolist()))
    end_time = time.time()
    print(f"创建spanID到pod_name的映射耗时: {end_time - start_time}秒")
    
    # 步骤7: 提取父spanID对应的pod_name
    print("提取parent_pod...")
    start_time = time.time()
    merged_df['parent_pod'] = merged_df['parent_spanID'].map(lambda x: span_to_pod.get(x))
    end_time = time.time()
    print(f"提取parent_pod耗时: {end_time - start_time}秒")
    
    # 步骤8: 创建spanID到service_name的映射
    print("创建spanID到service_name的映射...")
    start_time = time.time()
    span_to_service = dict(zip(merged_df['spanID'].tolist(), merged_df['service_name'].tolist()))
    end_time = time.time()
    print(f"创建spanID到service_name的映射耗时: {end_time - start_time}秒")
    
    # 步骤9: 提取父spanID对应的service_name
    print("提取parent_service...")
    start_time = time.time()
    merged_df['parent_service'] = merged_df['parent_spanID'].map(lambda x: span_to_service.get(x))
    end_time = time.time()
    print(f"提取parent_service耗时: {end_time - start_time}秒")
    
    # 步骤10: 重命名列以匹配新的命名规范
    print("重命名列...")
    merged_df = merged_df.rename(columns={'pod_name': 'child_pod'})
    
    # 步骤11: 按时间戳排序
    print("按时间戳排序...")
    start_time = time.time()
    merged_df = merged_df.sort_values(by='timestamp_ns')
    end_time = time.time()
    print(f"按时间戳排序耗时: {end_time - start_time}秒")
    
    # 步骤12: 重新排列列顺序，将相关列放在一起
    print("重新排列列顺序...")
    columns = merged_df.columns.tolist()
    
    # 获取需要放在一起的列的索引
    spanid_idx = columns.index('spanID')
    parent_spanid_idx = columns.index('parent_spanID')
    child_pod_idx = columns.index('child_pod')
    parent_pod_idx = columns.index('parent_pod')
    service_name_idx = columns.index('service_name')
    parent_service_idx = columns.index('parent_service')
    node_name_idx = columns.index('node_name')
    
    # 从列表中移除这些列
    for idx in sorted([spanid_idx, parent_spanid_idx, child_pod_idx, parent_pod_idx, 
                       service_name_idx, parent_service_idx, node_name_idx], reverse=True):
        columns.pop(idx)
    
    # 按照指定顺序重新插入这些列
    new_columns = ['parent_spanID', 'spanID', 'parent_pod', 'child_pod', 
                   'parent_service', 'service_name', 'node_name'] + columns
    merged_df = merged_df[new_columns]
    
    # 步骤13: 统计信息
    print("\n统计信息:")
    print(f"总行数: {len(merged_df)}")
    print(f"唯一child_pod数量: {merged_df['child_pod'].nunique()}")
    print(f"唯一service_name数量: {merged_df['service_name'].nunique()}")
    print(f"唯一node_name数量: {merged_df['node_name'].nunique()}")
    print(f"唯一spanID数量: {merged_df['spanID'].nunique()}")
    print(f"时间范围: {pd.to_datetime(merged_df['timestamp_ns'].min(), unit='ns')} 到 {pd.to_datetime(merged_df['timestamp_ns'].max(), unit='ns')}")
    
    # 步骤14: 提取正常时期的trace数据
    normal_traces = _extract_normal_traces(sampled_df, merged_df, minutes_after)
    
    # 步骤15: 保存数据
    if output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        start_time = time.time()
        merged_df.to_parquet(output_path)
        end_time = time.time()
        print(f"合并后的数据已保存至: {output_path}")
        print(f"合并后的数据保存耗时: {end_time - start_time}秒")
        
        # 保存正常trace数据
        normal_traces_path = os.path.join(os.path.dirname(output_path), 'train_iforest_normal_traces.pkl')
        start_time = time.time()
        with open(normal_traces_path, 'wb') as f:
            pickle.dump(normal_traces, f)
        end_time = time.time()
        print(f"用于训练iforest的正常trace数据已保存至: {normal_traces_path}")
        print(f"用于训练iforest的正常trace数据保存耗时: {end_time - start_time}秒")
    
    return merged_df, normal_traces


def load_filtered_trace(df_input_timestamp: pd.DataFrame, index: int) -> tuple[str, dict, str]:
    """
    加载并过滤trace异常数据，返回前20个异常组合的统计CSV格式字符串、三项唯一值字典和status组合统计
    
    参数:
        df_input_timestamp: 包含故障起止时间戳的DataFrame
        index: 要查询的行索引
        
    返回:
        tuple: (filtered_traces_csv, trace_unique_dict, status_combinations_csv)
                - filtered_traces_csv: 前20个异常组合统计数据的CSV格式字符串
                - trace_unique_dict: {'pod_name': [...], 'service_name': [...], 'node_name': [...]} 三项唯一值字典
                - status_combinations_csv: 故障期间status组合统计的CSV格式字符串（前20个）
                如果没有异常或出错则返回空字符串和空字典
    """
    # ========== 第一部分：异常检测器的预处理和训练（训练完成后可以注释掉以下几行） ==========
    trace_detectors = _load_or_train_anomaly_detection_model()
    if trace_detectors is None:
        print("无法获取异常检测模型")
        return "", {}, ""
    
    # 加载正常数据统计信息
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    normal_stats_file = os.path.join(project_root, 'models', 'trace_detectors_normal_stats.pkl')
    normal_stats = {}
    try:
        if os.path.exists(normal_stats_file):
            with open(normal_stats_file, 'rb') as f:
                normal_stats = pickle.load(f)
            print(f"成功加载正常数据统计信息，包含 {len(normal_stats)} 个组合")
    except Exception as e:
        print(f"加载正常数据统计信息失败: {e}")
    
    # ========== 如果已有训练好的模型，可以取消注释以下代码直接加载 ==========
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # detector_file = os.path.join(project_root, 'models', 'trace_detectors.pkl')
    # try:
    #     with open(detector_file, 'rb') as f:
    #         trace_detectors = pickle.load(f)
    #     print(f"成功加载异常检测模型，包含 {len(trace_detectors)} 个检测器")
    # except Exception as e:
    #     print(f"加载异常检测模型失败: {e}")
    #     return ""
    
    # ========== 第二部分：单独的异常检测操作 ==========
    matching_files, start_time, end_time = _get_period_info(df_input_timestamp, index)
    print(matching_files, start_time, end_time)
    
    try:
        if not matching_files:
            print("未找到匹配的trace文件")
            return "", {}, ""
            
        # 读取trace数据
        df_trace = pd.read_parquet(matching_files[0])
        print("原始trace行数：", len(df_trace))
        
        # 过滤时间范围内的数据
        df_filtered_traces = _filter_traces_by_timerange(matching_files, start_time, end_time, df_trace)
        
        if df_filtered_traces is None or len(df_filtered_traces) == 0:
            print("时间范围内没有trace数据")
            return "", {}, ""
            
        print("过滤时间范围后的trace行数：", len(df_filtered_traces))
        
        # 预处理trace数据
        print("预处理trace数据...")
        start_preprocess_time = time.time()
        
        # 提取pod_name, service_name, node_name
        df_filtered_traces['pod_name'] = df_filtered_traces['process'].apply(_extract_pod_name)
        df_filtered_traces['service_name'] = df_filtered_traces['process'].apply(_extract_service_name)
        df_filtered_traces['node_name'] = df_filtered_traces['process'].apply(_extract_node_name)
        
        # 提取父spanID
        df_filtered_traces['parent_spanID'] = df_filtered_traces['references'].apply(_extract_parent_spanid)
        
        # 创建spanID到pod_name的映射
        span_to_pod = dict(zip(df_filtered_traces['spanID'].tolist(), df_filtered_traces['pod_name'].tolist()))
        
        # 提取父spanID对应的pod_name
        df_filtered_traces['parent_pod'] = df_filtered_traces['parent_spanID'].map(lambda x: span_to_pod.get(x))
        
        # 重命名pod_name为child_pod
        df_filtered_traces = df_filtered_traces.rename(columns={'pod_name': 'child_pod'})
        
        # 按时间戳排序
        df_filtered_traces = df_filtered_traces.sort_values(by='timestamp_ns')
        
        end_preprocess_time = time.time()
        print(f"预处理trace数据耗时: {end_preprocess_time - start_preprocess_time:.2f}秒")
        
        # 分析故障期间的status组合（在预处理完成后）
        status_combinations_csv = _analyze_status_combinations_in_fault_period(df_filtered_traces)
        
        # 检测异常
        anomaly_events = _detect_anomalies(df_filtered_traces, trace_detectors)
        
        print(f"检测到 {len(anomaly_events)} 个异常事件")
        
        # 将异常事件转换为DataFrame格式
        if not anomaly_events:
            return "", {}, status_combinations_csv
        
        anomaly_data = []
        for event in anomaly_events:
            timestamp, parent_pod, child_pod, operation_name, anomaly_type, duration, service_name, node_name = event
            # 转换为北京时间 (UTC+8)
            beijing_time = pd.to_datetime(timestamp, unit='ns') + pd.Timedelta(hours=BEIJING_TIMEZONE_OFFSET)
            anomaly_data.append({
                'timestamp': timestamp,
                'timestamp_readable': beijing_time,
                'parent_pod': parent_pod,
                'child_pod': child_pod,
                'operation_name': operation_name,
                'anomaly_type': anomaly_type,
                'duration': duration,
                'service_name': service_name,
                'node_name': node_name
            })
        
        df_anomalies = pd.DataFrame(anomaly_data)
        
        # ========== 第三部分：统计前10个异常组合 ==========
        # 按时间排序
        df_anomalies = df_anomalies.sort_values('timestamp_readable')
        
        # 创建组合列
        df_anomalies['combination'] = (df_anomalies['parent_pod'].astype(str) + '_' + 
                                        df_anomalies['child_pod'].astype(str) + '_' + 
                                        df_anomalies['operation_name'].astype(str))
        
        # duration信息已经在异常检测时直接提取并包含在异常数据中，无需额外匹配
        
        # 按组合分组进行统计
        combination_stats = []
        
        for combination_name, group in df_anomalies.groupby('combination'):
            parent_pod = group['parent_pod'].iloc[0]
            child_pod = group['child_pod'].iloc[0]
            operation_name = group['operation_name'].iloc[0]
            service_name = group['service_name'].iloc[0] if 'service_name' in group.columns else None
            node_name = group['node_name'].iloc[0] if 'node_name' in group.columns else None
            
            # 计算平均duration（如果有的话）
            if 'duration' not in group.columns or len(group['duration'].dropna()) == 0:
                continue  # 跳过没有有效duration数据的组合
            anomaly_avg_duration = group['duration'].mean()
            
            # 获取正常数据的平均时间
            normal_avg_time = 0
            combination_key = f"{parent_pod}_{child_pod}_{operation_name}"
            if combination_key in normal_stats:
                normal_avg_time = normal_stats[combination_key].get('mean', 0)
            
            stats = {
                'node_name': node_name,
                'service_name': service_name,
                'parent_pod': parent_pod,
                'child_pod': child_pod,
                'operation_name': operation_name,
                'normal_avg_duration': normal_avg_time,
                'anomaly_avg_duration': anomaly_avg_duration,
                'anomaly_count': len(group)
            }
            combination_stats.append(stats)
        
        # 转换为DataFrame并按出现次数排序，取前20个
        if not combination_stats:
            return "", {}, status_combinations_csv
            
        stats_df = pd.DataFrame(combination_stats)
        

        # 按出现次数排序，取前20个（在添加文字之前排序）
        top_20_stats = stats_df.sort_values('anomaly_count', ascending=False).head(20)
        
        # 添加文字描述到anomaly_count列
        top_20_stats['anomaly_count'] = top_20_stats['anomaly_count'].apply(lambda x: f"出现次数:{x}")
        
        # 重新排列列顺序
        desired_column_order = ['node_name', 'service_name', 'parent_pod', 'child_pod', 
                               'operation_name', 'normal_avg_duration', 'anomaly_avg_duration', 'anomaly_count']
        # 确保只包含存在的列，并按指定顺序排列
        existing_columns = [col for col in desired_column_order if col in top_20_stats.columns]
        top_20_stats = top_20_stats[existing_columns]
                
        # 从df_filtered_traces中提取三项唯一值：pod_name, service_name, node_name
        trace_unique_dict = {
            'pod_name': [],
            'service_name': [],
            'node_name': []
        }
        
        # 从child_pod和parent_pod中提取pod_name
        pod_names = []
        if 'child_pod' in top_20_stats.columns:
            pod_names.extend(top_20_stats['child_pod'].dropna().unique().tolist())
        if 'parent_pod' in top_20_stats.columns:
            pod_names.extend(top_20_stats['parent_pod'].dropna().unique().tolist())
        trace_unique_dict['pod_name'] = sorted(list(set([str(name) for name in pod_names if pd.notna(name)])))
        
        # 从service_name列提取唯一值
        if 'service_name' in top_20_stats.columns:
            service_names = top_20_stats['service_name'].dropna().unique().tolist()
            trace_unique_dict['service_name'] = sorted(list(set([str(name) for name in service_names if pd.notna(name)])))
        
        # 从node_name列提取唯一值
        if 'node_name' in top_20_stats.columns:
            node_names = top_20_stats['node_name'].dropna().unique().tolist()
            trace_unique_dict['node_name'] = sorted(list(set([str(name) for name in node_names if pd.notna(name)])))
        
        # 返回CSV格式字符串、三项唯一值字典和status组合统计
        filtered_traces_csv = top_20_stats.to_csv(index=False)
        return filtered_traces_csv, trace_unique_dict, status_combinations_csv
        
    except Exception as e:
        print(f"检测trace异常失败: {e}")
        return "", {}, ""


if __name__ == "__main__":
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 使用绝对路径构建input_timestamp.csv的路径
    input_path = os.path.join(project_root, 'input', 'input_timestamp.csv')
    df_input_timestamp = pd.read_csv(input_path)
    
    total_count = len(df_input_timestamp)
    no_match_count = 0
    zero_anomalies_count = 0
    non_zero_anomalies_count = 0
    non_zero_counts = []  # 新增列表存储非零异常数量
    all_anomaly_csv_data = []  # 存储所有异常CSV数据用于后续分析
    
    for index, row in df_input_timestamp.iterrows():
        try:
            print(">>>>>>>>>>>>>>>>>>>")
            print(f"index: {index}")
            
            filtered_traces_csv, trace_unique_dict, status_combinations_csv = load_filtered_trace(df_input_timestamp, index)
            print("filtered_traces_csv:\n", filtered_traces_csv)
            print("trace_unique_dict:\n", trace_unique_dict)
            print("status_combinations_csv:\n", status_combinations_csv)
            
            if not filtered_traces_csv:
                print("未找到匹配的trace或无异常")
                if filtered_traces_csv == "":
                    zero_anomalies_count += 1
                else:
                    no_match_count += 1
                print("<<<<<<<<<<<<<<<<<<")
                continue
                
            # 计算CSV行数（减去header行）
            csv_lines = filtered_traces_csv.strip().split('\n')
            anomaly_count = len(csv_lines) - 1 if len(csv_lines) > 1 else 0
            
            print(f"过滤后的异常trace行数: {anomaly_count}")
            print("<<<<<<<<<<<<<<<<<<")
            
            if anomaly_count == 0:
                zero_anomalies_count += 1
            else:
                non_zero_anomalies_count += 1
                non_zero_counts.append(anomaly_count)
                # 保存CSV数据用于后续分析
                all_anomaly_csv_data.append(filtered_traces_csv)
                
        except Exception as e:
            print(f"处理索引 {index} 时出错: {e}")
            no_match_count += 1
            print("<<<<<<<<<<<<<<<<<<")
        # if index == 1:
        #     break

    print()
    print(f"统计结果:")
    print(f"总数据量: {total_count}")
    print(f"无匹配trace文件的数量: {no_match_count}")
    print(f"过滤后异常数量为0的数量（没有检测到异常的trace）: {zero_anomalies_count}") 
    print(f"过滤后异常数量不为0的数量: {non_zero_anomalies_count}")
    print(f"非零异常数量统计列表: {non_zero_counts}")
    