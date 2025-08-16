import pandas as pd
import os
import sys
from typing import Optional

# 添加项目根目录到系统路径，确保可以导入utils.io_util
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import utils.io_util as io


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
    search_pattern = os.path.join(project_root, 'data', 'processed', '*', 'log-parquet', f'*{start_time_hour}*')
    matching_files = glob.glob(search_pattern, recursive=True)
    
    # print(f"时间点: {start_time_hour}")
    # print("匹配的文件路径:")
    
    # if not matching_files:
    #     print("- 未匹配到任何文件")
    # else:
    #     for file in matching_files:
    #         print(f"- {file}")
            
    # print(f"start_time: {start_time}")
    # print(f"end_time: {end_time}\n")
    
    return matching_files, start_time, end_time


def _filter_logs_by_timerange(matching_files: list[str], start_time: int, end_time: int, df_log: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    根据时间范围过滤日志数据
    
    参数:
        matching_files: 匹配的文件路径列表
        start_time: 开始时间戳
        end_time: 结束时间戳
        df_log: 包含日志数据的DataFrame，如果为None则会尝试读取匹配的文件
        
    返回:
        DataFrame: 过滤后的日志数据，只包含时间范围内的行；如果没有匹配文件则返回None
    """
    import pandas as pd
    
    # 检查是否找到匹配的文件
    if not matching_files:
        print("未找到匹配的日志文件")
        return None
    
    # 如果没有提供df_log，则读取匹配的文件
    if df_log is None:
        df_log = pd.DataFrame()
        for file in matching_files:
            temp_df = pd.read_parquet(file)
            df_log = pd.concat([df_log, temp_df])
    
    # 确保df_log包含timestamp_ns列
    if 'timestamp_ns' not in df_log.columns:
        print("日志数据中缺少timestamp_ns列")
        return None
    
    # 使用时间戳过滤数据
    filtered_df = df_log[(df_log['timestamp_ns'] >= start_time) & (df_log['timestamp_ns'] <= end_time)]
    
    # # 打印结果统计
    # print(f"原始数据行数: {len(df_log)}")
    # print(f"过滤后数据行数: {len(filtered_df)}")
    # print(f"过滤时间范围: {start_time} 到 {end_time}")
    
    return filtered_df


def _filter_logs_by_error(df: Optional[pd.DataFrame], column: str = 'message') -> Optional[pd.DataFrame]:
    """
    过滤包含'error'（不区分大小写）的日志数据
    
    参数:
        df: 输入的DataFrame
        column: 要检查的列名，默认为'message'
        
    返回:
        DataFrame: 包含error的日志数据；如果输入为None或列不存在则返回None
    """
    if df is None:
        print("输入数据为空")
        return None
    
    if column not in df.columns:
        print(f"列{column}不存在")
        return None
    
    error_logs = df[df[column].str.contains('error', case=False, na=False)]
    print(f"找到{len(error_logs)}条包含error的日志")
    return error_logs


def _filter_out_injected_errors(df: Optional[pd.DataFrame], column: str = 'message') -> Optional[pd.DataFrame]:
    """
    过滤掉包含'java'的日志数据
    
    参数:
        df: 输入的DataFrame
        column: 要检查的列名，默认为'message'
        
    返回:
        DataFrame: 过滤后的日志数据；如果输入为None或列不存在则返回None
    """
    if df is None:
        print("输入数据为空")
        return None
    
    if column not in df.columns:
        print(f"列{column}不存在")
        return None
    
    # 记录过滤前的行数
    before_count = len(df)
    
    # 过滤掉包含注入错误的行
    filtered_df = df[~df[column].str.contains('java', na=False)]
    
    # 计算过滤掉的行数
    filtered_out_count = before_count - len(filtered_df)
    
    print(f"过滤掉了{filtered_out_count}条包含'java'的日志 ({filtered_out_count / before_count:.2%})")
    
    return filtered_df

    
def _filter_logs_by_columns(filtered_df: Optional[pd.DataFrame], columns: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    """
    从已过滤的日志数据中进一步筛选指定的列
    
    参数:
        filtered_df: 已经过时间范围过滤的DataFrame
        columns: 需要保留的列名列表，如果为None则返回所有列
        
    返回:
        DataFrame: 只包含指定列的数据；如果输入为None则返回None
    """
    if filtered_df is None:
        print("输入数据为空")
        return None
    
    if columns is None:
        print("未指定列，返回所有列")
        return filtered_df
    
    # 检查请求的列是否存在于DataFrame中
    missing_cols = [col for col in columns if col not in filtered_df.columns]
    if missing_cols:
        print(f"警告: 以下列不存在: {missing_cols}")
    
    # 只保留存在的列
    valid_cols = [col for col in columns if col in filtered_df.columns]
    if not valid_cols:
        print("没有有效的列名")
        return None
    # 保证返回DataFrame而不是Series
    return filtered_df.loc[:, valid_cols]


def _sample_logs_by_pod(df: Optional[pd.DataFrame], group_col: str = 'k8_pod', max_samples: int = 3, random_state: int = 42) -> Optional[pd.DataFrame]:
    """
    按指定列分组并随机采样每个组的日志
    
    参数:
        df: 输入的DataFrame
        group_col: 用于分组的列名，默认为'k8_pod'
        max_samples: 每组最大采样数量，默认为3
        random_state: 随机种子，默认为42
        
    返回:
        DataFrame: 采样后的数据
    """
    if df is None:
        print("输入数据为空")
        return None
    
    # 按group_col分组并采样
    sampled_df = df.groupby(group_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), max_samples), random_state=random_state)
    )
    
    print(f"采样后数据行数: {len(sampled_df)}")
    return sampled_df


def _extract_log_templates(df: Optional[pd.DataFrame], message_col: str = 'message') -> Optional[pd.DataFrame]:
    """
    从日志消息中提取模板并添加模板ID和模板内容列
    
    参数:
        df: 包含日志消息的DataFrame
        message_col: 包含日志消息的列名，默认为'message'
        
    返回:
        DataFrame: 添加了template_id和template列的DataFrame
    """
    if df is None or len(df) == 0:
        print("输入数据为空，无法提取模板")
        return df
    
    if message_col not in df.columns:
        print(f"列{message_col}不存在，无法提取模板")
        return df
    
    try:
        # 加载预训练的Drain模型
        drain_model_path = os.path.join(project_root, 'utils', 'drain', 'error_log-drain.pkl')
        if not os.path.exists(drain_model_path):
            print(f"Drain模型文件不存在: {drain_model_path}")
            return df
            
        miner = io.load(drain_model_path)
        print(f"成功加载Drain模型")
        
        # 提取模板ID和模板内容
        templates = []
        
        for log in df[message_col]:
            # 使用Drain匹配日志模板
            cluster = miner.match(log)
            if cluster is None:
                templates.append(None)
            else:
                templates.append(cluster.get_template())
        
        # 将模板信息添加到DataFrame
        df['template'] = templates
        
        print(f"成功为{len(df)}条日志提取模板")
        return df
        
    except Exception as e:
        print(f"提取日志模板失败: {e}")
        return df


def _deduplicate_pod_template_combinations(df: Optional[pd.DataFrame], pod_col: str = 'k8_pod', template_col: str = 'template') -> Optional[pd.DataFrame]:
    """
    对DataFrame按pod和模板组合进行去重，只保留每种组合的第一次出现，并添加计数列

    参数:
        df: 包含pod和模板列的DataFrame
        pod_col: pod列的名称，默认为'k8_pod'
        template_col: 模板列的名称，默认为'template'

    返回:
        DataFrame: 去重后的DataFrame，只保留每种pod和模板组合的第一次出现，并添加occurrence_count列
    """
    if df is None or len(df) == 0:
        print("输入数据为空，无法进行去重")
        return df

    if pod_col not in df.columns:
        print(f"列{pod_col}不存在，无法进行去重")
        return df

    if template_col not in df.columns:
        print(f"列{template_col}不存在，无法进行去重")
        return df

    try:
        # 记录原始行数
        original_count = len(df)

        # 对None值进行特殊处理，将None转换为字符串'None'以便分组
        df_copy = df.copy()
        df_copy[template_col] = df_copy[template_col].fillna('None')

        # 计算每种pod和模板组合出现的次数
        pod_template_counts = df_copy.groupby([pod_col, template_col]).size().reset_index().rename(columns={0: 'occurrence_count'})
        # 修改 occurrence_count 列格式
        pod_template_counts['occurrence_count'] = pod_template_counts['occurrence_count'].apply(
            lambda x: f"出现次数:{x}"
        )
        # 按pod和模板组合进行去重，保留第一次出现的行
        df_deduplicated = df_copy.drop_duplicates(subset=[pod_col, template_col], keep='first')

        # 将计数信息合并到去重后的DataFrame
        df_deduplicated = pd.merge(df_deduplicated, pod_template_counts, on=[pod_col, template_col], how='left')

        # 将'None'值转回None
        df_deduplicated.loc[df_deduplicated[template_col] == 'None', template_col] = None

        # 打印去重结果
        deduplicated_count = len(df_deduplicated)
        print(f"去重前行数: {original_count}")
        print(f"去重后行数: {deduplicated_count}")
        print(f"减少了 {original_count - deduplicated_count} 行 ({(original_count - deduplicated_count) / original_count:.2%})")

        return df_deduplicated

    except Exception as e:
        print(f"按pod和模板组合去重失败: {e}")
        return df

def _extract_service_name(pod_name: str) -> str:
    """
    从pod_name中提取service_name（如frontend-1 -> frontend）

    参数:
        pod_name: pod名称字符串，例如'frontend-1'
    返回:
        str: 提取的service_name（如'frontend'），如果无法提取则返回原始pod_name
    """
    if not isinstance(pod_name, str):
        return None
    # 取第一个'-'前的部分
    import re
    match = re.match(r'([a-zA-Z0-9]+)', pod_name)
    if match:
        return match.group(1)
    return pod_name

def load_filtered_log(df_input_timestamp: pd.DataFrame, index: int) -> Optional[tuple[str, dict]]:
    """
    加载并过滤日志数据，返回CSV格式字符串和唯一pod/service/node列表

    参数:
        df_input_timestamp: 包含故障起止时间戳的DataFrame
        index: 要查询的行索引

    返回:
        tuple: (filtered_logs_csv, log_unique_dict)
               - filtered_logs_csv: CSV格式的过滤后日志字符串
               - log_unique_dict: {'pod_name': [...], 'service_name': [...], 'node_name': [...]} 三项唯一值列表
               如果没有匹配文件或处理过程中出错则返回None
    """
    matching_files, start_time, end_time = _get_period_info(df_input_timestamp, index)
    print(matching_files, start_time, end_time)

    try:
        if not matching_files:
            print("未找到匹配的日志文件")
            return None

        df_log = pd.read_parquet(matching_files[0])
        print("原始日志文件的数据量：", len(df_log))
        df_filtered_logs = _filter_logs_by_timerange(matching_files, start_time, end_time, df_log=df_log)
        if df_filtered_logs is None:
            print("时间过滤后日志文件为空")
            return None
        print("时间过滤后日志文件的数据量：", len(df_filtered_logs))
        df_filtered_logs = _filter_logs_by_error(df_filtered_logs, column='message')
        if df_filtered_logs is None:
            print("错误过滤后日志文件为空")
            return None
        print("错误过滤后日志文件的数据量：", len(df_filtered_logs))
        df_filtered_logs = _filter_logs_by_columns(filtered_df=df_filtered_logs, columns=['time_beijing', 'k8_pod', 'message', 'k8_node_name'])
        if df_filtered_logs is None:
            print("列过滤后日志文件为空")
            return None
        print("列过滤后日志文件的数据量：", len(df_filtered_logs))
        df_filtered_logs = _extract_log_templates(df_filtered_logs, message_col='message')
        if df_filtered_logs is None:
            print("模板提取后日志文件为空")
            return None
        print("模板提取后日志文件的数据量：", len(df_filtered_logs))
        df_filtered_logs = _deduplicate_pod_template_combinations(df_filtered_logs, pod_col='k8_pod', template_col='template')
        if df_filtered_logs is None:
            print("模板去重后日志文件为空")
            return None
        print("模板去重后日志文件的数据量：", len(df_filtered_logs))

        # 增加service_name列
        df_filtered_logs['service_name'] = df_filtered_logs['k8_pod'].apply(_extract_service_name)
        # pod_name和node_name直接重命名
        df_filtered_logs = df_filtered_logs.rename(columns={'k8_pod': 'pod_name', 'k8_node_name': 'node_name'})

        # 保留occurrence_count列，但不保留template列和time_beijing列，node_name和service_name在pod_name前面
        df_filtered_logs = df_filtered_logs[['node_name', 'service_name', 'pod_name', 'message', 'occurrence_count']]

        # 按出现次数降序排序，使高频错误排在前面
        df_filtered_logs = df_filtered_logs.sort_values(by='occurrence_count', ascending=False)

        # 过滤掉包含java.lang.Error Injected error的日志
        df_filtered_logs = _filter_out_injected_errors(df_filtered_logs, column='message')

        # 检查过滤后是否还有数据
        if df_filtered_logs is None or len(df_filtered_logs) == 0:
            print("过滤后没有剩余日志数据")
            return None

        # 提取唯一的pod_name、service_name、node_name列表
        log_unique_dict = {
            'pod_name': df_filtered_logs['pod_name'].unique().tolist() if df_filtered_logs is not None else [],
            'service_name': df_filtered_logs['service_name'].unique().tolist() if df_filtered_logs is not None else [],
            'node_name': df_filtered_logs['node_name'].unique().tolist() if df_filtered_logs is not None else []
        }
        filtered_logs_csv = df_filtered_logs.to_csv(index=False)

        return filtered_logs_csv, log_unique_dict

    except Exception as e:
        print(f"加载日志数据失败: {e}")
        return None  # 明确返回None而不是使用continue


if __name__ == "__main__":
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 使用绝对路径构建input_timestamp.csv的路径
    input_path = os.path.join(project_root, 'input', 'input_timestamp.csv')
    df_input_timestamp = pd.read_csv(input_path)
    
    total_count = len(df_input_timestamp)
    no_match_count = 0
    zero_logs_count = 0
    non_zero_logs_count = 0
    non_zero_counts = []  # 新增列表存储非零日志数量
    
    for index, row in df_input_timestamp.iterrows():
        try:
            print(">>>>>>>>>>>>>>>>>>>")
            print(f"index: {index}")
            result = load_filtered_log(df_input_timestamp, index)
            
            if result is None:
                print("未找到匹配的日志")
                no_match_count += 1
                print("<<<<<<<<<<<<<<<<<<")
                continue
            
            filtered_logs_csv, log_unique_pods = result
            print("filtered_logs_csv:\n", filtered_logs_csv)
            print("log_unique_pods:\n", log_unique_pods)

            # 通过计算CSV中的行数来获取日志数量（减去header行）
            log_count = len(filtered_logs_csv.split('\n')) - 1 if filtered_logs_csv.strip() else 0
            print("过滤后的日志行数: ", log_count)
            print("<<<<<<<<<<<<<<<<<<")
            
            if log_count == 0:
                zero_logs_count += 1
            else:
                non_zero_logs_count += 1
                non_zero_counts.append(log_count)  # 记录非零日志数量
        except Exception as e:
            print(f"处理索引 {index} 时出错: {e}")
            no_match_count += 1
            print("<<<<<<<<<<<<<<<<<<")

    print()
    print(f"统计结果:")
    print(f"总数据量: {total_count}")
    print(f"无匹配日志文件的数量: {no_match_count}")
    print(f"过滤后日志数量为0的数量（没有找到 'message' 列包含 'error' 的日志）: {zero_logs_count}") 
    print(f"过滤后日志数量不为0的数量: {non_zero_logs_count}")
    print(f"非零日志数量统计列表: {non_zero_counts}")  # 打印非零日志数量列表