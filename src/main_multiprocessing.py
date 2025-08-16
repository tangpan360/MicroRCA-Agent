import os
import re
import json
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from collections import OrderedDict
from agent.prompts import get_log_analysis_prompt, get_multimodal_analysis_prompt
from utils.log_utils import load_filtered_log
from utils.trace_utils import load_filtered_trace
from utils.file_utils import load_result_jsonl, update_single_result, extract_json_from_text
from agent.agents import create_log_agent
from utils.fill_empty_results import fill_empty_results


# 单次处理尝试的函数
def process_single_attempt(index, row, df_input_timestamp, uuid, log_agent, attempt_num=1):
    """执行单次处理尝试"""
    print(f"Processing index: {index}, UUID: {uuid}, attempt: {attempt_num}")
    
    # 加载三种模态的数据
    log_data = None
    trace_data = None
    metric_data = None  # 暂时为None，等待metric_utils实现
    
    # 尝试加载日志数据
    try:
        log_result = load_filtered_log(df_input_timestamp, index)
        if log_result is not None:
            log_data = log_result
            print(f"成功加载日志数据")
    except Exception as e:
        print(f"加载日志数据失败: {e}")
    
    # 尝试加载trace数据
    try:
        trace_result = load_filtered_trace(df_input_timestamp, index)
        if trace_result is not None and trace_result[0]:  # 检查CSV不为空
            trace_data = trace_result
            print(f"成功加载trace数据")
    except Exception as e:
        print(f"加载trace数据失败: {e}")
    
    print(f"trace数据处理完毕，准备生成多模态prompt，uuid={uuid}, attempt={attempt_num}")
    # 尝试加载metric数据
    try:
        from utils.metric_utils import analyze_fault_comprehensive
        metric_result = analyze_fault_comprehensive(df_input_timestamp, index, uuid)
        if metric_result:
            metric_data = metric_result  # 注意：这里是字符串类型，不是tuple
            print(f"成功加载metric数据")
    except Exception as e:
        print(f"加载metric数据失败: {e}")
    
    # 检查是否至少有一种模态的数据
    if log_data is None and trace_data is None and metric_data is None:
        print(f"uuid: {uuid} 没有找到任何有效的监控数据")
        return None
    
    # 使用多模态分析prompt
    content = get_multimodal_analysis_prompt(
        log_data=log_data,
        trace_data=trace_data,
        metric_data=metric_data
    )
    print(f"多模态prompt已生成，准备调用LLM agent，uuid={uuid}, attempt={attempt_num}")
    
    messages = [
        {
            "content": content,
            "role": "user"
        }
    ]

    print(f"调用LLM agent前: uuid={uuid}, attempt={attempt_num}")
    reply = log_agent.generate_reply(
        messages=messages
    )
    print(f"调用LLM agent后: uuid={uuid}, attempt={attempt_num}")
    
    # 处理不同模型的返回格式
    response_content = reply['content'] if isinstance(reply, dict) and 'content' in reply else reply
    print(response_content)
    
    # 记录第3次大模型调用
    try:
        from utils.llm_record_utils import record_llm_call
        record_llm_call(uuid, "第3次调用-多模态综合分析", content, response_content)
    except Exception as e:
        print(f"记录大模型调用失败: {e}")
    
    # 使用正则表达式提取JSON内容
    response_data = extract_json_from_text(response_content)
    
    if response_data:
        # 准备结果数据，按照指定顺序
        result_data = OrderedDict()
        result_data["component"] = response_data.get("component", "")
        result_data["uuid"] = uuid
        result_data["reason"] = response_data.get("reason", "")
        result_data["reasoning_trace"] = response_data.get("reasoning_trace", [])
        
        return (uuid, result_data)
    else:
        print(f"Failed to extract valid JSON from reply for {uuid} on attempt {attempt_num}")
        return None

# 处理单个时间段的函数（带重试机制）
def process_input_csv(args):    
    index, row, df_input_timestamp, results = args
    uuid = row['uuid']
    
    # 如果该uuid的结果已经有内容，则跳过
    if uuid in results and results[uuid]['component'] and results[uuid]['reason']:
        print(f"Skipping {uuid}, already processed")
        return None
    
    max_retries = 3
    
    try:
        # 创建新的agent实例（在每个进程中创建）
        log_agent = create_log_agent()
        
        # 尝试处理，最多重试3次
        for attempt in range(1, max_retries + 1):
            result = process_single_attempt(index, row, df_input_timestamp, uuid, log_agent, attempt)
            
            if result is not None:
                print(f"Successfully processed {uuid} on attempt {attempt}")
                return result
            
            if attempt < max_retries:
                print(f"Retrying {uuid}, attempt {attempt + 1}/{max_retries}")
                time.sleep(1)  # 短暂延迟后重试
        
        print(f"Failed to process {uuid} after {max_retries} attempts, skipping")
        return None

    except Exception as e:
        print(f"Error processing {uuid}: {e}")
        return None

# 主函数
def main():    
    # 使用绝对路径构建input_timestamp.csv的路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(project_root, 'input', 'input_timestamp.csv')
    df_input_timestamp = pd.read_csv(input_path)
    print(f"Loaded {len(df_input_timestamp)} time periods")
    
    # 加载现有结果
    results = load_result_jsonl()
    print(f"Loaded {len(results)} existing results")
    
    # 准备要处理的数据
    tasks = [(index, row, df_input_timestamp, results) for index, row in df_input_timestamp.iterrows()]
    
    # 确定进程数量（使用CPU核心数量的50%，至少1个）
    num_processes = max(1, int(cpu_count() * 0.5))
    print(f"Using {num_processes} processes")
    
    # 创建进程池并并行处理
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        results_list = pool.map(process_input_csv, tasks)

    # 保存results_list到本地
    results_list_path = os.path.join(project_root, 'output', 'results_list.json')
    os.makedirs(os.path.dirname(results_list_path), exist_ok=True)
    with open(results_list_path, 'w') as f:
        json.dump(results_list, f)
    print(f"Results list saved to {results_list_path}")
    
    # 更新结果
    updated_count = 0
    for result in results_list:
        if result:
            uuid, data = result
            update_single_result(uuid, data)
            updated_count += 1
    
    # 随机选择已有结果填充空结果
    """因为我只做了log分析，有些input中的时间段内没有log，或者说有些有log文件，
    但log中没有包含'error'，所以需要随机选择已有结果填充空结果，有成绩加成"""
    # fill_empty_results()
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Updated {updated_count} results")
    
    # 汇总所有大模型调用记录
    try:
        from utils.llm_record_utils import merge_all_records, get_record_statistics
        
        # 获取统计信息
        stats = get_record_statistics()
        print(f"\n大模型调用统计:")
        print(f"  总UUID数量: {stats['total_uuids']}")
        print(f"  总调用次数: {stats['total_calls']}")
        
        # 汇总记录文件
        merged_file = merge_all_records()
        if merged_file:
            print(f"大模型调用记录已汇总到: {merged_file}")
    except Exception as e:
        print(f"汇总大模型调用记录失败: {e}")

if __name__ == "__main__":
    main() 