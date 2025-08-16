"""
文件处理工具模块
"""
import json
import re
from collections import OrderedDict
from typing import Any, Optional

def load_result_jsonl(file_path: str = './submission/result.jsonl') -> dict[str, dict[str, Any]]:
    """
    读取现有的result.jsonl文件
    
    参数:
        file_path: 结果文件路径
        
    返回:
        包含结果的字典，以uuid为键
    """
    results = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                results[data['uuid']] = data
    except Exception as e:
        print(f"Error loading result.jsonl: {e}")
    return results

def update_single_result(uuid: str, data: dict[str, Any], file_path: str = './submission/result.jsonl') -> None:
    """
    更新单个结果到result.jsonl文件，保持指定的字段顺序
    
    参数:
        uuid: 要更新的记录的uuid
        data: 新的数据
        file_path: 结果文件路径
    """
    # 先读取所有现有结果
    results = load_result_jsonl(file_path)

    # 更新或添加当前UUID的结果
    results[uuid] = data

    # 重写整个文件，确保字段顺序
    with open(file_path, 'w', encoding='utf-8') as f:
        for u, d in sorted(results.items()):
            # 创建有序字典确保字段顺序
            ordered_data = OrderedDict()
            ordered_data["component"] = d.get("component", "")
            ordered_data["uuid"] = u
            ordered_data["reason"] = d.get("reason", "")
            ordered_data["reasoning_trace"] = d.get("reasoning_trace", [])

            f.write(json.dumps(ordered_data, ensure_ascii=False) + '\n')
    
    print(f"Updated result for {uuid} in {file_path}")

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    使用正则表达式从文本中提取JSON
    
    参数:
        text: 包含JSON的文本
        
    返回:
        解析后的JSON字典，如果解析失败则返回None
    """
    # 尝试找到JSON对象 - 从第一个{开始到最后一个}结束
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            # 尝试解析找到的JSON字符串
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Found JSON-like string but failed to parse: {json_str}")
            return None
    return None
