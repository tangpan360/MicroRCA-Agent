#!/usr/bin/env python3
"""
填充空结果脚本

功能：将result.jsonl中的空结果随机填充已有内容
"""
import json
import os
import random
from collections import OrderedDict
from utils.file_utils import load_result_jsonl, update_single_result

def fill_empty_results() -> None:
    """
    将空结果随机填充已有内容
    
    返回值:
        None: 无返回值，直接修改文件
    """
    # 文件路径
    file_path = './submission/result.jsonl'
    
    # 读取所有结果
    results = load_result_jsonl(file_path)
    
    # 分离有内容和无内容的记录
    filled_records = []
    empty_records = []
    
    for uuid, record in results.items():
        # 检查是否有内容
        if record["component"] and record["reason"]:
            # 保存有内容的记录
            filled_records.append({
                "component": record["component"],
                "reason": record["reason"],
                "reasoning_trace": record["reasoning_trace"],
                "uuid": uuid  # 保存原始UUID以便调试
            })
        else:
            empty_records.append(uuid)
    
    print(f"找到 {len(filled_records)} 条有内容的记录")
    print(f"找到 {len(empty_records)} 条空记录")
    
    if not filled_records:
        print("没有找到有内容的记录，无法填充")
        return
    
    # 备份原文件
    backup_path = file_path + '.backup'
    if not os.path.exists(backup_path):
        with open(file_path, 'r', encoding='utf-8') as f_in:
            with open(backup_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        print(f"原文件已备份为 {backup_path}")
    
    # 填充空记录
    updated_count = 0
    for empty_uuid in empty_records:
        # 随机选择一个有内容的记录
        filled_record = random.choice(filled_records)
        
        # 准备结果数据，按照指定顺序
        result_data = OrderedDict()
        result_data["component"] = filled_record["component"]
        result_data["uuid"] = empty_uuid
        result_data["reason"] = filled_record["reason"]
        result_data["reasoning_trace"] = filled_record["reasoning_trace"]
        
        # 更新结果
        update_single_result(empty_uuid, result_data, file_path)
        updated_count += 1
    
    print(f"处理完成，已将 {updated_count} 条空记录随机填充")

if __name__ == "__main__":
    fill_empty_results()