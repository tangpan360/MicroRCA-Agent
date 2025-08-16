import os
import json
import glob
from datetime import datetime
from typing import Dict, List


def get_record_file_path(uuid: str) -> str:
    """
    获取记录文件路径，每个uuid对应一个独立的txt文件
    
    参数:
        uuid: 样本UUID
        
    返回:
        记录文件的完整路径
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    records_dir = os.path.join(project_root, 'output', 'llm_records')
    os.makedirs(records_dir, exist_ok=True)
    
    return os.path.join(records_dir, f'llm_record_{uuid}.txt')


def record_llm_call(uuid: str, call_type: str, prompt: str, response: str) -> None:
    """
    记录单次大模型调用的输入和输出
    
    参数:
        uuid: 样本UUID
        call_type: 调用类型（如"第1次调用-Service级别分析"）
        prompt: 输入的prompt
        response: 大模型的响应
    """
    try:
        record_file = get_record_file_path(uuid)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建记录条目
        record_entry = {
            "timestamp": timestamp,
            "call_type": call_type,
            "prompt": prompt,
            "response": response
        }
        
        # 追加写入文件
        with open(record_file, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"调用类型: {call_type}\n")
            f.write("=" * 80 + "\n")
            f.write("\n【输入Prompt】:\n")
            f.write(prompt)
            f.write("\n\n【输出Response】:\n")
            f.write(response)
            f.write("\n\n")
            
        print(f"已记录大模型调用: {uuid} - {call_type}")
        
    except Exception as e:
        print(f"记录大模型调用失败 {uuid}: {e}")


def init_record_file() -> None:
    """
    初始化记录目录，清理旧的记录文件
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        records_dir = os.path.join(project_root, 'output', 'llm_records')
        
        # 创建目录
        os.makedirs(records_dir, exist_ok=True)
        
        # 清理旧的记录文件
        old_files = glob.glob(os.path.join(records_dir, 'llm_record_*.txt'))
        for old_file in old_files:
            try:
                os.remove(old_file)
            except:
                pass
                
        print(f"初始化大模型记录目录: {records_dir}")
        
    except Exception as e:
        print(f"初始化记录目录失败: {e}")


def merge_all_records() -> str:
    """
    汇总所有记录文件到一个完整的文件
    
    返回:
        汇总文件的路径
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        records_dir = os.path.join(project_root, 'output', 'llm_records')
        
        # 获取所有记录文件
        record_files = glob.glob(os.path.join(records_dir, 'llm_record_*.txt'))
        
        if not record_files:
            print("未找到任何记录文件")
            return ""
        
        # 创建汇总文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = os.path.join(records_dir, f'merged_llm_records_{timestamp}.txt')
        
        with open(merged_file, 'w', encoding='utf-8') as merged_f:
            merged_f.write("=" * 100 + "\n")
            merged_f.write("大模型调用记录汇总文件\n")
            merged_f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            merged_f.write(f"包含文件数量: {len(record_files)}\n")
            merged_f.write("=" * 100 + "\n\n")
            
            # 按UUID排序文件
            record_files.sort()
            
            for record_file in record_files:
                # 从文件名中提取UUID
                filename = os.path.basename(record_file)
                uuid = filename.replace('llm_record_', '').replace('.txt', '')
                
                merged_f.write("#" * 100 + "\n")
                merged_f.write(f"UUID: {uuid}\n")
                merged_f.write("#" * 100 + "\n\n")
                
                # 复制文件内容
                try:
                    with open(record_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        merged_f.write(content)
                        merged_f.write("\n\n")
                except Exception as e:
                    merged_f.write(f"读取文件 {record_file} 失败: {e}\n\n")
        
        print(f"成功汇总 {len(record_files)} 个记录文件到: {merged_file}")
        return merged_file
        
    except Exception as e:
        print(f"汇总记录文件失败: {e}")
        return ""


def get_record_statistics() -> Dict[str, int]:
    """
    获取记录统计信息
    
    返回:
        包含统计信息的字典
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        records_dir = os.path.join(project_root, 'output', 'llm_records')
        
        record_files = glob.glob(os.path.join(records_dir, 'llm_record_*.txt'))
        
        stats = {
            'total_uuids': len(record_files),
            'total_calls': 0
        }
        
        for record_file in record_files:
            try:
                with open(record_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 统计调用次数（通过"调用类型:"的出现次数）
                    call_count = content.count('调用类型:')
                    stats['total_calls'] += call_count
            except:
                pass
        
        return stats
        
    except Exception as e:
        print(f"获取统计信息失败: {e}")
        return {'total_uuids': 0, 'total_calls': 0} 