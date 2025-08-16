"""
指标(metric)文件处理脚本

功能说明：
1. 文件筛选：自动跳过名字中包含"deleted"的文件
2. 目录管理：自动创建处理后的目录结构 data/phaseone/processed/
3. 时间处理：
    - 标准化时间格式（time列转换为datetime类型）
    - 生成纳秒级时间戳（timestamp_ns）
    - 时区转换：UTC时间转换为北京时间（Asia/Shanghai）
    - 保留UTC原始时间（time_utc）和北京时间（time_beijing）双版本
4. 数据排序：按时间戳（timestamp_ns）升序排列
5. 结果保存：以parquet格式输出，且不保存索引列
6. 多进程并行处理：
    - 可配置CPU核心使用比例
    - 自动计算合适的进程数量（最少1个进程）
    - 提供详细的处理统计信息和耗时记录

特殊说明：
- 使用递归搜索(recursive=True)获取所有子目录下的metric parquet文件
- 原始时间列名称为'time'
"""

import pandas as pd
import glob
import os
from multiprocessing import Pool, cpu_count
import time

def process_single_file(file_path):
    """处理单个文件的函数"""
    try:
        # 如果文件名中包含"deleted"，则跳过不进行处理
        if "deleted" in file_path:
            return {"status": "skipped", "file": file_path, "reason": "contains 'deleted'"}
        
        # 构建目标路径 - 将 'raw' 替换为 'processed'
        target_path = file_path.replace('raw', 'processed')
        target_dir = os.path.dirname(target_path)
        
        # 创建目标目录
        os.makedirs(target_dir, exist_ok=True)
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        # 将time转换为datetime类型，然后创建纳秒级时间戳的新列
        if 'time' in df.columns:
            # 在转换datetime时直接指定UTC时区
            df['time_utc'] = pd.to_datetime(df['time'], utc=True)
            
            # 首先基于UTC时间生成正确的时间戳
            df['timestamp_ns'] = df['time_utc'].astype('int64')  # 直接获取纳秒级时间戳（19位数字：1749690631336000000）
            # df['timestamp_us'] = df['time_utc'].astype('int64') // 1_000         # 纳秒转微秒（16位数字：1749690631336000）
            # df['timestamp_ms'] = df['time_utc'].astype('int64') // 1_000_000     # 纳秒转毫秒（13位数字：1749690631336）
            # df['timestamp_s'] = df['time_utc'].astype('int64') // 1_000_000_000  # 纳秒转秒（10位数字：1749690631）
            
            # 使用标准时区转换方法转换为北京时间
            df['time_beijing'] = df['time_utc'].dt.tz_convert('Asia/Shanghai')

            # 按照时间戳升序排序并重置索引
            df = df.sort_values('timestamp_ns', ascending=True).reset_index(drop=True)
        
        # 保存处理后的文件，确保按时间排序后的数据被正确保存
        df.to_parquet(target_path, index=False)
        
        return {"status": "success", "file": file_path, "target": target_path}
        
    except Exception as e:
        return {"status": "error", "file": file_path, "error": str(e)}

def main(cpu_ratio=0.3):
    """
    主函数
    :param cpu_ratio: CPU核心使用比例，默认0.3(30%)
    """
    # 获取脚本文件所在目录，然后构建项目根目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 脚本在scripts目录下，项目根目录是上一级

    # 使用绝对路径构建数据路径 - 利用 ** 递归搜索所有子目录
    base_path = os.path.join(project_root, 'data', 'raw', '2025-06-*', 'metric-parquet', '**', '*.parquet')
    parquet_files = glob.glob(base_path, recursive=True)
    
    if not parquet_files:
        print("未找到任何parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个文件需要处理")
    
    # 计算进程数：根据传入的CPU使用比例计算
    total_cores = cpu_count()
    num_processes = max(1, int(total_cores * cpu_ratio))
    
    print(f"系统总CPU核心数: {total_cores}")
    print(f"使用进程数: {num_processes} (约占{num_processes/total_cores:.1%}的CPU资源)")
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建进程池并并行处理文件
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, parquet_files)
    
    # 记录结束时间
    end_time = time.time()
    
    # 统计处理结果
    success_count = 0
    skipped_count = 0
    error_count = 0
    skipped_files = []
    error_files = []
    
    for result in results:
        if result["status"] == "success":
            success_count += 1
            # print(f"处理完成: {result['file']} -> {result['target']}")
        elif result["status"] == "skipped":
            skipped_count += 1
            skipped_files.append(result["file"])
        elif result["status"] == "error":
            error_count += 1
            error_files.append((result["file"], result["error"]))
            print(f"处理失败: {result['file']} - 错误: {result['error']}")
    
    # 打印统计信息
    print(f"\n处理统计:")
    print(f"- 成功处理: {success_count} 个文件")
    print(f"- 跳过处理: {skipped_count} 个文件")
    print(f"- 处理失败: {error_count} 个文件")
    print(f"- 总耗时: {end_time - start_time:.2f} 秒")
    
    # 打印所有跳过的文件
    if skipped_files:
        print("\n以下文件因包含'deleted'被跳过处理:")
        for skipped in skipped_files:
            print(f"- {skipped}")
    
    # 打印错误文件
    if error_files:
        print("\n以下文件处理失败:")
        for file_path, error in error_files:
            print(f"- {file_path}: {error}")
    
    print("\n所有 metric 文件处理完毕！")

if __name__ == "__main__":
    # 默认使用30% CPU核心
    main(cpu_ratio=0.3)