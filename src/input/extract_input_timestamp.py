import json
import re
import pandas as pd

# 读取input.json文件
with open('input.json', 'r') as f:
    data = json.load(f)

# 创建空的结果列表
results = []

# 更通用的ISO 8601时间格式正则表达式
# 匹配形如 YYYY-MM-DDThh:mm:ssZ 的时间格式
time_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'

for item in data:
    uuid = item["uuid"]
    description = item["Anomaly Description"]
    
    # 查找描述中的所有时间戳
    timestamps = re.findall(time_pattern, description)
    
    if len(timestamps) >= 2:
        start_time_str = timestamps[0]
        end_time_str = timestamps[1]
        results.append({
            "uuid": uuid,
            "start_time_str": start_time_str,
            "end_time_str": end_time_str,
            "Anomaly Description": description
        })
    else:
        print(f"警告: 在UUID {uuid}的描述中没有找到足够的时间戳")

# 转换为DataFrame
df = pd.DataFrame(results)

# 先转为 UTC 时区的时间戳
df['start_time_utc'] = pd.to_datetime(df['start_time_str'], utc=True)
df['end_time_utc'] = pd.to_datetime(df['end_time_str'], utc=True)

# 首先基于UTC时间生成正确的时间戳
df['start_timestamp'] = df['start_time_utc'].astype('int64')
df['end_timestamp'] = df['end_time_utc'].astype('int64')

# 使用标准时区转换方法转换为北京时间
df['start_time_beijing'] = df['start_time_utc'].dt.tz_convert('Asia/Shanghai')
df['end_time_beijing'] = df['end_time_utc'].dt.tz_convert('Asia/Shanghai')

# 格式化输出（字符串）——可选，直接用 datetime 类型也行
df['start_time_beijing'] = df['start_time_beijing'].dt.strftime('%Y-%m-%d_%H-%M-%S')
df['end_time_beijing'] = df['end_time_beijing'].dt.strftime('%Y-%m-%d_%H-%M-%S')

# 事件分组
df['date'] = pd.to_datetime(df['start_time_utc']).dt.tz_convert('Asia/Shanghai').dt.date
df['hour'] = pd.to_datetime(df['start_time_utc']).dt.tz_convert('Asia/Shanghai').dt.strftime('%H-00-00')
df['start_time_hour'] = pd.to_datetime(df['start_time_utc']).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d_%H')

# 持续时间（秒和分钟），依然用UTC来计算，防止跨时区计算误差
df['duration_seconds'] = (df['end_time_utc'] - df['start_time_utc']).dt.total_seconds()
df['duration_minutes'] = df['duration_seconds'] / 60

# 重新排列列顺序
df = df[['uuid', 'start_time_utc', 'end_time_utc',
            'start_time_beijing', 'end_time_beijing',
            'start_timestamp', 'end_timestamp',
            'date', 'hour', 'start_time_hour',
            'duration_seconds', 'duration_minutes',
            'Anomaly Description']]

# 显示结果
print(f"已成功提取 {len(df)} 组数据的时间周期")

# 将 df 保存为 csv 文件
df.to_csv('input_timestamp.csv', index=False)