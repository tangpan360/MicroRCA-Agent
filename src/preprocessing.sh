#!/bin/bash

# # 功能说明：设置日志记录功能，将脚本的标准输出和错误输出同时显示在终端并保存到日志文件
# # 使用环境变量__TEE_ACTIVE确保日志记录只初始化一次
# # 注意：在其他脚本中使用时，需要修改以下内容：
# # 1. 将 preprocessing.log 替换为其他日志文件名
# # 2. 确保 __TEE_ACTIVE 变量名不与脚本中其他变量名冲突
# if [ -z "$__TEE_ACTIVE" ]; then
#     export __TEE_ACTIVE=1
#     exec > >(tee preprocessing.log) 2>&1
# fi

# # 下载 submission 文件
# git clone http://www.aiops.cn/gitlab/aiops-live-benchmark/aiopschallenge2025-submission.git

# # 将 aiopschallenge2025-submission 文件夹重新命名为 submission
# mv aiopschallenge2025-submission submission

echo "================================================"
echo "第一阶段：并行下载和准备所有数据"
echo "================================================"

echo ""
echo "================================================"
echo "开始并行下载 phaseone 和 phasetwo 数据..."
echo "================================================"

# 并行下载
echo "同时开始下载两个数据集..."
git clone http://www.aiops.cn/gitlab/aiops-live-benchmark/phaseone.git &
git clone http://www.aiops.cn/gitlab/aiops-live-benchmark/phasetwo.git &

# 等待下载完成
wait

echo "两个数据集下载完成，开始处理..."

# 处理 phaseone
echo ""
echo "================================================"
echo "处理 phaseone 数据..."
echo "================================================"

cd phaseone

# 检查文件完整性
echo "------------------------------------------------"
echo "正在校验 phaseone 文件完整性..."

# 检查 checksums.md5 文件是否存在并处理行结束符
if [ -f "checksums.md5" ]; then
    # 转换行结束符为 Unix 格式（去除 \r 字符）
    sed -i 's/\r$//' checksums.md5
    
    # 进行 MD5 校验
    md5sum -c checksums.md5
    if [ $? -eq 0 ]; then
        echo "所有 phaseone 文件下载完整且未被篡改"
    else
        echo "以下 phaseone 文件存在问题:"
        md5sum -c checksums.md5 | grep -v '成功\|OK' | awk -F: '{print $1}'
    fi
else
    echo "未找到 checksums.md5 文件，跳过完整性校验"
fi

# 解压 phaseone 数据文件
echo ""
echo "------------------------------------------------"
echo "正在解压 phaseone 文件..."

if ls *.tar.gz 1> /dev/null 2>&1; then
    for file in *.tar.gz; do tar -zxf "$file"; done
    echo "phaseone 文件解压完成"
else
    echo "未找到需要解压的 .tar.gz 文件"
fi

# 将解压后的文件夹移动到 data/raw/ 目录下
echo ""
echo "------------------------------------------------"
echo "正在移动解压后的 phaseone 数据文件..."

# 创建目标目录结构
mkdir -p ../data/raw

# 移动所有解压后的日期目录到目标位置
for dir in 2025-06-*; do
    if [ -d "$dir" ]; then
        echo "移动 phaseone 目录: $dir"
        mv "$dir" ../data/raw/
    fi
done


echo "phaseone 数据文件移动完成"

# 返回到项目根目录
cd ..

# 处理 phasetwo  
echo ""
echo "================================================"
echo "处理 phasetwo 数据..."
echo "================================================"

cd phasetwo

# 检查文件完整性
echo "------------------------------------------------"
echo "正在校验 phasetwo 文件完整性..."

# 检查 checksums.md5 文件是否存在并处理行结束符
if [ -f "checksums.md5" ]; then
    # 转换行结束符为 Unix 格式（去除 \r 字符）
    sed -i 's/\r$//' checksums.md5
    
    # 进行 MD5 校验
    md5sum -c checksums.md5
    if [ $? -eq 0 ]; then
        echo "所有 phasetwo 文件下载完整且未被篡改"
    else
        echo "以下 phasetwo 文件存在问题:"
        md5sum -c checksums.md5 | grep -v '成功\|OK' | awk -F: '{print $1}'
    fi
else
    echo "未找到 checksums.md5 文件，跳过完整性校验"
fi

# 解压 phasetwo 数据文件
echo ""
echo "------------------------------------------------"
echo "正在解压 phasetwo 文件..."

if ls *.tar.gz 1> /dev/null 2>&1; then
    for file in *.tar.gz; do tar -zxf "$file"; done
    echo "phasetwo 文件解压完成"
else
    echo "未找到需要解压的 .tar.gz 文件"
fi

# 将解压后的文件夹移动到 data/raw/ 目录下
echo ""
echo "------------------------------------------------"
echo "正在移动解压后的 phasetwo 数据文件..."

# 移动所有解压后的日期目录到目标位置
for dir in 2025-*; do
    if [ -d "$dir" ]; then
        echo "移动 phasetwo 目录: $dir"
        mv "$dir" ../data/raw/
    fi
done


echo "phasetwo 数据文件移动完成"

# 返回到项目根目录
cd ..

echo ""
echo "================================================"
echo "所有数据下载和准备工作完成！"
echo "开始数据预处理..."
echo "================================================"

# ================================================
# 第二阶段：数据预处理（跳过input合并处理）
# ================================================

# 进入 scripts 目录并运行数据处理脚本
cd scripts

echo ""
echo "------------------------------------------------"
echo "正在转换所有 log 数据时间戳..."
python raw_log_processor.py

echo ""
echo "------------------------------------------------"
echo "正在转换所有 metric 数据时间戳..."
python raw_metric_processor.py

echo ""
echo "------------------------------------------------"
echo "正在转换所有 trace 数据时间戳..."
python raw_trace_processor.py

echo ""
echo "所有数据时间戳转换处理完成！"

echo ""
echo "------------------------------------------------"
echo "正在移动处理后的数据到最终位置..."

# 返回到src目录
cd ..

# 检查是否存在 data/processed 目录（当前在src目录下）
if [ -d "data/processed" ]; then
    # 创建目标目录（与src同级的data/processed）
    mkdir -p ../data/processed
    
    # 移动处理后的数据
    echo "移动 data/processed 到 ../data/processed"
    mv data/processed/* ../data/processed/
    
    # 删除空的 data/processed 目录
    rmdir data/processed
    
    echo "处理后的数据移动完成"
else
    echo "警告：未找到 data/processed 目录"
fi

echo ""
echo "------------------------------------------------"
echo "正在清理下载的临时文件夹..."

# 显示当前工作目录用于调试
echo "当前工作目录: $(pwd)"
echo "检查需要删除的文件夹..."
ls -la . | grep -E "(phaseone|phasetwo|data)" || echo "未找到 phaseone、phasetwo 或 data 文件夹"

# 清理下载的 phaseone 文件夹
if [ -d "phaseone" ]; then
    echo "删除 src/phaseone 文件夹"
    rm -rf phaseone
    if [ $? -eq 0 ]; then
        echo "src/phaseone 文件夹删除成功"
    else
        echo "src/phaseone 文件夹删除失败"
    fi
else
    echo "src/phaseone 文件夹不存在，跳过删除"
fi

# 清理下载的 phasetwo 文件夹  
if [ -d "phasetwo" ]; then
    echo "删除 src/phasetwo 文件夹"
    rm -rf phasetwo
    if [ $? -eq 0 ]; then
        echo "src/phasetwo 文件夹删除成功"
    else
        echo "src/phasetwo 文件夹删除失败"
    fi
else
    echo "src/phasetwo 文件夹不存在，跳过删除"
fi

# 清理src目录下的data文件夹（如果存在且为空）
if [ -d "data" ]; then
    echo "检查并删除 src/data 文件夹"
    # 先尝试删除空目录，如果不为空则强制删除
    if rmdir data 2>/dev/null; then
        echo "src/data 空目录删除成功"
    else
        echo "src/data 目录不为空，强制删除"
        rm -rf data
        if [ $? -eq 0 ]; then
            echo "src/data 文件夹强制删除成功"
        else
            echo "src/data 文件夹删除失败"
        fi
    fi
else
    echo "src/data 文件夹不存在，跳过删除"
fi

echo "临时文件夹清理完成"

echo ""
echo "================================================"
echo "所有数据预处理完成！"
echo "- 处理后的数据存储在 data/processed/ 目录下"
echo "- 已跳过input合并处理，使用现有input文件"
echo "- 已清理下载的临时文件夹（src/phaseone, src/phasetwo, src/data）"
echo "================================================"