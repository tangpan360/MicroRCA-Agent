#!/bin/bash

echo "=== 开始构建和运行 AIops 解决方案 ==="

# 检查 data/processed 是否存在
if [ -d "./data/processed" ]; then
    echo "检测到 data/processed 目录，跳过数据预处理"
else
    echo "未检测到 data/processed 目录，开始执行数据预处理..."
    # 进入 src 目录执行预处理脚本
    cd src
    bash preprocessing.sh
    if [ $? -ne 0 ]; then
        echo "错误: 数据预处理失败"
        cd .. # 确保即使失败也回到根目录
        exit 1
    fi
    # 返回到项目根目录
    cd ..
fi

# 每次运行前清理 output 目录和 answer.json 文件
if [ -d "./output" ]; then
    echo "检测到 output 目录，正在删除..."
    rm -rf ./output
fi

if [ -f "./answer.json" ]; then
    echo "检测到 answer.json 文件，正在删除..."
    rm -f ./answer.json
fi

# 检查Docker状态
if ! docker info >/dev/null 2>&1; then
    echo "错误: Docker 服务未运行或权限不足"
    echo "请确保: 1) Docker 服务已启动 2) 当前用户在docker组中"
    exit 1
fi

# 创建输出目录
mkdir -p ./output

# 删除旧的Docker镜像（如果存在）
echo "检查并删除旧的Docker镜像..."
if docker images aiops-solution:latest -q 2>/dev/null | grep -q .; then
    echo "发现旧的aiops-solution镜像，正在删除..."
    docker rmi aiops-solution:latest -f
    echo "旧镜像已删除"
fi

# 清理悬空镜像
echo "清理悬空镜像..."
PRUNED=$(docker image prune -f)
echo "$PRUNED"

# 清理Docker构建缓存（仅清理未使用的缓存，保留活跃缓存）
echo "清理Docker构建缓存..."
BUILD_CACHE_PRUNED=$(docker builder prune -f)
echo "$BUILD_CACHE_PRUNED"

# 显示当前Docker空间使用情况
echo "当前Docker空间使用情况:"
docker system df

# 构建Docker镜像（添加更多调试信息）
echo "开始构建Docker镜像..."
if ! docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t aiops-solution . --no-cache; then
    echo "错误: Docker 镜像构建失败"
    exit 1
fi

echo "Docker镜像构建成功，开始运行容器..."

# 运行容器
echo "启动Docker容器..."
CONTAINER_ID=$(docker run -d --rm -u $(id -u):$(id -g) -v $(pwd)/output:/app/output aiops-solution)

# 设置信号处理函数
cleanup() {
    echo "接收到中断信号，正在停止容器..."
    docker stop $CONTAINER_ID 2>/dev/null
    echo "容器已停止"
    exit 1
}

# 捕获中断信号
trap cleanup SIGINT SIGTERM

# 等待容器完成
echo "等待容器执行完成..."
echo "提示: 按 Ctrl+C 可以安全停止容器"
if ! docker wait $CONTAINER_ID; then
    echo "错误: 容器运行失败"
    docker logs $CONTAINER_ID
    exit 1
fi

# 获取容器退出状态
EXIT_CODE=$(docker inspect $CONTAINER_ID --format='{{.State.ExitCode}}' 2>/dev/null || echo "unknown")
if [ "$EXIT_CODE" != "0" ] && [ "$EXIT_CODE" != "unknown" ]; then
    echo "错误: 容器异常退出，退出码: $EXIT_CODE"
    docker logs $CONTAINER_ID
    exit 1
fi

# 检查结果文件是否存在
if [ ! -f "./output/result.jsonl" ]; then
    echo "错误: 结果文件 ./output/result.jsonl 不存在"
    echo "检查容器输出目录:"
    ls -la ./output/
    exit 1
fi

# 复制结果为最终答案文件
cp ./output/result.jsonl ./result.jsonl

echo "=== 答案文件已生成: result.jsonl ==="

# 可选：在成功完成后清理Docker镜像以节省空间
# 取消注释下面的行来启用自动清理
echo "清理Docker镜像以节省空间..."
docker rmi aiops-solution:latest -f 2>/dev/null || true
docker image prune -f >/dev/null 2>&1 || true