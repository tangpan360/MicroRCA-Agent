FROM python:3.10-slim

WORKDIR /app

# 设置pip镜像源（防止网络问题）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

# 复制requirements.txt先安装依赖（利用Docker缓存）
COPY src/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和数据
COPY src/ /app/
COPY data/ /app/data/

# 创建输出目录
RUN mkdir -p /app/output

# 创建非root用户（使用ARG来接收用户ID）
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID appuser && \
    useradd -m -u $USER_ID -g $GROUP_ID appuser && \
    chown -R appuser:appuser /app

# 切换到非root用户
USER appuser

# 运行程序，然后将结果从submission复制到output
CMD python main_multiprocessing.py && \
    cp submission/result.jsonl /app/output/result.jsonl