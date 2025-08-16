import os
from dotenv import load_dotenv
from typing import Any


def get_llm_config() -> dict[str, Any]:
    """获取LLM配置"""
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（向上两级）
    project_root = os.path.dirname(current_dir)
    # 构建.env文件的绝对路径
    env_path = os.path.join(project_root, '.env')
    _ = load_dotenv(env_path)

    config_list = [
        {
            'model': 'deepseek-chat',
            'api_key': os.getenv("KEJIYUN_API_KEY"),
            'base_url': os.getenv("KEJIYUN_API_BASE"),
        },
        # {
        #     'model': 'deepseek-v3:671b',
        #     'api_key': os.getenv("KEJIYUN_API_KEY"),
        #     'base_url': os.getenv("KEJIYUN_API_BASE"),
        # },
        # {
        #     'model': 'deepseek-r1:671b',
        #     'api_key': os.getenv("KEJIYUN_API_KEY"),
        #     'base_url': os.getenv("KEJIYUN_API_BASE"),
        # },
        # {
        #     'model': 'deepseek-r1:671b-0528',
        #     'api_key': os.getenv("KEJIYUN_API_KEY"),
        #     'base_url': os.getenv("KEJIYUN_API_BASE"),
        # },
        # {
        #     'model': 'deepseek-r1:671b-64k',
        #     'api_key': os.getenv("KEJIYUN_API_KEY"),
        #     'base_url': os.getenv("KEJIYUN_API_BASE"),
        # },
        # {
        #     'model': 'qwen3:235b',
        #     'api_key': os.getenv("KEJIYUN_API_KEY"),
        #     'base_url': os.getenv("KEJIYUN_API_BASE"),
        # },
    ]

    return {
        'config_list': config_list,
        'seed': 42,
        'temperature': 0,
    }