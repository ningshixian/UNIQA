from setuptools import setup, find_packages
import subprocess
import sys
import re


def run_command(cmd, capture=True):
    """运行命令的通用函数"""
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True, shell=True)
        return result.stdout if result.returncode == 0 else None
    except:
        return None

def get_cuda_version():
    """获取 CUDA 版本"""
    # 尝试多种方式获取 CUDA 版本
    commands = [
        "nvidia-smi | grep -oP 'CUDA Version:\\s*\\K[\\d.]+'",
        "nvcc --version | grep -oP 'release \\K[\\d.]+'",
    ]
    
    for cmd in commands:
        output = run_command(cmd)
        if output:
            return output.strip()
    
    # 尝试从 PyTorch 获取
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except:
        pass
    
    return None


# 检测 CUDA 版本
cuda_version = get_cuda_version()
if cuda_version.startswith("11.7"):
    with open("requirements-cu117.txt",encoding='utf-8') as fp:
        requirements = fp.read().splitlines()
elif cuda_version.startswith("12.1"):
    with open("requirements-cu121.txt",encoding='utf-8') as fp:
        requirements = fp.read().splitlines()

with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()
version = {}
with open("flashrag/version.py", encoding="utf8") as fp:
    exec(fp.read(), version)

extras_require = {
    'core': requirements,
    'retriever': ['pyserini', 'sentence-transformers>=3.0.1'],
    'generator': ['vllm'],
    'multimodal': ['timm', 'torchvision', 'pillow', 'qwen_vl_utils']
}
extras_require['full'] = sum(extras_require.values(), [])

setup(
    name="flashrag_dev",
    version=version['__version__'],
    packages=find_packages(),
    url="https://github.com/RUC-NLPIR/FlashRAG",
    license="MIT License",
    author="Jiajie Jin, Yutao Zhu, Chenghao Zhang, Xinyu Yang, Zhicheng Dou",
    author_email="jinjiajie@ruc.edu.cn",
    description="A library for efficient Retrieval-Augmented Generation research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={"": ["*.yaml"]},
    include_package_data=True,
    install_requires=extras_require['core'],
    extras_require=extras_require,
    python_requires=">=3.9",
)