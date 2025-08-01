import os
import json
import yaml
from typing import Dict, Any, Optional
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from jinja2 import Template


class ConfigParser:
    """配置文件解析器"""

    def __init__(self, env=None):
        # 根据环境变量选择配置文件
        self.env = env or os.getenv('FAQ_ENV', 'test')
        self.raw_config = self._load_config()
        self.parsed_config = self._parse_config()
    
    def _load_config(self):
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return raw_config
    
    def _parse_config(self):
        """
        解析配置，处理环境变量和模板变量
        
        Returns:
            SimpleNamespace: 解析后的配置对象
        """
        # 获取环境相关配置
        env_config = self.raw_config.get(self.env, self.raw_config['test'])
        common_config = {k: v for k, v in self.raw_config.items() if k not in ['prod', 'test']}
        # self.kafka_config = env_config['kafka']
        # self.redis_config = self.raw_config['redis']
        
        # 合并配置
        merged_config = {**env_config, **common_config}

        # 特殊处理kafka配置（因为它在环境配置中）
        if 'kafka' in env_config:
            merged_config['kafka'] = env_config['kafka']
        
        # 处理模板变量
        parsed_config = self._render_templates(merged_config)
        
        # 转换为SimpleNamespace对象以便访问
        return self._dict_to_namespace(parsed_config)
    

    # @property
    # def snapshot_key(self):
    #     return self.env_config['paths']['snapshot_key']

    def _render_templates(self, config):
        """递归渲染配置中的模板变量"""
        # 将配置转换为JSON字符串
        config_str = json.dumps(config)
        
        # 使用Jinja2渲染模板
        template = Template(config_str)
        
        # 多次渲染以处理嵌套的模板变量
        max_iterations = 10
        for _ in range(max_iterations):
            prev_str = config_str
            config_str = template.render(**config)
            template = Template(config_str)
            # config = json.loads(config_str)
            
            # 如果没有变化，说明所有模板都已渲染
            if prev_str == config_str:
                break
        
        # 将JSON字符串转换回字典
        return json.loads(config_str)
    
    def _dict_to_namespace(self, d):
        """
        将字典转换为SimpleNamespace对象
        
        Args:
            d (dict): 字典
            
        Returns:
            SimpleNamespace: 对象
        """
        if isinstance(d, dict):
            # 递归处理嵌套字典
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            # 处理列表中的字典元素
            return [self._dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
        else:
            return d
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.parsed_config
    
    def __repr__(self) -> str:
        return f"ConfigParser(env='{self.env}', config_path='{self.config_path}')"


def main():
    # 单例模式
    parsed_config = ConfigParser().parsed_config

    # 通过属性访问
    print("Path root:", parsed_config.path_root)
    print("Model ft folder:", parsed_config.fine_tuned_folder)
        
    # 访问嵌套配置
    kafka_config = parsed_config.kafka.__dict__    # namespace 转换成 dict 
    print("\nKafka配置:")
    print(f"  Bootstrap servers: {kafka_config['bootstrap_servers']}")
    print(f"  Topic: {kafka_config['topic_name']}")
    print(f"  Group ID: {kafka_config['group_id']}")
    print(f"  Data folder: {kafka_config['kafka_data_folder']}")

    redis_config = parsed_config.redis.__dict__    # namespace 转换成 dict 
    print("\nRedis配置:")
    print(f"  Sentinel list: {redis_config['redis_sentinel_list']}")
    print(f"  Master: {redis_config['sentinel_master']}")

    # 模型配置
    ft_models = parsed_config.fine_tuned_model_config.__dict__
    print("\n微调后模型路径:")
    for model_name, model_path in ft_models.items():
        if model_name != 'npy_path':
            print(f"  {model_name}: {model_path}")

    print("\n排序模型路径:")
    print("  Rerank model path:", parsed_config.rerank_model_path)

    # 访问文件路径配置
    print("\nQA文件路径:")
    print("  paths.qa:", parsed_config.paths.qa.__dict__)
    print("  paths.kafka:", parsed_config.paths.kafka.__dict__)
    print("  paths.entities:", parsed_config.paths.entities.__dict__)

    print("\n训练/测试数据:")
    print("train_similar_qq_path:", parsed_config.train.train_similar_qq_path)
    print("test_set_path:", parsed_config.eval.test_set_path)


# main()
# exit()

cp = ConfigParser() # $FAQ_ENV
parsed_config = cp.config

# 获取模型配置
pretrained_model_config = parsed_config.pretrained_model_config.__dict__
fine_tuned_model_config = parsed_config.fine_tuned_model_config.__dict__
rerank_model_path = parsed_config.rerank_model_path

# 获取定义的所有路径
paths = parsed_config.paths

# kafka配置
kafka_config = parsed_config.kafka

# redis配置
redis_config = parsed_config.redis
snapshot_key = parsed_config.redis.snapshot_key

# # 其他数据
# train_config = parsed_config.train.__dict__
# train_similar_qq_path = train_config["train_similar_qq_path"]
# eval_config = parsed_config.eval.__dict__

# 置信度阈值 & 知识优先级
threshold_priority_config = parsed_config.threshold_priority
# print(threshold_priority_config.__dict__)
