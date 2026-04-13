#!/usr/bin/env python3
"""
LLM差异分析器
用于生成问题、收集不同模型的回答、分析差异、打标签并分析相关性
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Union
from pathlib import Path

# 结果输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# 尝试导入可选依赖
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def load_env_file():
    """加载 .env 文件"""
    if HAS_DOTENV and os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE)


def expand_env_vars(value: str) -> str:
    """展开字符串中的环境变量引用 ${VAR_NAME}"""
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}]+)\}'
    
    def replace_env(match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))
    
    return re.sub(pattern, replace_env, value)


def expand_config_env_vars(config: dict) -> dict:
    """递归展开配置中所有环境变量引用"""
    if isinstance(config, dict):
        return {k: expand_config_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_config_env_vars(item) for item in config]
    elif isinstance(config, str):
        return expand_env_vars(config)
    return config


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    优先级:
    1. 指定的配置文件路径
    2. 环境变量 LLM_DIFF_CONFIG
    3. 默认的 config.yaml
    """
    # 先加载环境变量
    load_env_file()
    
    # 确定配置文件路径
    if config_path:
        path = config_path
    elif os.getenv('LLM_DIFF_CONFIG'):
        path = os.getenv('LLM_DIFF_CONFIG')
    else:
        path = CONFIG_FILE
    
    # 加载 YAML 配置
    if os.path.exists(path):
        if not HAS_YAML:
            raise ImportError("需要安装 PyYAML: pip install pyyaml")
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # 展开环境变量
        config = expand_config_env_vars(config)
        return config
    
    return {}


@dataclass
class ModelConfig:
    """单个模型的完整配置"""
    model_name: str  # 模型标识名称
    api_type: str = "openai"  # openai, ollama, custom
    api_key: str = ""  # API密钥，为空则从环境变量读取
    base_url: str = ""  # API基础URL，为空则使用默认值
    
    def __post_init__(self):
        """初始化后处理默认值"""
        if not self.api_key:
            self.api_key = os.getenv('OPENAI_API_KEY', '')
        if not self.base_url:
            default_urls = {
                'openai': 'https://api.openai.com/v1',
                'ollama': 'http://localhost:11434',
            }
            self.base_url = default_urls.get(self.api_type, '')


@dataclass
class ModelResponse:
    """模型回答的数据结构"""
    model_name: str
    response: str
    tags: list = field(default_factory=list)
    diff_keywords: list = field(default_factory=list)


@dataclass
class QuestionResult:
    """单个问题的结果"""
    question: str
    responses: list = field(default_factory=list)
    diff_summary: str = ""
    merged_tags: dict = field(default_factory=dict)
    tag_correlation: dict = field(default_factory=dict)


class LLMClient:
    """LLM客户端封装 - 支持多模型独立配置"""
    
    # OpenAI客户端缓存，避免重复创建
    _clients_cache: dict = {}
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.api_type = self.config.get('api_type', 'openai')
        self.api_key = self.config.get('api_key', os.getenv('OPENAI_API_KEY', ''))
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
    
    def _get_client(self, model_config: ModelConfig = None) -> tuple:
        """
        获取API配置参数
        返回: (api_type, api_key, base_url, openai_client_or_none)
        """
        if model_config:
            api_type = model_config.api_type
            api_key = model_config.api_key
            base_url = model_config.base_url
        else:
            api_type = self.api_type
            api_key = self.api_key
            base_url = self.base_url
        
        # 对于OpenAI类型，使用缓存的客户端
        openai_client = None
        if api_type == 'openai' and HAS_OPENAI:
            cache_key = (api_key, base_url)
            if cache_key not in self._clients_cache:
                self._clients_cache[cache_key] = OpenAI(api_key=api_key, base_url=base_url)
            openai_client = self._clients_cache[cache_key]
        
        return api_type, api_key, base_url, openai_client
    
    def chat(self, model: str, messages: list, model_config: ModelConfig = None, **kwargs) -> str:
        """
        发送聊天请求
        
        Args:
            model: 模型名称
            messages: 消息列表
            model_config: 可选的模型配置，用于覆盖默认配置
            **kwargs: 其他参数传递给API
        
        Returns:
            模型响应文本
        """
        api_type, api_key, base_url, openai_client = self._get_client(model_config)
        
        if api_type == 'openai' and openai_client:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        
        elif api_type == 'ollama' and HAS_REQUESTS:
            url = f"{base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            response = requests.post(url, json=payload)
            return response.json()['message']['content']
        
        elif api_type == 'custom' and HAS_REQUESTS:
            url = f"{base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            response = requests.post(url, json=payload, headers=headers)
            return response.json()['choices'][0]['message']['content']
        
        else:
            raise RuntimeError(f"不支持的API类型: {api_type} 或缺少依赖")


# 示例问题列表 - 用于生成类似风格的新问题
EXAMPLE_QUESTIONS = [
    "世界哪个国家一天当中吃的固体食物总克数平均最多，每顿分别哪个国家最多",
    "诺夫哥罗德共和国最东边最北边分别是哪里",
    "古菌是什么",
    "介绍南非金矿 ~5 km 深处的细菌",
    "缸中之脑和玻尔兹曼大脑哪个更恐怖",
    "16岁的人会骄傲于自己年轻吗",
    "对比大五 荣格八维 九型 16pf的逼格",
    "介绍埃尔米特矩阵",
    "github仓库的star多不多和开发者的智商 数学好坏 记忆力 年龄 性别 是否有钱有关吗，分别估计一下",
    "50后亲00后之后，50后和00后嘴的气味分别会有哪些变化",
    "中国00后最嫌弃中国几0后陌生人的口水（字面意思），估计一下，给个排名",
    "00后怎么看待00后口水里面细菌比50后少",
    "平均来看00后口水里面细菌是否比50后少",
    "要是00后和跟别人说话时别人口水（字面意思）喷到00后嘴里了，中国00后最嫌弃中国几0后陌生人的口水（字面意思），估计一下，给个排名",
    "为什么美国各州的平均智商不同",
    "抽象代数里面最优雅的是什么",
    "详细介绍域扩张",
    "数论包括什么",
    "运动员 数学家 科学家 哲学家口臭比例对比所有人平均水平",
    "∣a∣和∥A∥的区别是什么",
    "左伴随是什么",
    "遗忘函子是什么，自由群函子是什么",
    "有没有像缸中之脑和玻尔兹曼大脑一样恐怖的",
    "拉美黑人常染比例最高的国家是哪个",
    "16岁大五和16岁时感兴趣的领域有关吗",
    "对比一下德国各地区工业化完成的时间",
    "群论 环论 域论哪个最优雅",
    "哪个年龄开发者更在意逼格，国内",
    "对比2004至今不同时期中国16岁开发者眼中什么逼格最高",
    "1886年以前德兰士瓦按照五段论是资本主义社会还是封建社会",
    "排中律是什么",
]


class LLMDiffAnalyzer:
    """LLM差异分析器主类"""
    
    def __init__(self, llm_client: LLMClient, analyzer_model: str = "gpt-4",
                 analyzer_config: ModelConfig = None):
        """
        初始化分析器
        
        Args:
            llm_client: LLM客户端实例
            analyzer_model: 分析器使用的模型名称（向后兼容）
            analyzer_config: 分析器模型的完整配置（可选，用于独立配置分析器）
        """
        self.llm = llm_client
        self.analyzer_model = analyzer_model
        self.analyzer_config = analyzer_config
        self.results: list[QuestionResult] = []
    
    def generate_questions(self, topic: str, count: int = 5, 
                          generator_model: str = "gpt-4",
                          generator_config: ModelConfig = None) -> list[str]:
        """用AI生成问题"""
        prompt = f"""请生成{count}个关于"{topic}"的深度问题。
要求：
1. 问题应该有深度，需要分析和思考
2. 问题之间要有差异性，覆盖不同角度
3. 每个问题一行，不要编号

请直接输出问题，每行一个："""
        
        response = self.llm.chat(
            generator_model, 
            [{"role": "user", "content": prompt}],
            model_config=generator_config
        )
        questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return questions[:count]
    
    def generate_similar_questions(self, count: int = 5,
                                   generator_model: str = "gpt-4",
                                   generator_config: ModelConfig = None,
                                   example_questions: list[str] = None) -> list[str]:
        """
        根据示例问题生成类似风格的新问题
        
        Args:
            count: 生成问题数量
            generator_model: 生成模型名称
            generator_config: 生成模型配置
            example_questions: 自定义示例问题列表，为None则使用默认示例
        
        Returns:
            生成的新问题列表
        """
        examples = example_questions if example_questions else EXAMPLE_QUESTIONS
        
        # 随机选择一些示例作为参考
        import random
        sample_examples = random.sample(examples, min(15, len(examples)))
        
        prompt = f"""以下是{len(sample_examples)}个问题示例，请仔细分析这些问题的风格特点，然后生成{count}个风格类似的新问题。

示例问题：
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(sample_examples))}

问题风格特点分析：
- 跨学科、奇特、深入思考
- 涵盖数学、哲学、生物学、社会学、心理学、历史、地理等领域
- 有些问题涉及比较、对比、排名
- 有些问题涉及具体概念解释
- 有些问题涉及社会现象和群体行为
- 语言风格直接、简洁、有时带有幽默感
- 问题可能涉及"逼格"、"口水"、"口臭"等非正式用语
- 问题可能涉及跨代比较（00后、50后等）
- 问题可能涉及抽象数学概念（群论、域论、函子等）

请生成{count}个新问题，要求：
1. 保持类似的风格和语调
2. 覆盖不同领域（数学、科学、社会、文化等）
3. 问题应该有趣、有深度、引人思考
4. 不要重复示例问题
5. 每个问题一行，不要编号

请直接输出问题，每行一个："""
        
        response = self.llm.chat(
            generator_model, 
            [{"role": "user", "content": prompt}],
            model_config=generator_config
        )
        questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        # 过滤掉可能的编号前缀
        questions = [re.sub(r'^\d+[\.\)、\s]+', '', q).strip() for q in questions]
        return questions[:count]
    
    def get_responses(self, question: str, models: Union[list[str], list[ModelConfig]]) -> list[ModelResponse]:
        """
        用不同模型回答问题
        
        Args:
            question: 问题文本
            models: 模型列表，可以是：
                    - 字符串列表 ["gpt-4", "gpt-3.5-turbo"]（使用默认客户端配置）
                    - ModelConfig列表（每个模型独立配置）
        
        Returns:
            模型响应列表
        """
        responses = []
        for model in models:
            if isinstance(model, ModelConfig):
                # 新模式：使用独立的模型配置
                model_name = model.model_name
                try:
                    answer = self.llm.chat(
                        model.model_name, 
                        [{"role": "user", "content": question}],
                        model_config=model
                    )
                    responses.append(ModelResponse(model_name=model_name, response=answer))
                except Exception as e:
                    print(f"模型 {model_name} 回答失败: {e}")
                    responses.append(ModelResponse(model_name=model_name, response=f"[错误: {e}]"))
            else:
                # 向后兼容：字符串模型名，使用默认客户端配置
                model_name = model
                try:
                    answer = self.llm.chat(model_name, [{"role": "user", "content": question}])
                    responses.append(ModelResponse(model_name=model_name, response=answer))
                except Exception as e:
                    print(f"模型 {model_name} 回答失败: {e}")
                    responses.append(ModelResponse(model_name=model_name, response=f"[错误: {e}]"))
        return responses
    
    def analyze_differences(self, question: str, responses: list[ModelResponse],
                          model_config: ModelConfig = None) -> tuple[str, list[list[str]]]:
        """分析不同模型输出的差异"""
        # 构建对比提示
        responses_text = "\n\n".join([
            f"【{r.model_name}】:\n{r.response}"
            for r in responses
        ])
        
        prompt = f"""请分析以下不同模型对同一问题的回答差异。

问题：{question}

{responses_text}

请完成以下任务：
1. 用3-5个关键词概括每个回答的主要特点（格式：模型名: 关键词1, 关键词2, 关键词3）
2. 总结不同回答之间的主要差异（格式：差异总结: ...）

请按以下格式输出：
关键词分析:
模型名1: 词1, 词2, 词3
模型名2: 词1, 词2, 词3
...

差异总结: 具体差异描述"""
        
        analysis = self.llm.chat(
            self.analyzer_model, 
            [{"role": "user", "content": prompt}],
            model_config=model_config
        )
        
        # 解析结果
        diff_keywords = {}
        diff_summary = ""
        
        lines = analysis.split('\n')
        in_keywords = False
        
        for line in lines:
            line = line.strip()
            if '关键词分析' in line:
                in_keywords = True
                continue
            if '差异总结' in line:
                in_keywords = False
                # 提取差异总结
                if ':' in line:
                    diff_summary = line.split(':', 1)[1].strip()
                continue
            
            if in_keywords and ':' in line:
                parts = line.split(':', 1)
                model = parts[0].strip()
                keywords = [k.strip() for k in parts[1].split(',')]
                diff_keywords[model] = keywords
        
        # 更新responses的diff_keywords
        for r in responses:
            if r.model_name in diff_keywords:
                r.diff_keywords = diff_keywords[r.model_name]
        
        return diff_summary, [r.diff_keywords for r in responses]
    
    def tag_responses(self, question: str, responses: list[ModelResponse],
                      model_config: ModelConfig = None) -> None:
        """为各个回答打标签"""
        for response in responses:
            prompt = f"""请为以下回答打上标签。

问题：{question}
回答：{response.response}

请给出5-8个标签来描述这个回答的特点，包括：
- 内容风格（如：详细、简洁、学术化、口语化等）
- 论证方式（如：举例说明、逻辑推理、引用数据等）
- 内容质量（如：准确、有深度、有创意等）
- 其他特点

请直接输出标签，用逗号分隔："""
            
            tags_str = self.llm.chat(
                self.analyzer_model, 
                [{"role": "user", "content": prompt}],
                model_config=model_config
            )
            response.tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    
    def merge_similar_tags(self, results: list[QuestionResult],
                          model_config: ModelConfig = None) -> dict:
        """合并同义词或近义词标签"""
        # 收集所有标签
        all_tags = []
        for result in results:
            for response in result.responses:
                all_tags.extend(response.tags)
        
        if not all_tags:
            return {}
        
        unique_tags = list(set(all_tags))
        
        prompt = f"""请分析以下标签，将同义词或近义词合并。

标签列表：
{', '.join(unique_tags)}

请将意思相近的标签归类，输出格式：
类别名: 标签1, 标签2, 标签3

每个类别一行，确保所有标签都被归类。类别名应该是一个能代表该类标签的通用词。"""
        
        merge_result = self.llm.chat(
            self.analyzer_model, 
            [{"role": "user", "content": prompt}],
            model_config=model_config
        )
        
        # 解析合并结果
        merged = {}
        for line in merge_result.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                category = parts[0].strip()
                tags = [t.strip() for t in parts[1].split(',')]
                merged[category] = tags
        
        # 为每个结果添加合并后的标签映射
        for result in results:
            for response in result.responses:
                tag_mapping = {}
                for tag in response.tags:
                    for category, tags in merged.items():
                        if tag in tags:
                            tag_mapping[tag] = category
                            break
                result.merged_tags = merged
        
        return merged
    
    def analyze_tag_correlation(self, results: list[QuestionResult],
                                model_config: ModelConfig = None) -> dict:
        """分析不同标签在不同回答之间的相关性"""
        # 统计标签共现
        tag_pairs = defaultdict(int)
        tag_counts = defaultdict(int)
        model_tags = defaultdict(set)
        
        for result in results:
            for response in result.responses:
                model_tags[response.model_name].update(response.tags)
                for tag in response.tags:
                    tag_counts[tag] += 1
                
                # 统计标签共现
                tags = response.tags
                for i in range(len(tags)):
                    for j in range(i + 1, len(tags)):
                        pair = tuple(sorted([tags[i], tags[j]]))
                        tag_pairs[pair] += 1
        
        # 用LLM深入分析相关性
        correlation_data = {
            "tag_counts": dict(tag_counts),
            "tag_pairs": dict(tag_pairs),
            "model_tags": {k: list(v) for k, v in model_tags.items()}
        }
        
        prompt = f"""请分析以下标签数据，找出标签之间的相关性和模式。

标签出现次数：{json.dumps(correlation_data['tag_counts'], ensure_ascii=False, indent=2)}

标签共现次数（同一回答中同时出现的标签）：{json.dumps(correlation_data['tag_pairs'], ensure_ascii=False, indent=2)}

各模型常见标签：{json.dumps(correlation_data['model_tags'], ensure_ascii=False, indent=2)}

请分析：
1. 哪些标签经常一起出现？说明什么特点？
2. 不同模型的标签有什么偏好？
3. 标签之间有什么潜在的相关性或因果关系？

请输出JSON格式的分析结果："""
        
        analysis = self.llm.chat(
            self.analyzer_model, 
            [{"role": "user", "content": prompt}],
            model_config=model_config
        )
        
        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', analysis)
            if json_match:
                correlation_analysis = json.loads(json_match.group())
            else:
                correlation_analysis = {"raw_analysis": analysis}
        except json.JSONDecodeError:
            correlation_analysis = {"raw_analysis": analysis}
        
        # 更新结果
        for result in results:
            result.tag_correlation = correlation_analysis
        
        return correlation_analysis
    
    def run_full_analysis(self, topic: str = None, 
                         models: Union[list[str], list[ModelConfig]] = None,
                         question_count: int = 5,
                         generator_model: str = "gpt-4",
                         generator_config: ModelConfig = None,
                         use_example_style: bool = False,
                         example_questions: list[str] = None) -> list[QuestionResult]:
        """
        运行完整分析流程
        
        Args:
            topic: 分析主题（use_example_style=True时可为None）
            models: 模型列表，可以是：
                    - 字符串列表 ["gpt-4", "gpt-3.5-turbo"]
                    - ModelConfig列表（每个模型独立配置）
            question_count: 生成问题数量
            generator_model: 问题生成模型名称
            generator_config: 问题生成模型的独立配置
            use_example_style: 是否使用示例问题风格生成问题
            example_questions: 自定义示例问题列表
        
        Returns:
            分析结果列表
        """
        if models is None:
            models = []
        
        if use_example_style:
            print(f"=== 开始分析：示例风格问题生成模式 ===")
        else:
            print(f"=== 开始分析：{topic} ===")
        
        # 1. 生成问题
        print("\n[1/6] 生成问题...")
        if use_example_style:
            questions = self.generate_similar_questions(
                count=question_count,
                generator_model=generator_model,
                generator_config=generator_config,
                example_questions=example_questions
            )
        else:
            questions = self.generate_questions(
                topic, question_count, 
                generator_model=generator_model,
                generator_config=generator_config
            )
        print(f"生成了 {len(questions)} 个问题")
        
        # 2. 收集回答
        print("\n[2/6] 收集各模型回答...")
        for i, question in enumerate(questions):
            print(f"  处理问题 {i+1}/{len(questions)}: {question[:50]}...")
            result = QuestionResult(question=question)
            result.responses = self.get_responses(question, models)
            self.results.append(result)
        
        # 3. 分析差异
        print("\n[3/6] 分析回答差异...")
        for result in self.results:
            diff_summary, _ = self.analyze_differences(
                result.question, result.responses,
                model_config=self.analyzer_config
            )
            result.diff_summary = diff_summary
            print(f"  差异摘要: {diff_summary[:100]}...")
        
        # 4. 打标签
        print("\n[4/6] 为回答打标签...")
        for result in self.results:
            self.tag_responses(
                result.question, result.responses,
                model_config=self.analyzer_config
            )
            for r in result.responses:
                print(f"  {r.model_name} 标签: {', '.join(r.tags[:5])}")
        
        # 5. 合并相似标签
        print("\n[5/6] 合并相似标签...")
        merged = self.merge_similar_tags(self.results, model_config=self.analyzer_config)
        print(f"合并为 {len(merged)} 个类别")
        
        # 6. 分析相关性
        print("\n[6/6] 分析标签相关性...")
        correlation = self.analyze_tag_correlation(self.results, model_config=self.analyzer_config)
        
        print("\n=== 分析完成 ===")
        return self.results
    
    def save_results(self, filepath: str, output_dir: str = None) -> None:
        """保存结果到JSON文件"""
        # 确保输出目录存在
        if output_dir is None:
            output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果filepath是相对路径，则拼接到output_dir
        if not os.path.isabs(filepath):
            filepath = os.path.join(output_dir, filepath)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        for result in self.results:
            result_data = {
                "question": result.question,
                "diff_summary": result.diff_summary,
                "merged_tags": result.merged_tags,
                "tag_correlation": result.tag_correlation,
                "responses": []
            }
            
            for response in result.responses:
                result_data["responses"].append({
                    "model_name": response.model_name,
                    "response": response.response,
                    "tags": response.tags,
                    "diff_keywords": response.diff_keywords
                })
            
            data["results"].append(result_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {filepath}")
    
    def generate_report(self, filepath: str, output_dir: str = None) -> None:
        """生成Markdown格式的分析报告"""
        # 确保输出目录存在
        if output_dir is None:
            output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果filepath是相对路径，则拼接到output_dir
        if not os.path.isabs(filepath):
            filepath = os.path.join(output_dir, filepath)
        
        report = []
        report.append("# LLM差异分析报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.append("---\n\n")
        
        for i, result in enumerate(self.results, 1):
            report.append(f"## 问题 {i}\n\n")
            report.append(f"**{result.question}**\n\n")
            
            report.append("### 各模型回答\n\n")
            for response in result.responses:
                report.append(f"#### {response.model_name}\n\n")
                report.append(f"**关键词**: {', '.join(response.diff_keywords)}\n\n")
                report.append(f"**标签**: {', '.join(response.tags)}\n\n")
                report.append(f"**回答**:\n\n{response.response[:500]}...\n\n")
            
            report.append(f"### 差异摘要\n\n{result.diff_summary}\n\n")
            report.append("---\n\n")
        
        # 标签相关性分析
        if self.results and self.results[0].tag_correlation:
            report.append("## 标签相关性分析\n\n")
            correlation = self.results[0].tag_correlation
            if "raw_analysis" in correlation:
                report.append(correlation["raw_analysis"])
            else:
                report.append("```json\n")
                report.append(json.dumps(correlation, ensure_ascii=False, indent=2))
                report.append("\n```\n")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"报告已保存到: {filepath}")


def create_model_config_from_dict(model_dict: dict) -> ModelConfig:
    """从配置字典创建 ModelConfig"""
    return ModelConfig(
        model_name=model_dict.get('model_name', ''),
        api_type=model_dict.get('api_type', 'openai'),
        api_key=model_dict.get('api_key', ''),
        base_url=model_dict.get('base_url', '')
    )


def main():
    """主函数 - 使用配置文件和环境变量"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM差异分析器')
    parser.add_argument('--topic', type=str, default=None, 
                        help='分析主题（不指定则使用示例问题风格）')
    parser.add_argument('--count', type=int, default=None, 
                        help='生成问题数量')
    parser.add_argument('--example-style', action='store_true', default=None,
                        help='使用示例问题风格生成问题')
    parser.add_argument('--no-example-style', action='store_false', dest='example_style',
                        help='不使用示例问题风格，改用传统主题模式')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（默认: config.yaml）')
    parser.add_argument('--list-models', action='store_true',
                        help='列出配置中的模型并退出')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果没有配置文件，提示用户
    if not config:
        print("警告: 未找到配置文件，请创建 config.yaml 或设置环境变量")
        print("参考 config.example.yaml 和 .env.example 创建配置")
        return
    
    # 获取分析配置
    analysis_config = config.get('analysis', {})
    analyzer_dict = config.get('analyzer', {})
    generator_dict = config.get('generator', {})
    models_list = config.get('models', [])
    
    # 命令行参数覆盖配置文件
    question_count = args.count if args.count is not None else analysis_config.get('question_count', 5)
    use_example_style = args.example_style if args.example_style is not None else analysis_config.get('use_example_style', True)
    
    # 列出模型模式
    if args.list_models:
        print("=== 配置中的模型 ===")
        print(f"\n分析模型: {analyzer_dict.get('model_name', 'N/A')}")
        print(f"问题生成模型: {generator_dict.get('model_name', 'N/A')}")
        print(f"\n回答模型 ({len(models_list)} 个):")
        for m in models_list:
            status = "启用" if m.get('enabled', True) else "禁用"
            print(f"  - {m.get('name', m.get('model_name'))}: {status}")
        return
    
    # 创建分析器配置
    analyzer_config = create_model_config_from_dict(analyzer_dict)
    generator_config = create_model_config_from_dict(generator_dict)
    
    # 创建客户端
    client_config = {
        'api_type': analyzer_dict.get('api_type', 'openai'),
        'api_key': analyzer_dict.get('api_key', ''),
        'base_url': analyzer_dict.get('base_url', '')
    }
    client = LLMClient(client_config)
    
    # 创建分析器
    analyzer = LLMDiffAnalyzer(
        llm_client=client,
        analyzer_model=analyzer_config.model_name,
        analyzer_config=analyzer_config
    )
    
    # 创建回答模型配置列表
    model_configs = []
    for m in models_list:
        if m.get('enabled', True):
            model_name = m.get('name', m.get('model_name'))
            mc = ModelConfig(
                model_name=model_name,
                api_type=m.get('api_type', 'openai'),
                api_key=m.get('api_key', ''),
                base_url=m.get('base_url', '')
            )
            model_configs.append(mc)
    
    if not model_configs:
        print("错误: 没有启用的回答模型，请检查配置文件")
        return
    
    # 运行分析
    print("=== 模型配置信息 ===")
    print(f"分析模型: {analyzer_config.model_name}")
    print(f"问题生成模型: {generator_config.model_name}")
    print(f"回答模型: {', '.join(m.model_name for m in model_configs)}")
    print(f"问题数量: {question_count}")
    if use_example_style:
        print(f"问题生成模式: 示例问题风格")
    else:
        print(f"问题生成模式: 传统主题模式 - {args.topic or '人工智能与人类未来的关系'}")
    print()
    
    results = analyzer.run_full_analysis(
        topic=args.topic,
        models=model_configs,
        question_count=question_count,
        generator_model=generator_config.model_name,
        generator_config=generator_config,
        use_example_style=use_example_style
    )
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analyzer.save_results(f"analysis_results_{timestamp}.json")
    analyzer.generate_report(f"analysis_report_{timestamp}.md")


if __name__ == "__main__":
    main()
