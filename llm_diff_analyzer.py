#!/usr/bin/env python3
"""
LLM差异分析器
用于生成问题、收集不同模型的回答、分析差异、打标签并分析相关性
支持长期持续运行和问题去重
"""

import json
import os
import re
import time
import signal
import hashlib
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Union, Optional, List, Dict
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


class QuestionHistory:
    """问题历史管理类 - 用于追踪已生成的问题并避免重复"""
    
    def __init__(self, history_file: str = None, similarity_threshold: float = 0.85):
        """
        初始化问题历史管理器
        
        Args:
            history_file: 历史记录文件路径
            similarity_threshold: 相似度阈值（0-1），超过此值视为重复
        """
        self.history_file = history_file or os.path.join(OUTPUT_DIR, "question_history.json")
        self.similarity_threshold = similarity_threshold
        self.questions: Dict[str, dict] = {}  # question_hash -> {question, timestamp, count}
        self.question_hashes: set = set()  # 快速查找用
        self._lock = threading.Lock()
        self._load_history()
    
    def _load_history(self):
        """加载历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.questions = data.get('questions', {})
                    self.question_hashes = set(self.questions.keys())
                print(f"已加载 {len(self.questions)} 条历史问题记录")
            except Exception as e:
                print(f"加载历史记录失败: {e}")
                self.questions = {}
                self.question_hashes = set()
    
    def _save_history(self):
        """保存历史记录"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with self._lock:
            try:
                data = {
                    'version': 1,
                    'last_updated': datetime.now().isoformat(),
                    'total_questions': len(self.questions),
                    'questions': self.questions
                }
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存历史记录失败: {e}")
    
    def _normalize_question(self, question: str) -> str:
        """标准化问题文本（去除空格、标点等）"""
        # 去除首尾空格
        q = question.strip()
        # 统一标点符号
        q = re.sub(r'[？?！!。，,.]', '', q)
        # 去除多余空格
        q = re.sub(r'\s+', '', q)
        return q.lower()
    
    def _compute_hash(self, question: str) -> str:
        """计算问题的哈希值"""
        normalized = self._normalize_question(question)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _compute_similarity(self, q1: str, q2: str) -> float:
        """
        计算两个问题的相似度（基于字符级别的Jaccard相似度）
        
        Args:
            q1, q2: 两个问题文本
        
        Returns:
            相似度值 0-1
        """
        # 标准化
        n1 = self._normalize_question(q1)
        n2 = self._normalize_question(q2)
        
        # 字符集合
        set1 = set(n1)
        set2 = set(n2)
        
        # Jaccard 相似度
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate(self, question: str) -> tuple[bool, Optional[str]]:
        """
        检查问题是否重复
        
        Args:
            question: 待检查的问题
        
        Returns:
            (是否重复, 相似问题的哈希值)
        """
        q_hash = self._compute_hash(question)
        
        with self._lock:
            # 精确匹配
            if q_hash in self.question_hashes:
                return True, q_hash
            
            # 相似度检查（对于新问题，检查是否与历史问题相似）
            normalized_q = self._normalize_question(question)
            
            # 如果问题太短（少于5个字符），不做相似度检查
            if len(normalized_q) < 5:
                return False, None
            
            # 检查与历史问题的相似度
            for h, data in self.questions.items():
                hist_q = data.get('question', '')
                similarity = self._compute_similarity(question, hist_q)
                
                if similarity >= self.similarity_threshold:
                    return True, h
        
        return False, None
    
    def add_question(self, question: str) -> bool:
        """
        添加问题到历史记录
        
        Args:
            question: 问题文本
        
        Returns:
            是否成功添加（如果重复则返回False）
        """
        is_dup, _ = self.is_duplicate(question)
        if is_dup:
            return False
        
        q_hash = self._compute_hash(question)
        
        with self._lock:
            self.questions[q_hash] = {
                'question': question,
                'timestamp': datetime.now().isoformat(),
                'count': 1
            }
            self.question_hashes.add(q_hash)
        
        # 异步保存
        threading.Thread(target=self._save_history, daemon=True).start()
        return True
    
    def add_questions(self, questions: List[str]) -> List[str]:
        """
        批量添加问题，返回实际添加成功的问题列表（去重后）
        
        Args:
            questions: 问题列表
        
        Returns:
            成功添加的问题列表
        """
        added = []
        for q in questions:
            if self.add_question(q):
                added.append(q)
        return added
    
    def filter_duplicates(self, questions: List[str]) -> List[str]:
        """
        过滤掉重复的问题
        
        Args:
            questions: 问题列表
        
        Returns:
            去重后的问题列表
        """
        unique = []
        seen_hashes = set()
        
        for q in questions:
            q_hash = self._compute_hash(q)
            
            # 检查是否在本次列表中已出现
            if q_hash in seen_hashes:
                continue
            
            # 检查是否在历史记录中
            is_dup, _ = self.is_duplicate(q)
            if not is_dup:
                unique.append(q)
                seen_hashes.add(q_hash)
        
        return unique
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                'total_questions': len(self.questions),
                'history_file': self.history_file,
                'similarity_threshold': self.similarity_threshold
            }
    
    def archive_old_questions(self, days: int = 30):
        """
        归档旧问题（将超过指定天数的问题移到归档文件）
        
        Args:
            days: 保留最近多少天的问题
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        archive_file = self.history_file.replace('.json', f'_archive_{datetime.now().strftime("%Y%m%d")}.json')
        
        with self._lock:
            to_archive = {}
            to_keep = {}
            
            for h, data in self.questions.items():
                try:
                    q_time = datetime.fromisoformat(data.get('timestamp', ''))
                    if q_time < cutoff_date:
                        to_archive[h] = data
                    else:
                        to_keep[h] = data
                except:
                    to_keep[h] = data
            
            if to_archive:
                # 保存归档
                try:
                    with open(archive_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'archived_at': datetime.now().isoformat(),
                            'questions': to_archive
                        }, f, ensure_ascii=False, indent=2)
                    print(f"已归档 {len(to_archive)} 条旧问题到 {archive_file}")
                except Exception as e:
                    print(f"归档失败: {e}")
                    return
                
                # 更新当前记录
                self.questions = to_keep
                self.question_hashes = set(to_keep.keys())
                self._save_history()


class ContinuousRunner:
    """持续运行管理器 - 支持长期运行和状态管理"""
    
    def __init__(self, history: QuestionHistory = None):
        """
        初始化持续运行管理器
        
        Args:
            history: 问题历史管理器实例
        """
        self.history = history or QuestionHistory()
        self.state_file = os.path.join(OUTPUT_DIR, "runner_state.json")
        self.running = False
        self.paused = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器（优雅退出）"""
        def signal_handler(signum, frame):
            print(f"\n收到信号 {signum}，正在优雅停止...")
            self.stop()
        
        # 捕获 SIGINT (Ctrl+C) 和 SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _load_state(self) -> dict:
        """加载运行状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'start_time': None,
            'total_runs': 0,
            'total_questions': 0,
            'last_run': None,
            'errors': 0
        }
    
    def _save_state(self, state: dict):
        """保存运行状态"""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存状态失败: {e}")
    
    def stop(self):
        """停止运行"""
        self.running = False
        self._stop_event.set()
        self._pause_event.set()  # 如果暂停中，也唤醒以便退出
    
    def pause(self):
        """暂停运行"""
        self.paused = True
        self._pause_event.clear()
        print("运行已暂停，发送 SIGCONT 或调用 resume() 继续")
    
    def resume(self):
        """恢复运行"""
        self.paused = False
        self._pause_event.set()
        print("运行已恢复")
    
    def wait_if_paused(self):
        """如果暂停则等待"""
        while self.paused and self.running:
            self._pause_event.wait(timeout=1)
    
    def run_loop(self, 
                 analyzer: 'LLMDiffAnalyzer',
                 models: List[ModelConfig],
                 question_count: int = 5,
                 interval_minutes: float = 30,
                 max_runs: int = None,
                 generator_model: str = "gpt-4",
                 generator_config: ModelConfig = None,
                 use_example_style: bool = True,
                 example_questions: List[str] = None,
                 on_complete: callable = None):
        """
        持续运行循环
        
        Args:
            analyzer: 分析器实例
            models: 模型配置列表
            question_count: 每次生成的问题数量
            interval_minutes: 运行间隔（分钟）
            max_runs: 最大运行次数（None表示无限）
            generator_model: 问题生成模型
            generator_config: 问题生成模型配置
            use_example_style: 是否使用示例问题风格
            example_questions: 自定义示例问题
            on_complete: 每次完成后的回调函数
        """
        self.running = True
        state = self._load_state()
        
        if state['start_time'] is None:
            state['start_time'] = datetime.now().isoformat()
        
        run_count = 0
        interval_seconds = interval_minutes * 60
        
        print(f"\n{'='*60}")
        print(f"持续运行模式启动")
        print(f"运行间隔: {interval_minutes} 分钟")
        print(f"每次问题数: {question_count}")
        print(f"最大运行次数: {'无限' if max_runs is None else max_runs}")
        print(f"历史问题数: {self.history.get_stats()['total_questions']}")
        print(f"{'='*60}\n")
        
        try:
            while self.running and (max_runs is None or run_count < max_runs):
                run_count += 1
                
                # 检查暂停
                self.wait_if_paused()
                if not self.running:
                    break
                
                print(f"\n{'─'*60}")
                print(f"第 {run_count} 次运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'─'*60}")
                
                try:
                    # 运行分析
                    results = analyzer.run_full_analysis(
                        topic=None,
                        models=models,
                        question_count=question_count,
                        generator_model=generator_model,
                        generator_config=generator_config,
                        use_example_style=use_example_style,
                        example_questions=example_questions
                    )
                    
                    # 记录问题到历史
                    actual_questions = [r.question for r in results]
                    new_questions = self.history.add_questions(actual_questions)
                    
                    # 保存结果
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    analyzer.save_results(f"analysis_results_{timestamp}.json")
                    analyzer.generate_report(f"analysis_report_{timestamp}.md")
                    
                    # 更新状态
                    state['total_runs'] += 1
                    state['total_questions'] += len(new_questions)
                    state['last_run'] = datetime.now().isoformat()
                    self._save_state(state)
                    
                    print(f"\n本次新增问题: {len(new_questions)}/{len(actual_questions)}")
                    print(f"累计运行次数: {state['total_runs']}")
                    print(f"累计问题数: {state['total_questions']}")
                    
                    # 回调
                    if on_complete:
                        on_complete(results, state)
                    
                except Exception as e:
                    print(f"运行出错: {e}")
                    state['errors'] += 1
                    self._save_state(state)
                
                # 等待下次运行
                if self.running and (max_runs is None or run_count < max_runs):
                    print(f"\n等待 {interval_minutes} 分钟后进行下次运行...")
                    print("(按 Ctrl+C 停止)")
                    
                    # 分段等待以支持及时响应停止信号
                    for _ in range(int(interval_seconds)):
                        if not self.running:
                            break
                        self.wait_if_paused()
                        if self._stop_event.wait(timeout=1):
                            break
        
        except KeyboardInterrupt:
            print("\n用户中断，正在停止...")
        
        finally:
            self.running = False
            self._save_state(state)
            print(f"\n{'='*60}")
            print(f"运行结束")
            print(f"总运行次数: {state['total_runs']}")
            print(f"总问题数: {state['total_questions']}")
            print(f"错误次数: {state['errors']}")
            print(f"{'='*60}")


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
                 analyzer_config: ModelConfig = None,
                 question_history: QuestionHistory = None):
        """
        初始化分析器
        
        Args:
            llm_client: LLM客户端实例
            analyzer_model: 分析器使用的模型名称（向后兼容）
            analyzer_config: 分析器模型的完整配置（可选，用于独立配置分析器）
            question_history: 问题历史管理器实例（可选，用于去重）
        """
        self.llm = llm_client
        self.analyzer_model = analyzer_model
        self.analyzer_config = analyzer_config
        self.question_history = question_history
        self.results: list[QuestionResult] = []
    
    def generate_questions(self, topic: str, count: int = 5, 
                          generator_model: str = "gpt-4",
                          generator_config: ModelConfig = None,
                          exclude_duplicates: bool = True,
                          max_attempts: int = 3) -> list[str]:
        """
        用AI生成问题
        
        Args:
            topic: 主题
            count: 生成数量
            generator_model: 生成模型
            generator_config: 生成模型配置
            exclude_duplicates: 是否排除重复问题
            max_attempts: 最大尝试次数（用于获取足够的不重复问题）
        
        Returns:
            问题列表
        """
        all_questions = []
        attempts = 0
        
        while len(all_questions) < count and attempts < max_attempts:
            attempts += 1
            remaining = count - len(all_questions)
            
            # 如果需要更多问题，提示生成更多
            generate_count = remaining + 5  # 多生成一些以防重复
            
            prompt = f"""请生成{generate_count}个关于"{topic}"的深度问题。
要求：
1. 问题应该有深度，需要分析和思考
2. 问题之间要有差异性，覆盖不同角度
3. 每个问题一行，不要编号
4. 不要生成已有类似问题

请直接输出问题，每行一个："""
            
            response = self.llm.chat(
                generator_model, 
                [{"role": "user", "content": prompt}],
                model_config=generator_config
            )
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            # 过滤掉可能的编号前缀
            questions = [re.sub(r'^\d+[\.\)、\s]+', '', q).strip() for q in questions]
            
            # 去重处理
            if exclude_duplicates and self.question_history:
                questions = self.question_history.filter_duplicates(questions)
            
            # 添加到结果
            for q in questions:
                if q not in all_questions:
                    all_questions.append(q)
                if len(all_questions) >= count:
                    break
        
        return all_questions[:count]
    
    def generate_similar_questions(self, count: int = 5,
                                   generator_model: str = "gpt-4",
                                   generator_config: ModelConfig = None,
                                   example_questions: list[str] = None,
                                   exclude_duplicates: bool = True,
                                   max_attempts: int = 3) -> list[str]:
        """
        根据示例问题生成类似风格的新问题
        
        Args:
            count: 生成问题数量
            generator_model: 生成模型名称
            generator_config: 生成模型配置
            example_questions: 自定义示例问题列表，为None则使用默认示例
            exclude_duplicates: 是否排除重复问题
            max_attempts: 最大尝试次数
        
        Returns:
            生成的新问题列表
        """
        examples = example_questions if example_questions else EXAMPLE_QUESTIONS
        
        all_questions = []
        attempts = 0
        
        while len(all_questions) < count and attempts < max_attempts:
            attempts += 1
            remaining = count - len(all_questions)
            
            # 随机选择一些示例作为参考
            import random
            sample_examples = random.sample(examples, min(15, len(examples)))
            
            # 如果需要更多问题，生成更多
            generate_count = remaining + 10  # 多生成一些以防重复
            
            prompt = f"""以下是{len(sample_examples)}个问题示例，请仔细分析这些问题的风格特点，然后生成{generate_count}个风格类似的新问题。

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

请生成{generate_count}个新问题，要求：
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
            
            # 去重处理
            if exclude_duplicates and self.question_history:
                questions = self.question_history.filter_duplicates(questions)
            
            # 添加到结果（确保不重复）
            for q in questions:
                if q not in all_questions:
                    all_questions.append(q)
                if len(all_questions) >= count:
                    break
        
        return all_questions[:count]
    
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
                # 使用友好名称用于显示
                display_name = getattr(model, 'display_name', model_name)
                try:
                    answer = self.llm.chat(
                        model.model_name, 
                        [{"role": "user", "content": question}],
                        model_config=model
                    )
                    responses.append(ModelResponse(model_name=display_name, response=answer))
                except Exception as e:
                    print(f"模型 {display_name} 回答失败: {e}")
                    responses.append(ModelResponse(model_name=display_name, response=f"[错误: {e}]"))
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
        # 将 tuple 键转换为字符串，以便 JSON 序列化
        tag_pairs_serializable = {f"{k[0]} & {k[1]}": v for k, v in tag_pairs.items()}
        
        correlation_data = {
            "tag_counts": dict(tag_counts),
            "tag_pairs": tag_pairs_serializable,
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
                         example_questions: list[str] = None,
                         exclude_duplicates: bool = True) -> list[QuestionResult]:
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
            exclude_duplicates: 是否排除重复问题
        
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
                example_questions=example_questions,
                exclude_duplicates=exclude_duplicates
            )
        else:
            questions = self.generate_questions(
                topic, question_count, 
                generator_model=generator_model,
                generator_config=generator_config,
                exclude_duplicates=exclude_duplicates
            )
        
        if not questions:
            print("警告: 没有生成新的不重复问题")
            return []
        
        print(f"生成了 {len(questions)} 个新问题（已去重）")
        
        # 将问题添加到历史记录
        if self.question_history:
            added = self.question_history.add_questions(questions)
            print(f"已记录 {len(added)} 个新问题到历史")
        
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
    # 持续运行相关参数
    parser.add_argument('--continuous', action='store_true',
                        help='启用持续运行模式')
    parser.add_argument('--interval', type=float, default=30,
                        help='持续运行间隔（分钟，默认: 30）')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='最大运行次数（不指定则无限运行）')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                        help='问题相似度阈值（0-1，默认: 0.85）')
    parser.add_argument('--history-file', type=str, default=None,
                        help='问题历史文件路径')
    parser.add_argument('--archive-days', type=int, default=365,
                        help='归档超过多少天的问题（默认: 365，即不归档）')
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
    continuous_config = config.get('continuous', {})
    analyzer_dict = config.get('analyzer', {})
    generator_dict = config.get('generator', {})
    models_list = config.get('models', [])
    
    # 命令行参数覆盖配置文件
    question_count = args.count if args.count is not None else analysis_config.get('question_count', 5)
    use_example_style = args.example_style if args.example_style is not None else analysis_config.get('use_example_style', True)
    
    # 持续运行配置
    continuous_mode = args.continuous or continuous_config.get('enabled', False)
    interval_minutes = args.interval if args.interval != 30 else continuous_config.get('interval_minutes', 30)
    max_runs = args.max_runs if args.max_runs is not None else continuous_config.get('max_runs')
    similarity_threshold = args.similarity_threshold if args.similarity_threshold != 0.85 else continuous_config.get('similarity_threshold', 0.85)
    
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
    
    # 创建问题历史管理器
    history_file = args.history_file or continuous_config.get('history_file')
    question_history = QuestionHistory(
        history_file=history_file,
        similarity_threshold=similarity_threshold
    )
    
    # 归档旧问题（如果需要）
    if args.archive_days < 365:
        question_history.archive_old_questions(days=args.archive_days)
    
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
        analyzer_config=analyzer_config,
        question_history=question_history
    )
    
    # 创建回答模型配置列表
    model_configs = []
    for m in models_list:
        if m.get('enabled', True):
            # 使用 model_name 作为实际调用的模型名称
            # name 只是友好名称，用于显示
            mc = ModelConfig(
                model_name=m.get('model_name', m.get('name', '')),
                api_type=m.get('api_type', 'openai'),
                api_key=m.get('api_key', ''),
                base_url=m.get('base_url', '')
            )
            # 保存友好名称用于显示
            mc.display_name = m.get('name', mc.model_name)
            model_configs.append(mc)
    
    if not model_configs:
        print("错误: 没有启用的回答模型，请检查配置文件")
        return
    
    # 运行分析
    print("=== 模型配置信息 ===")
    print(f"分析模型: {analyzer_config.model_name}")
    print(f"问题生成模型: {generator_config.model_name}")
    print(f"回答模型: {', '.join(getattr(m, 'display_name', m.model_name) for m in model_configs)}")
    print(f"问题数量: {question_count}")
    print(f"相似度阈值: {similarity_threshold}")
    print(f"历史问题数: {question_history.get_stats()['total_questions']}")
    if use_example_style:
        print(f"问题生成模式: 示例问题风格")
    else:
        print(f"问题生成模式: 传统主题模式 - {args.topic or '人工智能与人类未来的关系'}")
    
    if continuous_mode:
        print(f"\n持续运行模式: 启用")
        print(f"运行间隔: {interval_minutes} 分钟")
        print(f"最大运行次数: {'无限' if max_runs is None else max_runs}")
    
    print()
    
    if continuous_mode:
        # 持续运行模式
        runner = ContinuousRunner(history=question_history)
        runner.run_loop(
            analyzer=analyzer,
            models=model_configs,
            question_count=question_count,
            interval_minutes=interval_minutes,
            max_runs=max_runs,
            generator_model=generator_config.model_name,
            generator_config=generator_config,
            use_example_style=use_example_style
        )
    else:
        # 单次运行模式
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
