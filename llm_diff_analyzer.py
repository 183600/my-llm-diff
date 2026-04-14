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
    from openai import OpenAI, APITimeoutError, APIConnectionError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    APITimeoutError = Exception
    APIConnectionError = Exception

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
    timeout: float = 120.0  # API请求超时时间（秒）
    
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
                 on_complete: callable = None,
                 max_retries: int = 3,
                 retry_delay: int = 60,
                 error_cooldown: int = 300):
        """
        持续运行循环（支持无限运行）
        
        Args:
            analyzer: 分析器实例
            models: 模型配置列表
            question_count: 每次生成的问题数量
            interval_minutes: 运行间隔（分钟）
            max_runs: 最大运行次数（None表示无限运行）
            generator_model: 问题生成模型
            generator_config: 问题生成模型配置
            use_example_style: 是否使用示例问题风格
            example_questions: 自定义示例问题
            on_complete: 每次完成后的回调函数
            max_retries: 单次运行最大重试次数
            retry_delay: 重试间隔（秒）
            error_cooldown: 连续错误后的冷却时间（秒）
        """
        self.running = True
        state = self._load_state()
        
        if state['start_time'] is None:
            state['start_time'] = datetime.now().isoformat()
        
        run_count = 0
        interval_seconds = interval_minutes * 60
        consecutive_errors = 0  # 连续错误计数
        
        print(f"\n{'='*60}")
        print(f"持续运行模式启动（无限运行）")
        print(f"运行间隔: {interval_minutes} 分钟")
        print(f"每次问题数: {question_count}")
        print(f"最大运行次数: {'无限' if max_runs is None else max_runs}")
        print(f"最大重试次数: {max_retries}")
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
                
                # 带重试的运行
                run_success = False
                for retry in range(max_retries):
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
                        
                        run_success = True
                        consecutive_errors = 0  # 重置连续错误计数
                        break
                        
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        print(f"运行出错 ({error_type}): {error_msg}")
                        
                        # 判断是否是可重试的错误
                        is_retryable = any(keyword in error_msg.lower() for keyword in [
                            'timeout', 'rate limit', 'throttl', 'connection', 
                            'network', 'temporarily', 'unavailable', '503', '502', '429'
                        ])
                        
                        if is_retryable and retry < max_retries - 1:
                            wait_time = retry_delay * (retry + 1)  # 指数退避
                            print(f"可重试错误，{wait_time}秒后进行第 {retry + 2} 次尝试...")
                            for _ in range(wait_time):
                                if not self.running:
                                    break
                                time.sleep(1)
                        else:
                            state['errors'] += 1
                            self._save_state(state)
                
                # 处理运行失败
                if not run_success:
                    consecutive_errors += 1
                    print(f"本次运行失败，累计连续错误: {consecutive_errors}")
                    
                    # 连续错误过多时增加冷却时间
                    if consecutive_errors >= 3:
                        cooldown = error_cooldown * consecutive_errors
                        print(f"连续错误较多，进入冷却等待 {cooldown} 秒...")
                        for _ in range(cooldown):
                            if not self.running:
                                break
                            self.wait_if_paused()
                            time.sleep(1)
                        consecutive_errors = 0  # 重置，给新的机会
                
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
        
        except Exception as e:
            print(f"\n严重错误: {type(e).__name__}: {e}")
            print("程序将尝试继续运行...")
            # 不退出，而是尝试继续（如果还在运行）
            if self.running:
                self.run_loop(
                    analyzer, models, question_count, interval_minutes, max_runs,
                    generator_model, generator_config, use_example_style,
                    example_questions, on_complete, max_retries, retry_delay, error_cooldown
                )
        
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
            timeout = getattr(model_config, 'timeout', 120.0)
        else:
            api_type = self.api_type
            api_key = self.api_key
            base_url = self.base_url
            timeout = 120.0
        
        # 对于OpenAI类型，使用缓存的客户端
        openai_client = None
        if api_type == 'openai' and HAS_OPENAI:
            cache_key = (api_key, base_url, timeout)
            if cache_key not in self._clients_cache:
                self._clients_cache[cache_key] = OpenAI(
                    api_key=api_key, 
                    base_url=base_url,
                    timeout=timeout
                )
            openai_client = self._clients_cache[cache_key]
        
        return api_type, api_key, base_url, openai_client
    
    def chat(self, model: str, messages: list, model_config: ModelConfig = None, 
             max_retries: int = 3, **kwargs) -> str:
        """
        发送聊天请求
        
        Args:
            model: 模型名称
            messages: 消息列表
            model_config: 可选的模型配置，用于覆盖默认配置
            max_retries: 最大重试次数（针对超时和连接错误）
            **kwargs: 其他参数传递给API
        
        Returns:
            模型响应文本
        """
        api_type, api_key, base_url, openai_client = self._get_client(model_config)
        
        if api_type == 'openai' and openai_client:
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                    return response.choices[0].message.content
                except (APITimeoutError, APIConnectionError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 递增等待时间
                        print(f"API 请求超时/连接错误，{wait_time}秒后重试 (尝试 {attempt + 2}/{max_retries})...")
                        time.sleep(wait_time)
                    continue
            raise last_error or RuntimeError("API 请求失败")
        
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
        """为各个回答打标签，关注细节差异，使用标准化格式便于后续合并"""
        # 先构建所有回答的摘要，用于比较
        all_responses_summary = "\n\n".join([
            f"【{r.model_name}】: {r.response[:500]}..."
            for r in responses
        ])
        
        # 构建其他模型名称列表，用于对比
        other_models = [r.model_name for r in responses]
        
        for response in responses:
            prompt = f"""请为以下回答打上细粒度标签，重点描述这个回答与其他回答的具体差异细节。

问题：{question}

所有回答摘要（用于对比）：
{all_responses_summary}

当前需要打标签的回答：【{response.model_name}】:
{response.response}

---

**重要：标签格式规范（必须严格遵循）**

每个标签采用固定格式：`[动作类型]::[具体对象]`

**动作类型（固定词汇，便于后续合并）：**
- `提及` - 提及了某个概念/名词/术语
- `遗漏` - 遗漏了某个重要内容
- `补充` - 补充了额外信息
- `解释` - 详细解释了某个概念
- `强调` - 特别强调了某个观点
- `反对` - 明确反对某个观点
- `支持` - 明确支持某个观点
- `质疑` - 对某个观点提出疑问
- `列举` - 列举了具体案例/例子
- `给出` - 给出了具体数值/数据
- `引用` - 引用了具体来源/文献
- `对比` - 对比了具体对象/概念
- `聚焦` - 聚焦于某个方面/领域
- `使用` - 使用了具体方法/比喻/类比
- `结构` - 回答结构特点（如：结构::分层论述）
- `风格` - 表达风格特点（如：风格::学术严谨）
- `长度` - 回答长度特点（如：长度::详细）

**具体对象规则：**
1. 必须是具体的名词、概念、数值、案例名等
2. 不能是模糊的描述词（如"一些内容"、"某些观点"）
3. 多个对象用顿号分隔（如：提及::量子力学、相对论）

**示例标签：**
- 提及::量子力学、波函数坍缩
- 遗漏::实验数据验证
- 补充::历史背景介绍
- 解释::缸中之脑哲学思想实验
- 强调::主观体验无法被客观化
- 反对::物理主义还原论
- 列举::3个经典悖论案例
- 给出::具体概率数值50%
- 引用::笛卡尔《第一哲学沉思集》
- 对比::缸中之脑与玻尔兹曼大脑
- 聚焦::哲学认识论层面
- 使用::计算机模拟类比
- 结构::总分总结构
- 风格::通俗幽默
- 长度::中等篇幅

**打标签要求：**
1. 给出8-12个标签
2. 每个标签必须描述**具体差异**，而非笼统描述
3. 关注这个回答与其他回答的**不同之处**
4. 标签要细粒度，比如"提及::量子力学"比"提及::物理概念"更好
5. 按重要性排序，最重要的差异标签放前面

请直接输出标签，每行一个，格式为：动作类型::具体对象"""

            tags_str = self.llm.chat(
                self.analyzer_model, 
                [{"role": "user", "content": prompt}],
                model_config=model_config
            )
            # 解析标签，支持逗号分隔或换行分隔
            tags = []
            for line in tags_str.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 如果一行有多个逗号分隔的标签
                if ',' in line and '::' not in line:
                    tags.extend([t.strip() for t in line.split(',') if t.strip()])
                elif '，' in line and '::' not in line:
                    tags.extend([t.strip() for t in line.split('，') if t.strip()])
                else:
                    tags.append(line)
            
            response.tags = [t for t in tags if t.strip()]
    
    def merge_similar_tags(self, results: list[QuestionResult],
                          model_config: ModelConfig = None) -> dict:
        """合并同义词或近义词标签，使用LLM进行智能合并"""
        # 收集所有标签
        all_tags = []
        for result in results:
            for response in result.responses:
                all_tags.extend(response.tags)
        
        if not all_tags:
            return {}
        
        unique_tags = list(set(all_tags))
        
        # 首先进行本地预合并（动作类型层面的标准化）
        action_synonyms = {
            # 提及类
            '提到': '提及', '谈到': '提及', '涉及': '提及', '包含': '提及', '谈到过': '提及',
            # 遗漏类
            '缺少': '遗漏', '未提及': '遗漏', '忽略': '遗漏', '没提': '遗漏', '漏掉': '遗漏',
            # 补充类
            '增加': '补充', '添加': '补充', '额外提': '补充', '补充说明': '补充',
            # 解释类
            '阐述': '解释', '说明': '解释', '详解': '解释', '解释说明': '解释',
            # 强调类
            '突出': '强调', '着重': '强调', '特别提': '强调', '重点指出': '强调',
            # 反对类
            '否定': '反对', '批驳': '反对', '不同意': '反对',
            # 支持类
            '认同': '支持', '赞同': '支持', '同意': '支持',
            # 质疑类
            '怀疑': '质疑', '疑问': '质疑', '提出疑问': '质疑',
            # 列举类
            '罗列': '列举', '列出': '列举', '举例': '列举',
            # 给出类
            '提供': '给出', '呈现': '给出',
            # 引用类
            '引述': '引用', '援引': '引用', '参考': '引用',
            # 对比类
            '比较': '对比', '对照': '对比', '比较分析': '对比',
            # 聚焦类
            '关注': '聚焦', '集中于': '聚焦', '重点讨论': '聚焦',
            # 使用类
            '采用': '使用', '运用': '使用', '应用': '使用',
        }
        
        # 标准化动作类型
        standardized_tags = []
        for tag in unique_tags:
            if '::' in tag:
                action, obj = tag.split('::', 1)
                # 标准化动作
                std_action = action_synonyms.get(action, action)
                standardized_tags.append(f"{std_action}::{obj}")
            else:
                # 兼容旧格式，尝试解析
                standardized_tags.append(tag)
        
        unique_tags = list(set(standardized_tags))
        
        prompt = f"""请分析以下标签，将语义相近的标签合并为类别。

标签列表（格式：动作类型::具体对象）：
{chr(10).join(f'- {tag}' for tag in unique_tags)}

**合并规则：**

1. **动作类型相同的标签**：如果具体对象语义相近，可以合并
   - 例如："提及::量子力学" 和 "提及::量子理论" → 合并为 "提及::量子力学/量子理论"
   - 例如："解释::缸中之脑" 和 "解释::缸中之脑实验" → 合并为 "解释::缸中之脑"

2. **动作类型不同但语义相关**：根据语义判断是否合并
   - 例如："强调::主观体验" 和 "聚焦::主观体验层面" → 可合并为 "聚焦主观体验"
   - 例如："列举::3个案例" 和 "给出::具体案例" → 可合并为 "列举/给出案例"

3. **不合并的情况**：
   - 具体对象完全不同："提及::量子力学" 和 "提及::相对论" → 不合并
   - 动作语义相反："支持::某观点" 和 "反对::某观点" → 不合并
   - 数量差异明显："列举::3个案例" 和 "列举::10个案例" → 不合并

4. **类别命名规则**：
   - 使用标准化格式：`动作类型::对象摘要`
   - 对象摘要应简洁概括，如 "量子力学相关概念"、"具体案例和数据"
   - 如果合并了多个动作类型，用"/"分隔，如 "提及/解释::哲学概念"

**输出格式（JSON）：**
```json
{{
  "categories": [
    {{
      "category_name": "提及::量子力学相关概念",
      "tags": ["提及::量子力学", "提及::量子理论", "提及::波函数"],
      "description": "涉及量子力学概念的标签"
    }},
    {{
      "category_name": "列举::具体案例",
      "tags": ["列举::3个案例", "给出::具体案例"],
      "description": "列举或给出具体案例的标签"
    }}
  ]
}}
```

请确保：
1. 所有标签都必须被归类，不能遗漏
2. 每个标签只能属于一个类别
3. 类别描述简洁明了
4. 输出必须是有效的JSON格式"""

        merge_result = self.llm.chat(
            self.analyzer_model, 
            [{"role": "user", "content": prompt}],
            model_config=model_config
        )
        
        # 解析JSON结果
        merged = {}
        try:
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', merge_result)
            if json_match:
                data = json.loads(json_match.group())
                for cat in data.get('categories', []):
                    category_name = cat.get('category_name', '')
                    tags = cat.get('tags', [])
                    if category_name and tags:
                        merged[category_name] = tags
        except json.JSONDecodeError as e:
            print(f"解析合并结果失败: {e}")
            # 回退：简单按动作类型分组
            for tag in unique_tags:
                if '::' in tag:
                    action = tag.split('::')[0]
                    category = f"{action}类"
                    if category not in merged:
                        merged[category] = []
                    merged[category].append(tag)
        
        # 为每个结果添加合并后的标签映射
        for result in results:
            result.merged_tags = merged
            # 为每个回答创建标签到类别的映射
            for response in result.responses:
                tag_to_category = {}
                for tag in response.tags:
                    # 标准化标签
                    if '::' in tag:
                        action, obj = tag.split('::', 1)
                        std_action = action_synonyms.get(action, action)
                        std_tag = f"{std_action}::{obj}"
                    else:
                        std_tag = tag
                    
                    # 查找所属类别
                    for category, tags in merged.items():
                        if std_tag in tags or tag in tags:
                            tag_to_category[tag] = category
                            break
                
                # 存储映射到response
                if not hasattr(response, 'tag_categories'):
                    response.tag_categories = {}
                response.tag_categories = tag_to_category
        
        return merged
    
    def analyze_tag_correlation(self, results: list[QuestionResult],
                                model_config: ModelConfig = None) -> dict:
        """分析不同标签在不同回答之间的相关性，使用合并后的标签类别"""
        
        # 1. 收集原始标签统计
        tag_counts = defaultdict(int)
        tag_by_model = defaultdict(lambda: defaultdict(int))  # model -> tag -> count
        tag_by_question = defaultdict(lambda: defaultdict(list))  # question -> tag -> [models]
        
        # 2. 收集合并后的类别统计
        category_counts = defaultdict(int)
        category_by_model = defaultdict(lambda: defaultdict(int))  # model -> category -> count
        category_cooccurrence = defaultdict(lambda: defaultdict(int))  # category1 -> category2 -> count
        
        for result in results:
            question = result.question
            merged_tags = result.merged_tags
            
            for response in result.responses:
                model = response.model_name
                tags = response.tags
                
                # 原始标签统计
                for tag in tags:
                    tag_counts[tag] += 1
                    tag_by_model[model][tag] += 1
                    tag_by_question[question][tag].append(model)
                
                # 合并后的类别统计
                tag_categories = getattr(response, 'tag_categories', {})
                categories_in_response = set()
                
                for tag, category in tag_categories.items():
                    category_counts[category] += 1
                    category_by_model[model][category] += 1
                    categories_in_response.add(category)
                
                # 类别共现统计
                categories_list = list(categories_in_response)
                for i in range(len(categories_list)):
                    for j in range(i + 1, len(categories_list)):
                        cat1, cat2 = categories_list[i], categories_list[j]
                        # 双向记录
                        category_cooccurrence[cat1][cat2] += 1
                        category_cooccurrence[cat2][cat1] += 1
        
        # 3. 计算类别之间的关联强度（Jaccard相似度）
        category_associations = {}
        all_categories = list(category_counts.keys())
        
        for i, cat1 in enumerate(all_categories):
            for cat2 in all_categories[i+1:]:
                cooccur = category_cooccurrence[cat1].get(cat2, 0)
                if cooccur > 0:
                    # Jaccard相似度
                    union = category_counts[cat1] + category_counts[cat2] - cooccur
                    similarity = cooccur / union if union > 0 else 0
                    if similarity > 0.1:  # 只记录有一定关联的
                        category_associations[f"{cat1} ↔ {cat2}"] = {
                            "cooccurrence": cooccur,
                            "similarity": round(similarity, 3)
                        }
        
        # 4. 分析各模型的标签偏好特征
        model_profiles = {}
        for model, cats in category_by_model.items():
            total = sum(cats.values())
            if total > 0:
                # 找出该模型最显著的特征
                top_categories = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:5]
                model_profiles[model] = {
                    "total_tags": total,
                    "top_categories": [(cat, count, round(count/total, 3)) for cat, count in top_categories],
                    "unique_strength": []  # 该模型独特的特征
                }
        
        # 5. 找出各模型独特的标签特征（相比其他模型更常出现的）
        for model in model_profiles:
            model_cats = category_by_model[model]
            model_total = sum(model_cats.values())
            unique_strengths = []
            
            for cat, count in model_cats.items():
                model_ratio = count / model_total if model_total > 0 else 0
                
                # 其他模型的平均比例
                other_ratios = []
                for other_model, other_cats in category_by_model.items():
                    if other_model != model:
                        other_total = sum(other_cats.values())
                        if other_total > 0 and cat in other_cats:
                            other_ratios.append(other_cats[cat] / other_total)
                
                if other_ratios:
                    avg_other_ratio = sum(other_ratios) / len(other_ratios)
                    # 如果该模型的比例明显高于其他模型
                    if model_ratio > avg_other_ratio * 1.5:
                        unique_strengths.append({
                            "category": cat,
                            "model_ratio": round(model_ratio, 3),
                            "others_avg_ratio": round(avg_other_ratio, 3),
                            "strength": round(model_ratio / avg_other_ratio, 2) if avg_other_ratio > 0 else float('inf')
                        })
            
            model_profiles[model]["unique_strength"] = sorted(
                unique_strengths, key=lambda x: x["strength"], reverse=True
            )[:3]
        
        # 6. 构建LLM分析的输入数据
        analysis_data = {
            "category_counts": dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)),
            "category_associations": dict(sorted(category_associations.items(), 
                                                key=lambda x: x[1]["similarity"], reverse=True)[:10]),
            "model_profiles": model_profiles,
            "total_questions": len(results),
            "total_responses": sum(len(r.responses) for r in results)
        }
        
        # 7. 用LLM进行深度解读
        prompt = f"""请深入分析以下标签相关性数据，找出有意义的模式和洞察。

**数据概览：**
- 总问题数：{analysis_data['total_questions']}
- 总回答数：{analysis_data['total_responses']}
- 标签类别数：{len(analysis_data['category_counts'])}

**标签类别出现频次（前15）：**
{json.dumps(list(analysis_data['category_counts'].items())[:15], ensure_ascii=False, indent=2)}

**标签类别关联度（前10组）：**
{json.dumps(analysis_data['category_associations'], ensure_ascii=False, indent=2)}

**各模型标签特征画像：**
{json.dumps(analysis_data['model_profiles'], ensure_ascii=False, indent=2)}

---

**请分析以下方面：**

1. **标签分布特征**
   - 哪些类别的标签最常见？反映了什么？
   - 有没有明显缺失的类别？

2. **类别关联模式**
   - 哪些标签类别经常一起出现？说明了什么？
   - 有没有意外的关联？如何解释？

3. **模型差异分析**
   - 各模型的标签特征有何不同？
   - 哪个模型最独特？独特在哪里？
   - 哪些模型比较相似？

4. **回答质量洞察**
   - 基于标签分布，哪个模型的回答更全面？
   - 哪个模型更注重细节？

5. **潜在改进建议**
   - 标签体系是否需要调整？
   - 有没有遗漏的重要维度？

**输出格式（JSON）：**
```json
{{
  "distribution_insights": "标签分布特征的洞察...",
  "association_patterns": "类别关联模式的发现...",
  "model_differences": {{
    "模型A": "特征描述...",
    "模型B": "特征描述..."
  }},
  "quality_assessment": "回答质量评估...",
  "recommendations": ["建议1", "建议2", ...]
}}
```"""

        analysis = self.llm.chat(
            self.analyzer_model, 
            [{"role": "user", "content": prompt}],
            model_config=model_config
        )
        
        # 8. 解析LLM分析结果
        try:
            json_match = re.search(r'\{[\s\S]*\}', analysis)
            if json_match:
                llm_insights = json.loads(json_match.group())
            else:
                llm_insights = {"raw_analysis": analysis}
        except json.JSONDecodeError:
            llm_insights = {"raw_analysis": analysis}
        
        # 9. 构建最终结果
        correlation_result = {
            "statistics": {
                "tag_counts": dict(tag_counts),
                "category_counts": dict(category_counts),
                "category_by_model": {k: dict(v) for k, v in category_by_model.items()},
            },
            "associations": {
                "category_cooccurrence": {k: dict(v) for k, v in category_cooccurrence.items()},
                "category_similarity": category_associations,
            },
            "model_profiles": model_profiles,
            "insights": llm_insights,
            "raw_analysis": analysis
        }
        
        # 更新结果
        for result in results:
            result.tag_correlation = correlation_result
        
        return correlation_result
    
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
        
        # 4. 打标签（细粒度、细节导向）
        print("\n[4/6] 为回答打标签（细粒度、细节导向）...")
        total_tags = 0
        for result in self.results:
            self.tag_responses(
                result.question, result.responses,
                model_config=self.analyzer_config
            )
            for r in result.responses:
                total_tags += len(r.tags)
                print(f"  {r.model_name} 标签({len(r.tags)}个): {', '.join(r.tags[:3])}...")
        print(f"  共生成 {total_tags} 个细粒度标签")
        
        # 5. 合并同义词/近义词标签
        print("\n[5/6] 合并同义词/近义词标签...")
        merged = self.merge_similar_tags(self.results, model_config=self.analyzer_config)
        print(f"  原始标签类别: {len(set(tag for r in self.results for resp in r.responses for tag in resp.tags))} 个")
        print(f"  合并后类别: {len(merged)} 个")
        print(f"  压缩率: {round((1 - len(merged)/max(1, len(set(tag for r in self.results for resp in r.responses for tag in resp.tags))))*100, 1)}%")
        
        # 6. 分析标签相关性（基于合并后的类别）
        print("\n[6/6] 分析标签相关性（基于合并后类别）...")
        correlation = self.analyze_tag_correlation(self.results, model_config=self.analyzer_config)
        
        # 输出关键洞察
        if 'model_profiles' in correlation:
            print("\n  模型特征画像:")
            for model, profile in correlation['model_profiles'].items():
                unique = profile.get('unique_strength', [])
                if unique:
                    top_unique = unique[0] if unique else {}
                    print(f"    {model}: 最独特特征 - {top_unique.get('category', 'N/A')} (强度: {top_unique.get('strength', 'N/A')})")
        
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
        
        # 问题详情
        for i, result in enumerate(self.results, 1):
            report.append(f"## 问题 {i}\n\n")
            report.append(f"**{result.question}**\n\n")
            
            report.append("### 各模型回答\n\n")
            for response in result.responses:
                report.append(f"#### {response.model_name}\n\n")
                report.append(f"**关键词**: {', '.join(response.diff_keywords)}\n\n")
                
                # 显示原始标签和合并后的类别
                report.append(f"**细粒度标签** ({len(response.tags)}个):\n")
                for tag in response.tags[:8]:  # 显示前8个
                    report.append(f"- {tag}\n")
                if len(response.tags) > 8:
                    report.append(f"- ... 等{len(response.tags)}个标签\n")
                report.append("\n")
                
                # 显示标签对应的类别
                if hasattr(response, 'tag_categories') and response.tag_categories:
                    report.append(f"**标签类别映射**:\n")
                    for tag, category in list(response.tag_categories.items())[:5]:
                        report.append(f"- `{tag}` → `{category}`\n")
                    report.append("\n")
                
                report.append(f"**回答摘要**:\n\n{response.response[:500]}...\n\n")
            
            report.append(f"### 差异摘要\n\n{result.diff_summary}\n\n")
            report.append("---\n\n")
        
        # 标签合并结果
        if self.results and self.results[0].merged_tags:
            report.append("## 标签合并结果\n\n")
            merged = self.results[0].merged_tags
            report.append(f"合并为 **{len(merged)}** 个标签类别:\n\n")
            for category, tags in sorted(merged.items()):
                report.append(f"### {category}\n")
                report.append(f"包含标签: {', '.join(tags[:5])}")
                if len(tags) > 5:
                    report.append(f" ... 等{len(tags)}个")
                report.append("\n\n")
        
        # 标签相关性分析
        if self.results and self.results[0].tag_correlation:
            report.append("## 标签相关性分析\n\n")
            correlation = self.results[0].tag_correlation
            
            # 模型特征画像
            if 'model_profiles' in correlation:
                report.append("### 模型特征画像\n\n")
                for model, profile in correlation['model_profiles'].items():
                    report.append(f"#### {model}\n\n")
                    report.append(f"- 总标签数: {profile.get('total_tags', 0)}\n")
                    
                    top_cats = profile.get('top_categories', [])
                    if top_cats:
                        report.append(f"- 主要特征: {', '.join([f'{cat}({ratio:.0%})' for cat, _, ratio in top_cats[:3]])}\n")
                    
                    unique = profile.get('unique_strength', [])
                    if unique:
                        report.append(f"- 独特优势: ")
                        for u in unique[:2]:
                            report.append(f"{u['category']}(强度{u['strength']:.1f}x) ")
                        report.append("\n")
                    report.append("\n")
            
            # 类别关联
            if 'associations' in correlation and correlation['associations'].get('category_similarity'):
                report.append("### 类别关联分析\n\n")
                associations = correlation['associations']['category_similarity']
                for pair, data in list(associations.items())[:10]:
                    report.append(f"- **{pair}**: 共现{data['cooccurrence']}次, 相似度{data['similarity']:.2f}\n")
                report.append("\n")
            
            # LLM洞察
            if 'insights' in correlation:
                insights = correlation['insights']
                report.append("### 深度洞察\n\n")
                
                if 'distribution_insights' in insights:
                    report.append(f"**分布特征**: {insights['distribution_insights']}\n\n")
                if 'association_patterns' in insights:
                    report.append(f"**关联模式**: {insights['association_patterns']}\n\n")
                if 'model_differences' in insights:
                    report.append("**模型差异**:\n")
                    for model, diff in insights['model_differences'].items():
                        report.append(f"- {model}: {diff}\n")
                    report.append("\n")
                if 'quality_assessment' in insights:
                    report.append(f"**质量评估**: {insights['quality_assessment']}\n\n")
                if 'recommendations' in insights:
                    report.append("**改进建议**:\n")
                    for rec in insights['recommendations']:
                        report.append(f"- {rec}\n")
                    report.append("\n")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"报告已保存到: {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"报告已保存到: {filepath}")


def create_model_config_from_dict(model_dict: dict) -> ModelConfig:
    """从配置字典创建 ModelConfig"""
    return ModelConfig(
        model_name=model_dict.get('model_name', ''),
        api_type=model_dict.get('api_type', 'openai'),
        api_key=model_dict.get('api_key', ''),
        base_url=model_dict.get('base_url', ''),
        timeout=float(model_dict.get('timeout', 120.0))
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
                base_url=m.get('base_url', ''),
                timeout=float(m.get('timeout', 120.0))
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
