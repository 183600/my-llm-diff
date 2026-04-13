# LLM 差异分析器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个用于对比分析不同大语言模型回答差异的 Python 工具。该工具能够自动生成问题、收集多模型回答、分析差异、打标签并分析标签相关性。

## 功能特性

- **自动生成问题**: 使用 LLM 根据指定主题生成深度问题，支持示例问题风格
- **多模型对比**: 支持同时向多个模型发送相同问题并收集回答
- **差异分析**: 自动分析不同模型输出的差异，生成关键词概括
- **智能标签**: 为每个回答打上描述性标签
- **标签合并**: 自动识别并合并同义词或近义词标签
- **相关性分析**: 分析标签在不同回答之间的相关性和模式
- **结果导出**: 支持 JSON 和 Markdown 格式的结果导出

## 支持的 API / 模型提供商

本项目支持所有兼容 OpenAI API 格式的服务：

| 提供商 | 说明 | 示例模型 |
|--------|------|----------|
| **OpenAI** | 官方 OpenAI API | GPT-4, GPT-3.5-turbo |
| **NVIDIA NIM** | NVIDIA 模型推理服务 | Kimi, Llama, Mistral |
| **Modal** | Modal 云端推理 | GLM-5 |
| **iFlow CN** | 国内 API 服务 | DeepSeek-V3 |
| **iFlow Rome** | iFlow Rome 模型 | iflow-rome |
| **ZenMux** | ZenMux API 服务 | GLM-4.7 |
| **Routin** | Routin API 服务 | Kimi-K2.5 |
| **Ollama** | 本地部署模型 | Llama2, Mistral |
| **自定义 API** | 任何兼容 OpenAI 格式的 API | - |

## 安装

### 克隆仓库

```bash
git clone https://github.com/183600/my-llm-diff.git
cd my-llm-diff
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

### 配置方式

项目支持三种配置方式，按优先级从高到低：

1. **环境变量** - 可以直接设置环境变量，或在 `.env` 文件中配置
2. **配置文件** - 使用 `config.yaml` 进行集中配置
3. **命令行参数** - 部分参数可通过命令行指定

### 快速配置

1. 复制示例配置文件：
```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API 密钥：
```bash
# 编辑 .env 文件
ROUTIN_API_KEY=your-routin-api-key-here
NVIDIA_API_KEY_1=your-nvidia-api-key-here
# ... 其他密钥
```

3. 根据需要修改 `config.yaml` 调整模型配置。

### 配置文件说明 (config.yaml)

```yaml
# 分析器配置 - 用于分析差异、打标签等
analyzer:
  model_name: "gpt-4"
  api_type: "openai"
  api_key: "${OPENAI_API_KEY}"  # 从环境变量读取
  base_url: "https://api.openai.com/v1"

# 问题生成器配置
generator:
  model_name: "gpt-4"
  api_type: "openai"
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"

# 用于生成回答的模型列表
models:
  - name: "GPT-4"
    model_name: "gpt-4"
    api_type: "openai"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    enabled: true

  - name: "GPT-3.5"
    model_name: "gpt-3.5-turbo"
    api_type: "openai"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    enabled: true

# 分析参数
analysis:
  question_count: 5
  use_example_style: true
```

### 环境变量说明

| 变量名 | 说明 |
|--------|------|
| `LLM_DIFF_CONFIG` | 自定义配置文件路径 |
| `OPENAI_API_KEY` | OpenAI API 密钥（后备） |
| `OPENAI_BASE_URL` | OpenAI API 基础 URL（后备） |
| `ROUTIN_API_KEY` | Routin API 密钥（用于分析和问题生成） |
| `NVIDIA_API_KEY_1` | NVIDIA NIM API 密钥 |
| `NVIDIA_API_KEY_2` | NVIDIA NIM API 密钥（备用） |
| `MODAL_API_KEY_1` | Modal API 密钥 |
| `MODAL_API_KEY_2` | Modal API 密钥（备用） |
| `IFLOW_CN_API_KEY` | iFlow CN API 密钥 |
| `IFLOW_ROME_API_KEY` | iFlow Rome API 密钥 |
| `ZENMUX_API_KEY` | ZenMux API 密钥 |

### 在配置中使用环境变量

在 `config.yaml` 中可以使用 `${VAR_NAME}` 语法引用环境变量：

```yaml
analyzer:
  api_key: "${MY_API_KEY}"  # 将被替换为环境变量 MY_API_KEY 的值
```

## 快速开始

### 基本使用

```python
from llm_diff_analyzer import LLMClient, LLMDiffAnalyzer
import os

# 配置客户端
config = {
    'api_type': 'openai',
    'api_key': os.getenv('OPENAI_API_KEY'),
    'base_url': 'https://api.openai.com/v1'
}

# 创建分析器
client = LLMClient(config)
analyzer = LLMDiffAnalyzer(client, analyzer_model="gpt-4")

# 运行分析
results = analyzer.run_full_analysis(
    topic="人工智能伦理",
    models=["gpt-4", "gpt-3.5-turbo"],
    question_count=3
)

# 保存结果
analyzer.save_results("results.json")
analyzer.generate_report("report.md")
```

### 使用 Ollama

```python
config = {
    'api_type': 'ollama',
    'base_url': 'http://localhost:11434'
}

client = LLMClient(config)
analyzer = LLMDiffAnalyzer(client, analyzer_model="llama2")

results = analyzer.run_full_analysis(
    topic="机器学习",
    models=["llama2", "mistral"],
    question_count=5
)
```

### 使用自定义 API

```python
config = {
    'api_type': 'custom',
    'api_key': 'your-api-key',
    'base_url': 'https://your-api-endpoint.com/v1'
}

client = LLMClient(config)
```

## 核心流程

```
┌─────────────────┐
│   生成问题       │  根据主题用 LLM 生成问题
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   收集回答       │  向多个模型发送问题并收集回答
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   分析差异       │  用 LLM 对比不同回答，提取关键词
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   打标签         │  为每个回答打上描述性标签
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   合并标签       │  合并同义词和近义词标签
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   相关性分析     │  分析标签之间的相关性
└─────────────────┘
```

## 输出示例

### JSON 结果格式

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "results": [
    {
      "question": "人工智能在医疗领域的应用有哪些伦理挑战?",
      "diff_summary": "GPT-4 更注重深度分析，GPT-3.5 更简洁实用...",
      "merged_tags": {
        "详细": ["详尽", "深入", "全面"],
        "简洁": ["简明", "精炼", "简要"]
      },
      "responses": [
        {
          "model_name": "gpt-4",
          "response": "...",
          "tags": ["详细", "学术化", "有深度"],
          "diff_keywords": ["深入分析", "多角度", "引用研究"]
        }
      ]
    }
  ]
}
```

### Markdown 报告

工具会生成包含以下内容的 Markdown 报告：

- 每个问题的详细分析
- 各模型的回答摘要
- 差异分析和关键词
- 标签相关性分析

## API 说明

### LLMClient

LLM 客户端类，用于与不同的 LLM API 交互。

```python
client = LLMClient(config)
response = client.chat("gpt-4", [{"role": "user", "content": "Hello"}])
```

### LLMDiffAnalyzer

主分析器类，提供完整的分析流程。

| 方法 | 说明 |
|------|------|
| `generate_questions(topic, count)` | 生成指定数量的问题 |
| `get_responses(question, models)` | 收集多个模型的回答 |
| `analyze_differences(question, responses)` | 分析回答差异 |
| `tag_responses(question, responses)` | 为回答打标签 |
| `merge_similar_tags(results)` | 合并相似标签 |
| `analyze_tag_correlation(results)` | 分析标签相关性 |
| `run_full_analysis(topic, models, question_count)` | 运行完整分析 |
| `save_results(filepath)` | 保存 JSON 结果 |
| `generate_report(filepath)` | 生成 Markdown 报告 |

## 命令行使用

```bash
# 使用默认配置运行
python llm_diff_analyzer.py

# 指定配置文件
python llm_diff_analyzer.py --config /path/to/config.yaml

# 列出配置中的模型
python llm_diff_analyzer.py --list-models

# 指定问题数量
python llm_diff_analyzer.py --count 10

# 使用传统主题模式
python llm_diff_analyzer.py --no-example-style --topic "人工智能伦理"
```

| 参数 | 说明 |
|------|------|
| `--config` | 指定配置文件路径 |
| `--topic` | 分析主题（不指定则使用示例问题风格） |
| `--count` | 生成问题数量 |
| `--example-style` | 使用示例问题风格生成问题（默认） |
| `--no-example-style` | 不使用示例问题风格 |
| `--list-models` | 列出配置中的模型并退出 |

## 注意事项

1. 确保已配置正确的 API 密钥
2. 运行分析会产生 API 调用费用
3. 建议从小规模测试开始（少量问题和模型）
4. 大规模分析可能需要较长时间

## 故障排除

### 常见问题

**Q: 运行时提示 "需要安装 PyYAML"**

A: 运行 `pip install pyyaml` 安装依赖。

**Q: 运行时提示 "未找到配置文件"**

A: 确保 `config.yaml` 文件存在于项目目录，或通过 `--config` 参数指定配置文件路径。

**Q: 某个模型回答失败**

A: 检查以下几点：
- API 密钥是否正确配置
- API 端点 URL 是否正确
- 模型名称是否正确
- 网络连接是否正常

**Q: 环境变量没有被正确读取**

A: 确保 `.env` 文件位于项目根目录，且格式正确（每行格式为 `KEY=VALUE`，无空格）。

**Q: 如何添加新的模型？**

A: 在 `config.yaml` 的 `models` 列表中添加新的模型配置：

```yaml
- name: "新模型名称"
  model_name: "模型标识"
  api_type: "openai"
  api_key: "${YOUR_API_KEY}"
  base_url: "https://api.example.com/v1"
  enabled: true
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
