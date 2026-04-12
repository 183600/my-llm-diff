# LLM 差异分析器

一个用于对比分析不同大语言模型回答差异的 Python 工具。该工具能够自动生成问题、收集多模型回答、分析差异、打标签并分析标签相关性。

## 功能特性

- **自动生成问题**: 使用 LLM 根据指定主题生成深度问题
- **多模型对比**: 支持同时向多个模型发送相同问题并收集回答
- **差异分析**: 自动分析不同模型输出的差异，生成关键词概括
- **智能标签**: 为每个回答打上描述性标签
- **标签合并**: 自动识别并合并同义词或近义词标签
- **相关性分析**: 分析标签在不同回答之间的相关性和模式
- **结果导出**: 支持 JSON 和 Markdown 格式的结果导出

## 支持的 API

- **OpenAI API**: 支持 GPT-4, GPT-3.5-turbo 等模型
- **Ollama**: 支持本地部署的开源模型
- **自定义 API**: 支持兼容 OpenAI 格式的自定义 API

## 安装

### 克隆仓库

```bash
git clone https://github.com/your-username/llm-diff-analyzer.git
cd llm-diff-analyzer
```

### 安装依赖

```bash
pip install -r requirements.txt
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

## 环境变量

| 变量名 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | OpenAI API 基础 URL（可选） |

## 注意事项

1. 确保已配置正确的 API 密钥
2. 运行分析会产生 API 调用费用
3. 建议从小规模测试开始（少量问题和模型）
4. 大规模分析可能需要较长时间

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
