# 样例文档 - Modular RAG 快速开始指南

## 1. 项目简介

Modular RAG 是一个模块化、可插拔的检索增强生成（RAG）系统。

## 2. 核心特性

- **模块化设计**：每个组件都可以独立替换
- **多供应商支持**：OpenAI、Azure、Ollama 等
- **混合检索**：Dense + Sparse + RRF 融合
- **可观测性**：完整的 Trace 日志和 Dashboard

## 3. 快速开始

### 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 配置 API Key

```bash
export OPENAI_API_KEY="your-api-key"
```

### 运行主程序

```bash
python main.py
```

## 4. 系统架构

系统分为以下几个主要层：

- **MCP Server 层**：处理协议通信
- **Core 层**：核心业务逻辑
- **Ingestion Pipeline**：数据摄取
- **Libs 层**：可插拔组件
- **Observability 层**：可观测性

## 5. 下一步

详见 README.md 了解更多信息。
