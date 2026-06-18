# Modular RAG System - 模块化RAG系统

A modular, pluggable RAG (Retrieval-Augmented Generation) system with MCP (Model Context Protocol) server support. The system provides a complete end-to-end solution for building, managing, and querying knowledge bases with full observability.

**核心特性**：
- 🔌 **可插拔架构**：LLM、嵌入、向量存储、重排、评估器等组件均可灵活替换
- 🌍 **多源支持**：OpenAI、Azure OpenAI、Ollama、DeepSeek、Qwen 等
- 🤖 **MCP服务器**：原生支持 GitHub Copilot 和 Claude Desktop
- 🔍 **混合检索**：稠密+稀疏检索，RRF融合算法
- 🖼️ **多模态**：图像理解与自动标题生成
- 📊 **完整可观测性**：追踪收集、JSON Lines 日志、Web Dashboard
- ⚡ **零外部依赖**：本地优先，可离线运行

---

## 📋 目录

1. [快速开始](#快速开始)
2. [配置说明](#配置说明)
3. [运行示例](#运行示例)
4. [MCP 配置](#mcp-配置)
5. [Dashboard 使用指南](#dashboard-使用指南)
6. [运行测试](#运行测试)
7. [常见问题](#常见问题)
8. [项目结构](#项目结构)

---

## 🚀 快速开始

### 1. 环境准备

#### 克隆仓库
```bash
git clone https://github.com/Mozinapig/Modular-RAG.git
cd Modular-RAG
```

#### 创建虚拟环境
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

#### 安装依赖
```bash
# 基础依赖
pip install -e .

# 开发依赖（可选）
pip install -e ".[dev]"
```

### 2. 配置 API Key

#### 创建本地配置文件
```bash
cp config/settings.yaml config/settings.local.yaml
```

#### 编辑配置文件
打开 `config/settings.local.yaml`，配置以下内容：

**使用 OpenAI 示例**：
```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: sk-xxxxxxxxxxxxx
  temperature: 0.7

embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: sk-xxxxxxxxxxxxx
```

**使用 Azure OpenAI 示例**：
```yaml
llm:
  provider: azure
  model: gpt-4o-mini
  api_key: your-azure-key
  azure_endpoint: https://your-resource.openai.azure.com/
  azure_deployment: your-deployment-name
```

**使用本地模型 (Ollama)**：
```yaml
llm:
  provider: ollama
  model: llama2
  base_url: http://localhost:11434
```

#### 设置环境变量（可选）
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxx"

# Windows CMD
set OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# macOS / Linux
export OPENAI_API_KEY="sk-xxxxxxxxxxxxx"
```

### 3. 首次摄取（Ingestion）

#### 准备示例文档
系统已包含示例文档在 `tests/fixtures/sample_documents/` 目录。

#### 运行摄取脚本
```bash
python scripts/ingest.py \
  --path tests/fixtures/sample_documents/ \
  --collection default
```

**参数说明**：
- `--path`: 文档所在目录路径
- `--collection`: 集合名称（可选，默认为 'default'）

**输出示例**：
```
Loading documents from: tests/fixtures/sample_documents/
Processing 3 documents...
✓ Document 1: config_guide.pdf (12 chunks)
✓ Document 2: api_reference.pdf (18 chunks)
✓ Document 3: tutorial.pdf (15 chunks)
Ingestion complete: 45 chunks indexed
```

---

## ⚙️ 配置说明

### settings.yaml 字段详解

#### 顶级配置
```yaml
# LLM 提供商配置（用于生成答案）
llm:
  provider: openai          # 提供商：openai / azure / ollama / deepseek / qwen
  model: gpt-4o-mini       # 模型名称
  api_key: ${OPENAI_API_KEY}  # API Key（支持环境变量）
  temperature: 0.7         # 生成温度 (0-2)
  max_tokens: null         # 最大输出 token 数

# 嵌入模型配置（用于向量化）
embedding:
  provider: openai         # 提供商
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
  dimensions: 1536         # 嵌入向量维度

# 向量存储配置
vector_store:
  backend: chroma          # 后端：chroma / pinecone / weaviate
  persist_path: ./data/db/chroma

# 检索配置
retrieval:
  sparse_backend: bm25     # 稀疏检索：bm25 / elasticsearch
  fusion_algorithm: rrf    # 融合算法：rrf / weighted
  top_k_dense: 20         # 稠密检索返回数
  top_k_sparse: 20        # 稀疏检索返回数
  top_k_final: 10         # 最终返回数

# 重排配置
rerank:
  backend: cross_encoder   # 后端：cross_encoder / bge-reranker
  model: bge-reranker-v2-m3
  top_m: 30               # 重排前的候选数

# 可观测性配置
observability:
  enabled: true
  log_file: ./logs/traces.jsonl
  log_level: INFO
  detail_level: standard  # standard / detailed / minimal
```

#### 扩展配置（extra 字段）
```yaml
extra:
  # 文本分割器配置
  splitter:
    provider: recursive
    chunk_size: 1024
    chunk_overlap: 100

  # 评估配置
  evaluation:
    backends: [custom]
    golden_test_set: ./tests/fixtures/golden_test_set.json

  # Dashboard 配置
  dashboard:
    enabled: true
    port: 8501
    traces_dir: ./logs
    auto_refresh: true
    refresh_interval: 5

  # 摄取配置
  ingestion:
    chunk_refiner:
      use_llm: false
    metadata_enricher:
      use_llm: false
    image_captioner:
      enabled: false
```

---

## 🔍 运行示例

### 1. 在线查询（通过脚本）

```bash
python scripts/query.py \
  --query "如何配置 Azure OpenAI？" \
  --top_k 5 \
  --verbose
```

**参数说明**：
- `--query`: 查询文本（必需）
- `--top_k`: 返回结果数（默认 5）
- `--verbose`: 详细输出（可选）

**输出示例**：
```
Query: 如何配置 Azure OpenAI？

Results:
1. [config_guide.pdf] Azure OpenAI 配置步骤
   Score: 0.92
   Content: 首先在 Azure Portal 上创建 OpenAI 资源...
   
2. [api_reference.pdf] 认证方式
   Score: 0.85
   Content: 使用 API Key 或托管身份进行认证...
```

### 2. MCP Server 交互

#### 启动 MCP Server
```bash
python -m src.mcp_server
```

Server 会在 stdin/stdout 上监听 JSON-RPC 请求。

#### 客户端调用示例
```python
import json
import subprocess

# 启动服务器
server = subprocess.Popen(
    ["python", "-m", "src.mcp_server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1
)

# 发送请求
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "query_knowledge_hub",
        "arguments": {
            "query": "Azure OpenAI 配置",
            "top_k": 5
        }
    }
}

server.stdin.write(json.dumps(request) + "\n")
response = json.loads(server.stdout.readline())

print(response["result"])
```

### 3. 评估系统性能

```bash
python scripts/evaluate.py \
  --test_set tests/fixtures/golden_test_set.json \
  --evaluator ragas
```

**输出示例**：
```
Evaluating with 10 test cases...

Results:
- Hit Rate: 0.85 (85% of queries found relevant chunks)
- MRR: 0.72 (平均排名倒数)
- Precision@5: 0.90

Per-query results:
Query 1: "Azure 配置" - Hit Rate: 1.0, MRR: 1.0 ✓
Query 2: "API 认证" - Hit Rate: 0.8, MRR: 0.67 ✓
...
```

---

## 🔐 MCP 配置

### GitHub Copilot 配置

#### 1. 创建 mcp.json
在项目根目录创建 `mcp.json`：
```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/Modular-RAG"
    }
  }
}
```

#### 2. 配置 GitHub Copilot
在 VSCode 中：
1. 打开命令面板：`Ctrl+Shift+P`
2. 搜索 "Copilot: Edit Copilot settings"
3. 找到 MCP 配置部分，添加上述 mcp.json 内容

#### 3. 在 Copilot 中使用
```
@knowledge-hub 如何配置 Azure OpenAI？
```

---

### Claude Desktop 配置

#### 1. 编辑 claude_desktop_config.json
```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": [
        "-m",
        "src.mcp_server"
      ],
      "cwd": "/path/to/Modular-RAG",
      "env": {
        "PYTHONPATH": "/path/to/Modular-RAG"
      }
    }
  }
}
```

**配置位置**：
- macOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

#### 2. 重启 Claude Desktop
关闭并重新启动 Claude Desktop 应用。

#### 3. 使用工具
现在可以在 Claude 中使用 `query_knowledge_hub` 工具：
```
我想了解如何在 Azure 上配置 OpenAI。
```

Claude 会自动调用 MCP 工具进行知识库查询。

---

## 📊 Dashboard 使用指南

### 启动 Dashboard

```bash
streamlit run src/observability/dashboard/app.py
```

Dashboard 会在 `http://localhost:8501` 打开。

### Dashboard 页面说明

#### 1. 🏠 系统概览页 (System Overview)
显示系统当前配置和统计信息：
- LLM 提供商和模型
- 嵌入模型配置
- 向量存储后端
- 检索配置
- 数据统计（文档数、chunk 数）

#### 2. 📚 数据浏览器 (Data Browser)
浏览和搜索已摄取的文档：
- 按集合筛选
- 按文档名称搜索
- 查看文档元数据
- 预览文档内容

#### 3. 📤 摄取管理器 (Ingestion Manager)
上传和管理文档：
- 上传文件（PDF、Word、TXT 等）
- 实时查看摄取进度
- 删除已摄取的文档
- 查看摄取历史

#### 4. 🔍 摄取追踪页 (Ingestion Traces)
查看摄取过程的详细追踪：
- 摄取时间线
- 各阶段耗时
- 错误日志
- 性能指标

#### 5. ❓ 查询追踪页 (Query Traces)
分析查询过程：
- 查询响应时间
- 检索得分分布
- 重排效果对比
- 相关性分析

#### 6. 📊 评估面板 (Evaluation Panel)
评估和改进系统性能：
- 选择评估器（Ragas、Custom 等）
- 运行评估任务
- 查看 Hit Rate、MRR 等指标
- 对比历史评估结果

---

## 🧪 运行测试

### 快速检查

```bash
# 导入检查
python -m compileall src

# 冒烟测试
pytest tests/unit/test_smoke_imports.py -v
```

### 单元测试

```bash
# 运行所有单元测试
pytest tests/unit/ -v

# 运行特定测试文件
pytest tests/unit/test_settings.py -v

# 运行特定测试用例
pytest tests/unit/test_settings.py::test_load_settings -v

# 查看覆盖率
pytest tests/unit/ --cov=src --cov-report=html
```

### 集成测试

```bash
# 运行集成测试
pytest tests/integration/ -v

# 运行特定集成测试
pytest tests/integration/test_hybrid_search.py -v
```

### E2E 测试

```bash
# 运行所有 E2E 测试
pytest tests/e2e/ -v

# MCP 客户端测试
pytest tests/e2e/test_mcp_client.py -v

# Dashboard 冒烟测试
pytest tests/e2e/test_dashboard_smoke.py -v

# Recall 回归测试
pytest tests/e2e/test_recall.py -v
```

### 运行所有测试

```bash
# 详细输出
pytest -v

# 快速输出
pytest -q

# 显示最慢的 10 个测试
pytest --durations=10

# 并行运行（需要 pytest-xdist）
pytest -n auto
```

### 测试覆盖率报告

```bash
# 生成 HTML 覆盖率报告
pytest --cov=src --cov-report=html

# 查看报告
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
xdg-open htmlcov/index.html # Linux
```

---

## ❓ 常见问题

### API Key 配置

**Q: 如何安全地设置 API Key？**

A: 建议使用环境变量而不是直接在配置文件中存储：
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxx"
```

然后在 `settings.yaml` 中引用：
```yaml
llm:
  api_key: ${OPENAI_API_KEY}
```

**Q: 支持哪些 LLM 提供商？**

A: 支持以下提供商：
- OpenAI（gpt-4, gpt-4o-mini, gpt-3.5-turbo）
- Azure OpenAI
- Ollama（本地 LLM）
- DeepSeek
- Qwen（通义千问）
- 更多提供商可通过扩展添加

**Q: 如何切换不同的 LLM？**

A: 直接修改 `settings.yaml` 中的 `llm` 配置，无需重启应用。

### 依赖安装

**Q: 安装时出现 ImportError？**

A: 确保在虚拟环境中安装：
```bash
# 检查当前环境
which python  # macOS/Linux
where python  # Windows

# 如果不是虚拟环境中的 python，激活虚拟环境
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\activate   # Windows
```

**Q: 某些可选依赖缺失？**

A: 安装开发依赖：
```bash
pip install -e ".[dev]"
```

**Q: 如何升级依赖？**

A: 
```bash
pip install --upgrade -e .
```

### 连接问题

**Q: MCP Server 无法启动？**

A: 检查日志：
```bash
python -m src.mcp_server 2>&1 | head -20
```

常见原因：
- 配置文件不存在 → 运行 `cp config/settings.yaml config/settings.local.yaml`
- API Key 无效 → 检查 `settings.yaml` 中的 `api_key`
- 端口被占用 → 修改 Dashboard 端口配置

**Q: Dashboard 页面加载缓慢？**

A: 
1. 检查网络连接
2. 减少追踪数据量：`observability.log_level: ERROR`
3. 清理旧日志：`rm -rf ./logs/`

**Q: 查询返回结果为空？**

A: 
1. 确认文档已摄取：运行 `python scripts/query.py --query "test" --verbose`
2. 检查向量存储：`ls -la ./data/db/chroma/`
3. 调整检索参数：`retrieval.top_k_final` 设置更大的值

### 性能优化

**Q: 如何加快检索速度？**

A: 
1. 减小 `chunk_size`（但要保证语义完整）
2. 使用更快的嵌入模型（如 `text-embedding-3-small`）
3. 调整 `top_k_dense` 和 `top_k_sparse` 为更小的值
4. 启用重排缓存（如果支持）

**Q: 如何减少 API 调用成本？**

A: 
1. 使用本地 LLM（Ollama）而不是 OpenAI
2. 减少 `temperature` 值以降低计算成本
3. 使用缓存机制存储常见查询结果
4. 批量处理而不是逐个查询

---

## 📁 项目结构

```
Modular-RAG/
├── README.md                          # 本文件
├── DEV_SPEC.md                       # 详细设计文档
├── pyproject.toml                    # Python 项目配置
├── config/
│   ├── settings.yaml                 # 默认配置
│   └── prompts/                      # Prompt 模板
├── src/
│   ├── core/                         # 核心模块
│   │   ├── settings.py               # 配置加载
│   │   ├── types.py                  # 类型定义
│   │   ├── document_manager.py       # 文档管理
│   │   └── query_engine/             # 查询引擎
│   ├── libs/                         # 可插拔库
│   │   ├── llm/                      # LLM 实现
│   │   ├── embedding/                # 嵌入模型
│   │   ├── vector_store/             # 向量存储
│   │   ├── reranker/                 # 重排器
│   │   └── evaluator/                # 评估器
│   ├── ingestion/                    # 摄取流程
│   ├── mcp_server/                   # MCP 服务器
│   │   ├── server.py                 # 主服务器
│   │   ├── protocol_handler.py       # 协议处理
│   │   └── tools/                    # MCP 工具
│   └── observability/                # 可观测性
│       ├── logger.py                 # 日志
│       ├── tracer.py                 # 追踪
│       └── dashboard/                # Web Dashboard
├── tests/
│   ├── unit/                         # 单元测试
│   ├── integration/                  # 集成测试
│   ├── e2e/                          # E2E 测试
│   ├── fixtures/                     # 测试数据
│   └── conftest.py                   # Pytest 配置
├── scripts/
│   ├── ingest.py                     # 摄取脚本
│   ├── query.py                      # 查询脚本
│   └── evaluate.py                   # 评估脚本
└── logs/                             # 运行日志（自动生成）
```

---

## 🔗 相关资源

- **项目主页**: https://github.com/Mozinapig/Modular-RAG
- **设计文档**: [DEV_SPEC.md](./DEV_SPEC.md)
- **MCP 规范**: https://modelcontextprotocol.io/
- **Streamlit 文档**: https://docs.streamlit.io/

---

## 📝 许可证

MIT License - 详见 [LICENSE](./LICENSE) 文件

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交改动 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 💬 问题反馈

如有问题或建议，请在 GitHub Issues 中提出。

**更新时间**: 2026-06-18  
**当前版本**: v1.0.0-rc1  
**开发状态**: 阶段 I - 端到端验收与文档收口
