# AnySearch 集成说明

## 1. 内部工作方式

本地安装的 AnySearch skill 不是一个需要常驻进程的 MCP server，而是一组跨平台 CLI 和说明文件。Python CLI 的核心流程是：

1. 读取 `ANYSEARCH_API_KEY`。优先级为显式参数、skill 目录 `.env`、系统环境变量，最后允许匿名访问。
2. 将操作转换成 JSON-RPC 2.0 请求：`method` 为 `tools/call`，具体能力放在 `params.name`，输入放在 `params.arguments`。
3. 向 `https://api.anysearch.com/mcp` 发起 HTTP POST。
4. 使用 `Authorization: Bearer <key>` 和 `X-Anysearch-Client: skill/3.0.1` 请求头。
5. 从 `result.content` 中提取 `type=text` 的内容；搜索结果通常是文本/Markdown。

支持的主要能力：

- `search`：普通搜索或垂直领域搜索。
- `batch_search`：一次提交 1–5 个查询。
- `get_sub_domains`：在 finance、academic、health、code 等垂直领域搜索前发现可用子域和参数模式。
- `extract`：抓取并提取网页正文为 Markdown。

垂直搜索应先调用 `get_sub_domains`，再把返回的 `sub_domain` 和必需参数传给 `search`。

## 2. 直接从 Python 调用

AnySearch 自带 CLI，但应用代码可以直接使用相同的 JSON-RPC 协议：

```python
import os
import requests

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "search",
        "arguments": {
            "query": "B2B SaaS marketing trends",
            "max_results": 5,
        },
    },
}

headers = {
    "Content-Type": "application/json",
    "X-Anysearch-Client": "skill/3.0.1",
}
if os.getenv("ANYSEARCH_API_KEY"):
    headers["Authorization"] = f"Bearer {os.environ['ANYSEARCH_API_KEY']}"

response = requests.post(
    "https://api.anysearch.com/mcp",
    json=payload,
    headers=headers,
    timeout=30,
)
response.raise_for_status()
print(response.json())
```

本项目的 [tools/anysearch_tool.py](../tools/anysearch_tool.py) 已封装了认证、请求、错误处理和 LangChain tool schema，研究 agent 不需要重复编写这些细节。

## 3. 在 LangGraph 中使用

```python
from langgraph.prebuilt import create_react_agent

from tools.anysearch_tool import ANYSEARCH_TOOLS

research_agent = create_react_agent(
    model,
    tools=ANYSEARCH_TOOLS,
)
```

如果使用自定义 StateGraph，可将 `ANYSEARCH_TOOLS` 交给 ToolNode：

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(ANYSEARCH_TOOLS)
```

工具封装只负责搜索提供商调用；研究任务拆解、来源筛选、事实核验和报告生成应放在后续 agent/workflow 层。

